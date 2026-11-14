"""Runtime secret resolution for Argus.

Named argus_secrets rather than secrets: both Dockerfiles flatten their source
directory onto the working directory, which is first on sys.path, so a module
called secrets.py here would shadow the standard library's secrets module for
every dependency in the process.

Argus used to read every credential straight out of the process environment,
which in the deployed path meant a .env file copied next to the code. That
makes the secret an artifact: it lands in the image layer, in `docker inspect`,
in anything that dumps the environment, and it can only be changed by
rebuilding and redeploying.

This module resolves the same names from Google Secret Manager instead, using
Application Default Credentials so the workload authenticates as its own
service account and never holds a downloaded key. Each secret is named
individually, so the service account can be granted
`roles/secretmanager.secretAccessor` on exactly the secrets it owns and on
nothing else.

Two rules the tests pin down, because both are easy to get wrong in a way that
silently removes the protection:

* A Secret Manager lookup that fails with a permission error is raised, not
  swallowed. Falling back to the environment on a denial would turn a
  revocation into a silent downgrade to whatever stale value the environment
  still held.
* Secret values never appear in exception messages, reprs, or log output. Only
  the resource name does.

Local development keeps working: with no `ARGUS_SECRET_PROJECT` configured the
resolver reads the environment exactly as before.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Dict, Optional, Tuple

__all__ = [
    "SecretNotConfigured",
    "SecretAccessDenied",
    "SecretResolver",
    "default_resolver",
    "get_secret",
]

# Environment variable -> Secret Manager secret id. The environment variable
# name stays the same as before so local development and the tests are
# unchanged; only where the value comes from in a deployment changes.
SECRET_IDS: Dict[str, str] = {
    "SUPABASE_KEY": "argus-supabase-key",
    "ANTHROPIC_API_KEY": "argus-model-key",
}

# How long a resolved value is reused before Secret Manager is consulted again.
# Bounded so a rotation is picked up by a running process; short enough to
# matter, long enough that a request path is not one RPC per call.
DEFAULT_TTL_SECONDS = 300


class SecretNotConfigured(RuntimeError):
    """A required secret has no value in Secret Manager or the environment."""


class SecretAccessDenied(RuntimeError):
    """Secret Manager refused the read.

    Raised rather than falling back, so a revoked or misgranted identity fails
    loudly instead of quietly continuing on an environment value.
    """


def _redact(name: str) -> str:
    """Everything this module puts in a message goes through here."""
    return f"<secret {name}>"


class SecretResolver:
    """Resolves named secrets, preferring Secret Manager over the environment.

    `client_factory` and `clock` exist so the resolver can be tested without
    the google-cloud-secret-manager dependency or a live project.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        secret_ids: Optional[Dict[str, str]] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        client_factory: Optional[Callable[[], object]] = None,
        env: Optional[Dict[str, str]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._env = env if env is not None else os.environ
        self._project = project if project is not None else self._env.get("ARGUS_SECRET_PROJECT", "")
        self._secret_ids = dict(secret_ids if secret_ids is not None else SECRET_IDS)
        self._ttl = ttl_seconds
        self._client_factory = client_factory or _build_google_client
        self._clock = clock
        self._lock = threading.Lock()
        self._client: Optional[object] = None
        self._cache: Dict[str, Tuple[float, str]] = {}

    @property
    def uses_secret_manager(self) -> bool:
        return bool(self._project)

    def resource_name(self, name: str) -> str:
        secret_id = self._secret_ids.get(name)
        if secret_id is None:
            raise SecretNotConfigured(f"{name} has no Secret Manager mapping")
        return f"projects/{self._project}/secrets/{secret_id}/versions/latest"

    def invalidate(self, name: Optional[str] = None) -> None:
        with self._lock:
            if name is None:
                self._cache.clear()
            else:
                self._cache.pop(name, None)

    def get(self, name: str) -> str:
        if not self.uses_secret_manager:
            return self._from_env(name)

        with self._lock:
            cached = self._cache.get(name)
            if cached is not None and self._clock() < cached[0]:
                return cached[1]

        value = self._from_secret_manager(name)

        with self._lock:
            self._cache[name] = (self._clock() + self._ttl, value)
        return value

    # -- sources ---------------------------------------------------------

    def _from_env(self, name: str) -> str:
        value = self._env.get(name)
        if not value:
            raise SecretNotConfigured(
                f"{name} is not set. Set it in the environment for local use, "
                f"or set ARGUS_SECRET_PROJECT to read {_redact(name)} from Secret Manager."
            )
        return value

    def _from_secret_manager(self, name: str) -> str:
        resource = self.resource_name(name)
        with self._lock:
            if self._client is None:
                self._client = self._client_factory()
            client = self._client
        failure: Optional[Exception] = None
        try:
            response = client.access_secret_version(request={"name": resource})
        except Exception as exc:  # noqa: BLE001 - classified here, raised below
            if _is_permission_error(exc):
                # Deliberately not falling back to the environment: a denial is
                # the revocation working, and it has to be visible.
                failure = SecretAccessDenied(
                    f"Secret Manager refused {resource} for this workload identity. "
                    "Grant roles/secretmanager.secretAccessor on that secret, or "
                    "unset ARGUS_SECRET_PROJECT to use environment values."
                )
            else:
                failure = SecretNotConfigured(
                    f"could not read {resource}: {type(exc).__name__}"
                )
        if failure is not None:
            # Raised outside the except block on purpose. `raise ... from None`
            # only sets __suppress_context__; the provider's exception stays
            # reachable on __context__, and a provider error can quote the
            # request it failed on. Anything that walks the exception chain -
            # a structured log formatter, an error reporter - would then print
            # it. Raising here leaves the chain empty.
            raise failure

        payload = response.payload.data
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        payload = payload.strip()
        if not payload:
            raise SecretNotConfigured(f"{resource} holds an empty value")
        return payload


def _is_permission_error(exc: Exception) -> bool:
    """True for a Secret Manager denial, without importing google.api_core."""
    if getattr(exc, "code", None) in (403, 7):  # HTTP 403 / gRPC PERMISSION_DENIED
        return True
    name = type(exc).__name__
    if name in ("PermissionDenied", "Forbidden", "Unauthenticated", "Unauthorized"):
        return True
    return False


def _build_google_client():  # pragma: no cover - needs the real dependency
    from google.cloud import secretmanager

    return secretmanager.SecretManagerServiceClient()


_default: Optional[SecretResolver] = None
_default_lock = threading.Lock()


def default_resolver() -> SecretResolver:
    global _default
    with _default_lock:
        if _default is None:
            _default = SecretResolver()
        return _default


def get_secret(name: str) -> str:
    return default_resolver().get(name)
