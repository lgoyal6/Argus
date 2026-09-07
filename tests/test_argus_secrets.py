"""Tests for backend/argus_secrets.py.

Runs on the standard library alone: `python3 tests/test_argus_secrets.py`, and
under `pytest tests/` in CI, which is the run that must not stop happening.
The Secret Manager client is injected, so no dependency and no live project.

Every test that asserts a denial has a matching test asserting the same code
path can succeed, so a test that passes because nothing ran is visible.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from argus_secrets import (  # noqa: E402
    SecretAccessDenied,
    SecretNotConfigured,
    SecretResolver,
)

SECRET_VALUE = "sk-live-EXAMPLE-must-never-be-printed"


class FakePayload:
    def __init__(self, data):
        self.data = data


class FakeResponse:
    def __init__(self, data):
        self.payload = FakePayload(data)


class FakeClient:
    """Records what was asked for and returns what it was told to."""

    def __init__(self, contents=None, raises=None):
        self.contents = contents or {}
        self.raises = raises
        self.requested = []

    def access_secret_version(self, request):
        name = request["name"]
        self.requested.append(name)
        if self.raises is not None:
            raise self.raises
        if name not in self.contents:
            raise LookupError(name)
        return FakeResponse(self.contents[name])


class PermissionDenied(Exception):
    """Shaped like google.api_core.exceptions.PermissionDenied."""

    code = 403


IDS = {"SUPABASE_KEY": "argus-supabase-key", "ANTHROPIC_API_KEY": "argus-model-key"}
PROJECT = "test-project"
SUPA = f"projects/{PROJECT}/secrets/argus-supabase-key/versions/latest"
MODEL = f"projects/{PROJECT}/secrets/argus-model-key/versions/latest"


def resolver(client, env=None, **kw):
    return SecretResolver(
        project=PROJECT,
        secret_ids=IDS,
        client_factory=lambda: client,
        env=env if env is not None else {},
        **kw,
    )


class ReadsFromSecretManager(unittest.TestCase):
    def test_reads_the_configured_secret(self):
        c = FakeClient({SUPA: SECRET_VALUE})
        self.assertEqual(resolver(c).get("SUPABASE_KEY"), SECRET_VALUE)
        self.assertEqual(c.requested, [SUPA])

    def test_decodes_bytes_payloads(self):
        c = FakeClient({MODEL: SECRET_VALUE.encode("utf-8")})
        self.assertEqual(resolver(c).get("ANTHROPIC_API_KEY"), SECRET_VALUE)

    def test_secret_manager_wins_over_a_stale_environment_value(self):
        # The whole point: an environment value left over from an old image
        # must not shadow the one the platform is serving.
        c = FakeClient({SUPA: "from-secret-manager"})
        r = resolver(c, env={"SUPABASE_KEY": "stale-baked-into-the-image"})
        self.assertEqual(r.get("SUPABASE_KEY"), "from-secret-manager")

    def test_each_secret_is_addressed_by_its_own_resource_name(self):
        # A workload granted accessor on one secret must not be reaching for
        # the other one's resource name.
        c = FakeClient({SUPA: "a", MODEL: "b"})
        r = resolver(c)
        r.get("SUPABASE_KEY")
        r.get("ANTHROPIC_API_KEY")
        self.assertEqual(c.requested, [SUPA, MODEL])
        self.assertNotIn("argus-model-key", SUPA)


class DenialIsLoud(unittest.TestCase):
    def test_permission_denied_raises_and_does_not_fall_back(self):
        c = FakeClient(raises=PermissionDenied("denied"))
        r = resolver(c, env={"SUPABASE_KEY": "stale-value-still-in-the-env"})
        with self.assertRaises(SecretAccessDenied):
            r.get("SUPABASE_KEY")

    def test_control_same_resolver_succeeds_when_access_is_granted(self):
        # Negative control for the test above: without the denial, this exact
        # configuration returns a value, so the assertRaises is really being
        # driven by the denial and not by a broken resolver.
        c = FakeClient({SUPA: "granted"})
        r = resolver(c, env={"SUPABASE_KEY": "stale-value-still-in-the-env"})
        self.assertEqual(r.get("SUPABASE_KEY"), "granted")

    def test_an_empty_secret_version_is_an_error(self):
        c = FakeClient({SUPA: "   "})
        with self.assertRaises(SecretNotConfigured):
            resolver(c).get("SUPABASE_KEY")


class NeverLeaksTheValue(unittest.TestCase):
    def _all_message_text(self, exc):
        return " ".join(str(a) for a in exc.args) + " " + repr(exc)

    def test_denial_message_carries_the_resource_name_not_the_value(self):
        c = FakeClient(raises=PermissionDenied("denied"))
        r = resolver(c, env={"SUPABASE_KEY": SECRET_VALUE})
        with self.assertRaises(SecretAccessDenied) as ctx:
            r.get("SUPABASE_KEY")
        text = self._all_message_text(ctx.exception)
        self.assertNotIn(SECRET_VALUE, text)
        self.assertIn(SUPA, text)

    def test_underlying_exception_is_not_chained_into_the_traceback(self):
        # `raise ... from None`: the provider error can carry request context,
        # and a chained traceback would print it wherever the error is logged.
        c = FakeClient(raises=PermissionDenied(f"failed request with {SECRET_VALUE}"))
        r = resolver(c)
        with self.assertRaises(SecretAccessDenied) as ctx:
            r.get("SUPABASE_KEY")
        self.assertIsNone(ctx.exception.__cause__)
        self.assertIsNone(ctx.exception.__context__)

    def test_the_rendered_traceback_contains_no_secret_value(self):
        # The end-to-end version of the two tests above: format the traceback
        # the way a crash handler would and search the whole thing.
        import traceback

        c = FakeClient(raises=PermissionDenied(f"failed request with {SECRET_VALUE}"))
        r = resolver(c, env={"SUPABASE_KEY": SECRET_VALUE})
        try:
            r.get("SUPABASE_KEY")
        except SecretAccessDenied as exc:
            rendered = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        self.assertNotIn(SECRET_VALUE, rendered)
        self.assertIn("SecretAccessDenied", rendered)

    def test_missing_value_message_carries_no_value(self):
        r = SecretResolver(project="", secret_ids=IDS, env={})
        with self.assertRaises(SecretNotConfigured) as ctx:
            r.get("SUPABASE_KEY")
        self.assertNotIn(SECRET_VALUE, self._all_message_text(ctx.exception))

    def test_control_the_value_is_detectable_when_it_is_present(self):
        # Negative control for the three tests above: the assertion they use
        # does fire when a value really is in the text.
        self.assertIn(SECRET_VALUE, self._all_message_text(RuntimeError(SECRET_VALUE)))


class CachingAndRotation(unittest.TestCase):
    def test_repeat_reads_inside_the_ttl_do_not_re_fetch(self):
        c = FakeClient({SUPA: "v1"})
        now = [1000.0]
        r = resolver(c, ttl_seconds=300, clock=lambda: now[0])
        r.get("SUPABASE_KEY")
        r.get("SUPABASE_KEY")
        self.assertEqual(len(c.requested), 1)

    def test_a_rotated_value_is_picked_up_once_the_ttl_lapses(self):
        c = FakeClient({SUPA: "v1"})
        now = [1000.0]
        r = resolver(c, ttl_seconds=300, clock=lambda: now[0])
        self.assertEqual(r.get("SUPABASE_KEY"), "v1")
        c.contents[SUPA] = "v2"
        self.assertEqual(r.get("SUPABASE_KEY"), "v1", "still cached before the TTL lapses")
        now[0] += 301
        self.assertEqual(r.get("SUPABASE_KEY"), "v2")

    def test_invalidate_forces_an_immediate_re_read(self):
        c = FakeClient({SUPA: "v1"})
        r = resolver(c, ttl_seconds=10_000)
        r.get("SUPABASE_KEY")
        c.contents[SUPA] = "v2"
        r.invalidate("SUPABASE_KEY")
        self.assertEqual(r.get("SUPABASE_KEY"), "v2")


class LocalDevelopmentStillWorks(unittest.TestCase):
    def test_without_a_project_the_environment_is_used(self):
        r = SecretResolver(project="", secret_ids=IDS, env={"SUPABASE_KEY": "local"})
        self.assertFalse(r.uses_secret_manager)
        self.assertEqual(r.get("SUPABASE_KEY"), "local")

    def test_no_secret_manager_client_is_built_in_local_mode(self):
        def explode():
            raise AssertionError("built a Secret Manager client without a project")

        r = SecretResolver(
            project="", secret_ids=IDS, env={"SUPABASE_KEY": "local"}, client_factory=explode
        )
        self.assertEqual(r.get("SUPABASE_KEY"), "local")

    def test_a_missing_local_value_names_both_ways_to_supply_it(self):
        r = SecretResolver(project="", secret_ids=IDS, env={})
        with self.assertRaises(SecretNotConfigured) as ctx:
            r.get("SUPABASE_KEY")
        self.assertIn("ARGUS_SECRET_PROJECT", str(ctx.exception))


class UnknownNames(unittest.TestCase):
    def test_an_unmapped_name_is_rejected_rather_than_guessed(self):
        r = resolver(FakeClient({}))
        with self.assertRaises(SecretNotConfigured):
            r.get("SOME_OTHER_KEY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
