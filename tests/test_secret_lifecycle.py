"""Where the Supabase and Anthropic credentials can and cannot end up.

Four questions, each one a place a secret leaves the process:

* the working tree and the ignore rules that keep it out of one,
* the git history, which is the one that cannot be fixed by an edit,
* logs, tracebacks and API error responses,
* and whether a rotated credential actually takes effect, or whether the old one is
  cached for the lifetime of the process and the rotation silently does nothing.

The history check is the one that found something. It is written so that it fails on
the repository as it stands, because the finding is real and unresolved: rewriting
published history and rotating a live credential are both the owner's decisions, not a
test's. See RECORD_argus_contracts.md.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import obs  # noqa: E402

# Shapes, not values. A test that carries the credential it is looking for is itself a
# place the credential is stored.
SECRET_SHAPES = {
    "supabase anon or service JWT":
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "anthropic api key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "supabase project url": re.compile(r"https://[a-z0-9]{16,}\.supabase\.co"),
}

# .env.example is a template and must stay readable; the placeholder value "key" is not
# a credential. Lockfiles carry base64 integrity hashes that are not JWTs but can look
# like one to a loose pattern, so they are matched with the strict three-segment form
# above rather than excluded.
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
             ".agent-work", "dist", "build"}


# A test that proves a scrubber works has to contain strings of exactly the shape the
# scrubber removes. Those lines carry this marker. It is per line and has to be typed
# out, so a real credential pasted anywhere in the tree still fails the scan below.
FIXTURE_MARKER = "credential-shape-fixture"


def _in_a_git_checkout():
    """A source archive is not a checkout. `git ls-files` exits 128 there, and without
    this the git-backed cases fail as `.env is not ignored`, which is not what broke."""
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--git-dir"],
                       capture_output=True, text=True)
    return r.returncode == 0


needs_git = pytest.mark.skipif(not _in_a_git_checkout(),
                               reason="not a git checkout; git ls-files/check-ignore cannot run")


def _scan_lines(text):
    """Line-scoped scan that honours FIXTURE_MARKER. The shapes cannot span a newline,
    so this is exactly as strict as scanning the whole text."""
    kinds = []
    for line in text.splitlines():
        if FIXTURE_MARKER in line:
            continue
        for kind in _scan(line):
            if kind not in kinds:
                kinds.append(kind)
    return kinds


def _tracked_files():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                         capture_output=True, text=True, check=True)
    return [ROOT / line for line in out.stdout.splitlines() if line]


def _scan(text):
    return [name for name, pattern in SECRET_SHAPES.items() if pattern.search(text)]


# -- 1. the working tree --------------------------------------------------------
@needs_git
def test_no_credential_shape_appears_in_any_tracked_file():
    offenders = []
    for path in _tracked_files():
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:  # pragma: no cover
            continue
        for kind in _scan_lines(text):
            offenders.append(f"{path.relative_to(ROOT)}: {kind}")
    assert not offenders, "credential-shaped strings in tracked files:\n  " + "\n  ".join(offenders)


def test_the_example_env_file_carries_placeholders_only():
    text = (ROOT / ".env.example").read_text()
    assert not _scan(text), ".env.example carries a real credential shape"
    for var in ("SUPABASE_URL", "SUPABASE_KEY", "ANTHROPIC_API_KEY"):
        assert var in text, f"{var} missing from the template"


@needs_git
@pytest.mark.parametrize("candidate", [
    ".env", ".env.local", ".env.production", ".env.staging",
    "server.pem", "private.key", "cert.p12", "credentials.json", "secrets.yaml",
])
def test_git_ignores_every_shape_of_key_material(candidate):
    """`.env` alone left every other variant tracked."""
    r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", candidate])
    assert r.returncode == 0, f"{candidate} is not ignored by .gitignore"


@needs_git
def test_the_example_template_is_deliberately_not_ignored():
    """The negative control for the rule above: the negation has to actually work."""
    r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", ".env.example"])
    assert r.returncode != 0, ".env.example must stay tracked; the !negation broke"


# -- 2. the built artifact ------------------------------------------------------
def test_the_docker_build_context_excludes_credentials_and_history():
    """No .dockerignore existed, so `context: .` shipped .env and .git to the daemon.

    Checked here as rules rather than by building, so it runs in CI; the build itself
    was run once and is recorded in RECORD_argus_contracts.md (36,025 context entries
    down to 96, and the planted sentinel gone).
    """
    ignore = (ROOT / ".dockerignore")
    assert ignore.exists(), "docker-compose builds every service with `context: .`"
    rules = {line.strip() for line in ignore.read_text().splitlines()
             if line.strip() and not line.startswith("#")}
    for required in (".env", ".env.*", ".git", ".venv", "node_modules", ".agent-work"):
        assert required in rules, f".dockerignore does not exclude {required}"
    assert "!.env.example" in rules, "the template must stay available to the build"


def test_no_dockerfile_copies_the_whole_context():
    """`.dockerignore` is the belt; not copying the world is the braces.

    Neither alone is sufficient: a `COPY . .` with a stale ignore file is how a
    credential reaches a published layer.
    """
    for dockerfile in sorted((ROOT / "docker").glob("Dockerfile.*")):
        for line in dockerfile.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("COPY"):
                parts = stripped.split()
                assert parts[1] not in (".", "./"), \
                    f"{dockerfile.name} copies the entire build context: {stripped}"


# -- 3. logs, tracebacks and API error responses --------------------------------
def test_an_api_failure_does_not_relay_exception_text_to_the_caller(client, store, monkeypatch):
    """`detail=str(e)` was an unconditional relay of whatever the exception carried."""
    import db

    sentinel = "SENTINEL-SUPABASE-KEY-eyJhbGciOiJIUzI1NiJ9.aaaaaaaaaa.bbbbbbbbbb"

    def boom():
        raise RuntimeError(f"connection refused for {sentinel} at /Users/someone/Argus")

    monkeypatch.setattr(db, "get_client", boom)
    r = client.get("/runs/")
    assert r.status_code == 500
    body = r.text
    assert sentinel not in body, f"the exception text reached the caller: {body}"
    assert "/Users/" not in body, f"a host path reached the caller: {body}"
    assert "RuntimeError" not in body
    # What the caller does get: a stable shape and a correlation id, so a report is
    # actionable without being a disclosure. The shape is flat because that is what
    # the committed contract declares for 500; it used to be nested one level deeper,
    # which matched no schema and hid error_id from any client reading the document.
    payload = r.json()
    assert payload["detail"] == "internal error"
    assert len(payload["error_id"]) == 12


def test_the_correlation_id_in_the_response_matches_the_one_in_the_log(client, store, monkeypatch):
    import db

    monkeypatch.setattr(db, "get_client",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with obs.capture() as records:
        r = client.get("/runs/")
    error_id = r.json()["error_id"]
    logged = [rec for rec in records if rec["event"] == "api.request_failed"]
    assert len(logged) == 1
    assert logged[0]["error_id"] == error_id
    assert logged[0]["error_type"] == "RuntimeError"
    assert logged[0]["operation"] == "get_runs"


@pytest.mark.parametrize("var,value", [
    ("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiJ9.cnVudGltZQ.c2lnbmF0dXJl"),
    ("ANTHROPIC_API_KEY", "sk-ant-api03-RUNTIMESENTINELVALUE0000"),  # credential-shape-fixture
    ("SUPABASE_URL", "https://runtimesentinelref.supabase.co"),  # credential-shape-fixture
])
def test_a_live_credential_never_survives_a_log_record(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with obs.capture() as records:
        obs.log("probe", detail=f"the client said: {value}")
        obs.log("probe2", detail=f"POST {value}/rest/v1/metrics failed")
    rendered = json.dumps(records)
    assert value not in rendered, rendered


def test_the_agent_logger_scrubs_a_failed_supabase_write(monkeypatch, tmp_path):
    """agent/logger.py printed the raw exception on every failed write."""
    sys.path.insert(0, str(ROOT / "agent"))
    from agent import logger as agent_logger

    secret = "eyJhbGciOiJIUzI1NiJ9.bG9nZ2Vy.c2ln"
    monkeypatch.setattr(agent_logger, "_run_id", "run-x")
    monkeypatch.setattr(agent_logger, "get_supabase",
                        lambda: (_ for _ in ()).throw(RuntimeError(f"401 for {secret}")))
    monkeypatch.chdir(tmp_path)

    with obs.capture() as records:
        agent_logger._write_supabase({"timestamp": 1.0, "anomalies": [], "tools_used": [],
                                      "agent_response": "x", "fixed": False,
                                      "status": "failed", "attempt": None})
    rendered = json.dumps(records)
    assert secret not in rendered, rendered
    assert obs.REDACTED in rendered


def test_the_agent_never_logs_configuration_contents():
    """read_config returns the whole training config; the loop used to print it."""
    config = {"training": {"learning_rate": 0.001, "batch_size": 64},
              "monitoring": {"metrics_file": "metrics/metrics.jsonl"}}
    with obs.capture() as records:
        obs.log("recovery.tool_result", tool="read_config", refused=False, payload=config)
    rendered = json.dumps(records)
    assert "learning_rate" not in rendered and "0.001" not in rendered, rendered
    assert records[0]["tool"] == "read_config"


# -- 4. rotation ----------------------------------------------------------------
def test_a_rotated_credential_takes_effect_without_restarting_the_process(monkeypatch):
    """No client is cached, so the next call uses the current environment.

    This is the property that was already right and is worth pinning: a `get_client`
    that memoised its result would hold a revoked key for the lifetime of the process,
    and a rotation would appear to have worked while changing nothing.
    """
    sys.path.insert(0, str(ROOT / "backend"))
    import db

    seen = []
    monkeypatch.setattr(db, "create_client", lambda url, key: seen.append((url, key)))
    monkeypatch.setenv("SUPABASE_URL", "https://one.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "key-one")
    db.get_client()
    monkeypatch.setenv("SUPABASE_KEY", "key-two")   # rotated
    db.get_client()
    assert [k for _, k in seen] == ["key-one", "key-two"], \
        "the second call reused the first credential; the client is cached"


def test_the_agent_logger_also_rereads_the_credential_on_every_call(monkeypatch):
    from agent import logger as agent_logger

    seen = []
    monkeypatch.setattr(agent_logger, "create_client", lambda url, key: seen.append(key))
    monkeypatch.setenv("SUPABASE_URL", "https://one.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "key-one")
    agent_logger.get_supabase()
    monkeypatch.setenv("SUPABASE_KEY", "key-two")
    agent_logger.get_supabase()
    assert seen == ["key-one", "key-two"]


def test_a_revoked_credential_surfaces_as_a_failure_rather_than_a_silent_no_op(client, store, monkeypatch):
    """Revocation has to be visible. A swallowed auth error is a silent data loss."""
    import db

    monkeypatch.setattr(db, "get_client", lambda: (_ for _ in ()).throw(
        RuntimeError("401 Unauthorized: JWT expired")))
    r = client.get("/runs/")
    assert r.status_code == 500, "a revoked credential returned a success status"
    assert "JWT expired" not in r.text


def test_missing_credentials_are_refused_without_naming_their_values(monkeypatch):
    sys.path.insert(0, str(ROOT / "backend"))
    import db

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    with pytest.raises(ValueError) as e:
        db.get_client()
    assert "SUPABASE_URL" in str(e.value)
    assert not _scan(str(e.value))
