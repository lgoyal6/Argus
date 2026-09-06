"""Shared fixtures for the API contract and boundary-input tests.

Two things matter here and both are safety properties rather than conveniences.

**The app is loaded in the FLAT layout the container actually runs.**
`docker/Dockerfile.backend` does `COPY backend/ .`, so inside the image there is no
`backend` package: `main.py` sits at `/app/main.py` and imports `from routes import
runs`. Loading the app here as `backend.main` would exercise an import graph that does
not exist in production, so `backend/` goes on `sys.path` and the app is imported flat.

**No test may reach a real Supabase project.** The generated-input tests send malformed
and destructive bodies on purpose. `db.get_client` is replaced with an in-memory store,
and `supabase.create_client` is additionally poisoned so that any code path that
bypasses the fixture fails loudly instead of quietly opening a network client against
whatever `SUPABASE_URL` happens to be in the environment.
"""

import copy
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

for p in (str(ROOT), str(BACKEND)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _install_supabase_stub():
    """Stand in for the supabase package when it is not installed.

    The backend's logic under test is routing, validation and error shaping, none of
    which need a real client. CI installs pytest, pyyaml, fastapi and httpx only.
    """
    if "supabase" in sys.modules:
        return
    try:
        import supabase  # noqa: F401
        return
    except ImportError:
        pass
    stub = types.ModuleType("supabase")

    class Client:  # pragma: no cover - a name for type annotations only
        pass

    def create_client(url, key):  # pragma: no cover - poisoned below anyway
        raise AssertionError("create_client must never be called from the test suite")

    stub.Client = Client
    stub.create_client = create_client
    sys.modules["supabase"] = stub


_install_supabase_stub()


# ── an in-memory stand-in for the supabase query builder ───────────────────────
class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, rows, op, payload=None, on_conflict=None):
        self._rows = rows
        self._op = op
        self._payload = payload
        self._on_conflict = on_conflict
        self._filters = []
        self._order = None

    def eq(self, col, val):
        self._filters.append((col, val))
        return self

    def order(self, col, desc=False):
        self._order = (col, desc)
        return self

    def _matching(self):
        out = self._rows
        for col, val in self._filters:
            out = [r for r in out if r.get(col) == val]
        return out

    def execute(self):
        if self._op == "select":
            rows = self._matching()
            if self._order:
                col, desc = self._order
                rows = sorted(rows, key=lambda r: r.get(col), reverse=desc)
            return _Result(copy.deepcopy(rows))
        if self._op == "insert":
            items = self._payload if isinstance(self._payload, list) else [self._payload]
            self._rows.extend(copy.deepcopy(items))
            return _Result(copy.deepcopy(items))
        if self._op == "update":
            hit = self._matching()
            for r in hit:
                r.update(copy.deepcopy(self._payload))
            return _Result(copy.deepcopy(hit))
        if self._op == "upsert":
            keys = [k.strip() for k in (self._on_conflict or "").split(",") if k.strip()]
            items = self._payload if isinstance(self._payload, list) else [self._payload]
            for it in items:
                existing = None
                if keys:
                    for r in self._rows:
                        if all(r.get(k) == it.get(k) for k in keys):
                            existing = r
                            break
                if existing is not None:
                    existing.update(copy.deepcopy(it))
                else:
                    self._rows.append(copy.deepcopy(it))
            return _Result(copy.deepcopy(items))
        raise AssertionError(f"unsupported operation {self._op}")


class _Table:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *_a, **_k):
        return _Query(self._rows, "select")

    def insert(self, payload):
        return _Query(self._rows, "insert", payload)

    def update(self, payload):
        return _Query(self._rows, "update", payload)

    def upsert(self, payload, on_conflict=None):
        return _Query(self._rows, "upsert", payload, on_conflict)


class FakeSupabase:
    """Isolated store. Nothing here touches a network or a real project."""

    def __init__(self):
        self.tables = {"runs": [], "metrics": [], "decisions": []}

    def table(self, name):
        return _Table(self.tables.setdefault(name, []))


SEED_RUN_ID = "11111111-1111-1111-1111-111111111111"
OTHER_RUN_ID = "22222222-2222-2222-2222-222222222222"


def seed(store, tmp_path):
    """Two runs, so that a request scoped to one can be checked against the other."""
    for rid, name, created in ((SEED_RUN_ID, "seeded-run", 100.0),
                               (OTHER_RUN_ID, "other-run", 200.0)):
        store.tables["runs"].append({
            "id": rid,
            "name": name,
            "config_path": str(tmp_path / "config.yaml"),
            "metrics_file": str(tmp_path / f"{name}.jsonl"),
            "training_dir": str(tmp_path),
            "status": "running",
            "created_at": created,
            "updated_at": created,
        })
    for step in range(3):
        store.tables["metrics"].append({
            "id": f"m{step}", "run_id": SEED_RUN_ID, "step": step, "epoch": 0,
            "train_loss": 1.0 / (step + 1), "val_loss": 1.0, "val_acc": 0.5,
            "grad_norm": 0.5, "timestamp": 1000.0 + step, "anomaly_injected": None,
        })
    store.tables["decisions"].append({
        "id": "d0", "run_id": SEED_RUN_ID, "timestamp": 1000.0,
        "anomaly_types": [{"type": "loss_spike", "step": 2}],
        "tools_used": ["read_config"], "agent_response": "seeded",
        "fixed": False, "status": "failed",
        # Written by agent/logger.py on every decision. Present here because the
        # response contract has to be checked against what is really stored.
        "attempt": {"attempt_id": "a0", "outcome": "requested", "terminal": None},
    })
    return store


@pytest.fixture
def store(tmp_path, monkeypatch):
    import db

    fake = seed(FakeSupabase(), tmp_path)
    monkeypatch.setattr(db, "get_client", lambda: fake)

    import supabase

    def _poisoned(*_a, **_k):
        raise AssertionError(
            "a test reached supabase.create_client: it would have opened a real "
            "client against SUPABASE_URL"
        )

    monkeypatch.setattr(supabase, "create_client", _poisoned)
    # The tailer cache is process-global; a leftover entry would let one test's
    # cursor decide another test's ingest result.
    monkeypatch.setattr(db, "_tailers", {})
    return fake


@pytest.fixture
def app():
    import main
    return main.app


@pytest.fixture
def client(app, store):
    from fastapi.testclient import TestClient
    # raise_server_exceptions=False so an unhandled exception is observed as the 500
    # a real client would receive, which is the thing under test.
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
