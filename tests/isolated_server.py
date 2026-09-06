"""Serve the real Argus app over real HTTP against an isolated in-memory store.

Schemathesis drives an HTTP client, not an ASGI callable, so checking "actual request
and response behavior" the way C03 asks means a socket, a real uvicorn worker and real
serialisation - not a TestClient that short-circuits both. What must not change is the
isolation: the generated corpus includes destructive bodies, so the process that
answers them has to be unable to reach a real Supabase project.

This module reuses the exact fake and seed from `tests/conftest.py` rather than
building a second one, so the HTTP surface Schemathesis sees and the surface the
hand-rolled tests see are backed by identical rows. `supabase.create_client` is
poisoned here too: a code path that slipped past the patched `db.get_client` raises
instead of opening a client against whatever SUPABASE_URL is in the environment.

    python tests/isolated_server.py --port 8731

`GET /__store` is a test-only introspection route added by this harness and not by the
application. It exists so an authorization check can read the stored rows over a path
that is not the path under test; see tests/test_api_authorization.py.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for _p in (str(ROOT), str(BACKEND), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from conftest import FakeSupabase, seed, _install_supabase_stub  # noqa: E402

_install_supabase_stub()


def build_app(tmp_dir: Path):
    """The real app, with the store swapped and the network poisoned."""
    import db
    import supabase

    store = seed(FakeSupabase(), tmp_dir)
    db.get_client = lambda: store
    db._tailers = {}

    def _poisoned(*_a, **_k):
        raise AssertionError(
            "the isolated server reached supabase.create_client: it would have "
            "opened a real client against SUPABASE_URL"
        )

    supabase.create_client = _poisoned

    import main

    app = main.app

    @app.get("/__store", include_in_schema=False)
    def _dump_store():
        # Deliberately excluded from the schema: it is harness scaffolding, and
        # including it would make the committed contract describe a route the real
        # image does not serve.
        return store.tables

    return app


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()

    tmp = Path(args.tmp) if args.tmp else ROOT / ".agent-work" / "st-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "config.yaml").write_text("lr: 0.1\n")

    import uvicorn

    uvicorn.run(build_app(tmp), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
