import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Flat in both layouts; see the note in agent/detector.py.
import obs

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# ── supabase ───────────────────────────────────────────────────────────────────
from supabase import create_client

try:  # imported as a package by the tests, flat by the agent image
    from backend.argus_secrets import SecretNotConfigured, get_secret
except ImportError:  # pragma: no cover - exercised only by the flat layout
    from argus_secrets import SecretNotConfigured, get_secret


def get_supabase():
    url = os.environ.get("SUPABASE_URL")
    if not url:
        return None
    try:
        key = get_secret("SUPABASE_KEY")
    except SecretNotConfigured:
        # No credential configured at all: the caller already falls back to the local
        # decision log. A Secret Manager denial is a different case and is left to
        # propagate, because that one means the grant is wrong.
        return None
    return create_client(url, key)

# ── local fallback ─────────────────────────────────────────────────────────────
LOCAL_LOG_FILE = "logs/decisions.jsonl"

# ── run id (set once when loop starts) ────────────────────────────────────────
_run_id = None

def set_run_id(run_id):
    global _run_id
    _run_id = run_id


def log_decision(anomalies, agent_response, tools_used, status="failed", attempt=None):
    # status: "fixed" | "patched" | "failed", derived by agent/attempt.py from what
    # was observed in the metrics stream. `attempt` carries the evidence behind it -
    # the trigger, the approved action, the resulting config and the observations - so
    # a stored row can be re-audited rather than taken on trust.
    payload = {
        "timestamp": time.time(),
        "anomalies": anomalies,
        "tools_used": tools_used,
        "agent_response": agent_response,
        "status": status,
        "fixed": status == "fixed",
        "attempt": attempt
    }

    _write_local(payload)

    if _run_id:
        _write_supabase(payload)
    else:
        print("no run_id set, skipping Supabase write")

    return payload


def _write_local(payload):
    path = Path(LOCAL_LOG_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")
    print(f"\nlogged decision locally")


def _write_supabase(payload):
    try:
        client = get_supabase()
        if not client:
            print("supabase client unavailable, skipping remote log")
            return
        row = {
            "id": __import__('uuid').uuid4().__str__(),
            "run_id": _run_id,
            "timestamp": payload["timestamp"],
            "anomaly_types": payload["anomalies"],
            "tools_used": payload["tools_used"],
            "agent_response": payload["agent_response"],
            "fixed": payload["fixed"],
            "status": payload["status"],
            "attempt": payload["attempt"]
        }
        client.table("decisions").insert(row).execute()
        print("logged decision to Supabase")
    except Exception as e:
        # `print(f"...{e}")` put unreviewed exception text on stdout. The supabase
        # client holds the key in postgrest.session.headers under both `apikey` and
        # `authorization`, so anything that renders a request or session object writes
        # the credential to the container log, where it is retained for as long as the
        # log is. obs.log runs the same scrubber over every field.
        obs.log("logger.supabase_write_failed", level="error",
                error_type=type(e).__name__, detail=str(e))


def print_decision(payload):
    print("\n── decision log ───────────────────────────────────────────")
    print(f"timestamp : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(payload['timestamp']))}")
    print(f"anomalies : {[a['type'] for a in payload['anomalies']]}")
    print(f"tools used: {payload['tools_used']}")
    print(f"status    : {payload['status']}")
    print(f"response  :\n{payload['agent_response']}")
    print("───────────────────────────────────────────────────────────\n")
