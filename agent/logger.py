import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# ── supabase ───────────────────────────────────────────────────────────────────
from supabase import create_client

def get_supabase():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

# ── local fallback ─────────────────────────────────────────────────────────────
LOCAL_LOG_FILE = "logs/decisions.jsonl"

# ── run id (set once when loop starts) ────────────────────────────────────────
_run_id = None

def set_run_id(run_id):
    global _run_id
    _run_id = run_id


def log_decision(anomalies, agent_response, tools_used):
    payload = {
        "timestamp": time.time(),
        "anomalies": anomalies,
        "tools_used": tools_used,
        "agent_response": agent_response,
        "fixed": _infer_fixed(agent_response)
    }

    _write_local(payload)

    if _run_id:
        _write_supabase(payload)
    else:
        print("no run_id set, skipping Supabase write")

    return payload


def _infer_fixed(agent_response):
    if not agent_response:
        return False
    response_lower = agent_response.lower()
    negative = ["failed", "persists", "could not", "unable", "still detecting"]
    positive = ["resolved", "fixed", "improved", "recovered", "success", "no longer"]
    for word in negative:
        if word in response_lower:
            return False
    for word in positive:
        if word in response_lower:
            return True
    return None


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
            "fixed": payload["fixed"]
        }
        client.table("decisions").insert(row).execute()
        print("logged decision to Supabase")
    except Exception as e:
        print(f"supabase write failed: {e}")


def print_decision(payload):
    print("\n── decision log ───────────────────────────────────────────")
    print(f"timestamp : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(payload['timestamp']))}")
    print(f"anomalies : {[a['type'] for a in payload['anomalies']]}")
    print(f"tools used: {payload['tools_used']}")
    print(f"fixed     : {payload['fixed']}")
    print(f"response  :\n{payload['agent_response']}")
    print("───────────────────────────────────────────────────────────\n")
