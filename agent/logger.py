import json
import time
from pathlib import Path

# ── local fallback log (before Supabase is set up) ────────────────────────────
LOCAL_LOG_FILE = "logs/decisions.jsonl"


def log_decision(anomalies, agent_response, tools_used):
    payload = {
        "timestamp": time.time(),
        "anomalies": anomalies,
        "tools_used": tools_used,
        "agent_response": agent_response,
        "fixed": _infer_fixed(agent_response)
    }

    # write locally for now
    _write_local(payload)

    # TODO: replace with Supabase write once db.py is set up
    # _write_supabase(payload)

    return payload


# ── infer whether the fix worked ───────────────────────────────────────────────
def _infer_fixed(agent_response):
    if not agent_response:
        return False
    response_lower = agent_response.lower()
    positive = ["resolved", "fixed", "improved", "recovered", "success", "no longer"]
    negative = ["failed", "persists", "could not", "unable", "still detecting"]
    for word in negative:
        if word in response_lower:
            return False
    for word in positive:
        if word in response_lower:
            return True
    return None   # uncertain


# ── local writer ───────────────────────────────────────────────────────────────
def _write_local(payload):
    path = Path(LOCAL_LOG_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")
    print(f"\nlogged decision to {LOCAL_LOG_FILE}")


# ── pretty print for terminal ──────────────────────────────────────────────────
def print_decision(payload):
    print("\n── decision log ───────────────────────────────────────────")
    print(f"timestamp : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(payload['timestamp']))}")
    print(f"anomalies : {[a['type'] for a in payload['anomalies']]}")
    print(f"tools used: {payload['tools_used']}")
    print(f"fixed     : {payload['fixed']}")
    print(f"response  :\n{payload['agent_response']}")
    print("───────────────────────────────────────────────────────────\n")
