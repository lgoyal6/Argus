import os
import json
import uuid
import time
from pathlib import Path
from supabase import create_client, Client


# ── client ─────────────────────────────────────────────────────────────────────
def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
    return create_client(url, key)


# ── runs ───────────────────────────────────────────────────────────────────────
def create_run(name, config_path, metrics_file, training_dir):
    client = get_client()
    run = {
        "id": str(uuid.uuid4()),
        "name": name,
        "config_path": config_path,
        "metrics_file": metrics_file,
        "training_dir": training_dir,
        "status": "running",
        "created_at": time.time(),
        "updated_at": time.time()
    }
    client.table("runs").insert(run).execute()
    return run


def get_runs():
    client = get_client()
    response = client.table("runs").select("*").order("created_at", desc=True).execute()
    return response.data


def get_run(run_id):
    client = get_client()
    response = client.table("runs").select("*").eq("id", run_id).execute()
    return response.data[0] if response.data else None


def update_run_status(run_id, status):
    client = get_client()
    client.table("runs").update({
        "status": status,
        "updated_at": time.time()
    }).eq("id", run_id).execute()


# ── metrics ────────────────────────────────────────────────────────────────────
def insert_metrics(run_id, metrics_file):
    client = get_client()
    path = Path(metrics_file)
    if not path.exists():
        return {"error": "metrics file not found"}

    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    rows = []
    for line in lines:
        entry = json.loads(line)
        entry["run_id"] = run_id
        entry["id"] = str(uuid.uuid4())
        rows.append(entry)

    if rows:
        client.table("metrics").insert(rows).execute()

    return {"inserted": len(rows)}


def get_metrics(run_id):
    client = get_client()
    response = client.table("metrics").select("*").eq("run_id", run_id).order("step").execute()
    return response.data


# ── decisions ──────────────────────────────────────────────────────────────────
def insert_decision(run_id, decision_payload):
    client = get_client()
    row = {
        "id": str(uuid.uuid4()),
        "run_id": run_id,
        "timestamp": decision_payload["timestamp"],
        "anomaly_types": decision_payload["anomalies"],
        "tools_used": decision_payload["tools_used"],
        "agent_response": decision_payload["agent_response"],
        "fixed": decision_payload["fixed"]
    }
    client.table("decisions").insert(row).execute()
    return row


def get_decisions(run_id):
    client = get_client()
    response = client.table("decisions").select("*").eq("run_id", run_id).order("timestamp", desc=True).execute()
    return response.data
