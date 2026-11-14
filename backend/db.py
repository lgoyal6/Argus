from dotenv import load_dotenv
import os
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
import json
import uuid
import time
from pathlib import Path
from supabase import create_client, Client
import limits

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


def get_runs(limit=None, offset=0):
    """The run index grows without bound as runs accumulate; page it."""
    client = get_client()
    if limit is None:
        limit = min(limits.MAX_WORK_UNITS or 200, 200)
    limits.check_work(limit, "rows")
    response = (
        client.table("runs")
        .select("*")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
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
    """Ingest a metrics file under two bounds: bytes read and rows upserted.

    The file is named by whoever created the run, so both its size and its line
    count are caller-influenced. `readlines()` on the whole file was the unbounded
    part: a long-running job writes one line per step forever, and the sync
    endpoint would read all of it into memory and upsert all of it in one call.

    Bytes are bounded first, while reading, so an oversized file costs one read of
    the cap and not one read of the file. Rows are bounded second. Both refuse
    instead of truncating, because a partially ingested run that reports success is
    a silent data bug and a 413 that names the limit is not.
    """
    client = get_client()
    path = Path(metrics_file)
    if not path.exists():
        return {"error": "metrics file not found"}

    if limits.MAX_INGEST_BYTES:
        size = path.stat().st_size
        if size > limits.MAX_INGEST_BYTES:
            raise limits.WorkLimitExceeded(size, limits.MAX_INGEST_BYTES, "bytes")

    lines = []
    read = 0
    with open(path, "r") as f:
        for line in f:
            read += len(line)
            if limits.MAX_INGEST_BYTES and read > limits.MAX_INGEST_BYTES:
                raise limits.WorkLimitExceeded(
                    read, limits.MAX_INGEST_BYTES, "bytes"
                )
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
                if limits.MAX_WORK_UNITS and len(lines) > limits.MAX_WORK_UNITS:
                    raise limits.WorkLimitExceeded(
                        len(lines), limits.MAX_WORK_UNITS, "rows"
                    )

    rows = []
    for line in lines:
        entry = json.loads(line)
        entry["run_id"] = run_id
        rows.append(entry)

    if rows:
        client.table("metrics").upsert(rows, on_conflict="run_id,step").execute()

    return {"inserted": len(rows)}


def get_metrics(run_id, limit=None, offset=0):
    """Read a metrics series, bounded by rows.

    The dashboard charts a whole run and passes no limit, so a default page size
    would silently truncate every chart. Instead the server counts the series
    first, which is one cheap COUNT rather than a full fetch, and refuses above
    MAX_WORK_UNITS with a 413 that names `limit` and `offset`. A caller that wants
    a longer run pages through it; a caller that asks for a million rows in one
    request is told no rather than served.
    """
    client = get_client()
    if limit is None:
        counted = (
            client.table("metrics")
            .select("step", count="exact")
            .eq("run_id", run_id)
            .limit(1)
            .execute()
        )
        total = counted.count or 0
        limits.check_work(total, "rows")
        query = client.table("metrics").select("*").eq("run_id", run_id).order("step")
    else:
        limits.check_work(limit, "rows")
        query = (
            client.table("metrics")
            .select("*")
            .eq("run_id", run_id)
            .order("step")
            .range(offset, offset + limit - 1)
        )
    return query.execute().data


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


def get_decisions(run_id, limit=None, offset=0):
    """One decision row per agent intervention; bounded the same way as metrics."""
    client = get_client()
    if limit is None:
        limit = min(limits.MAX_WORK_UNITS or 500, 500)
    limits.check_work(limit, "rows")
    response = (
        client.table("decisions")
        .select("*")
        .eq("run_id", run_id)
        .order("timestamp", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return response.data
