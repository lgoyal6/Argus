from dotenv import load_dotenv
import os
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
import json
import uuid
import time
from pathlib import Path
from supabase import create_client, Client

try:  # imported as a package by the tests, flat by the backend image
    from backend.tailer import MetricsTailer
except ImportError:  # pragma: no cover - exercised only by the flat layout
    from tailer import MetricsTailer

# obs.py lives at the repository root and is copied to /app by both Dockerfiles, so it
# is flat in the image and importable from the repository root. It is not inside either
# package, which is why this one needs no try/except.
import obs

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
# One tailer per run, so repeated polls resume instead of restarting. The cursor
# also lives on disk beside the metrics file, so a backend restart does not
# re-ingest the run from step 0.
_tailers = {}


def _tailer_for(run_id, metrics_file):
    t = _tailers.get(run_id)
    if t is None or str(t.path) != str(metrics_file):
        state = Path(metrics_file).with_suffix(Path(metrics_file).suffix + f".cursor-{run_id}")
        t = MetricsTailer(metrics_file, state_path=state)
        _tailers[run_id] = t
    return t


def insert_metrics(run_id, metrics_file):
    """Ingest only the metrics appended since the last successful call.

    This used to read the whole file and upsert every row on every poll, which
    makes ingest cost proportional to the run's total length rather than to what
    is new - measured at 620x write amplification over a 1,239-step run, and
    2,500x over 5,000 steps, because the factor is (N+1)/2 and grows with N.

    The cursor is committed only after the upsert is acknowledged, so a crash in
    between replays the batch rather than skipping it; the (run_id, step) conflict
    target makes that replay a no-op.
    """
    # run_id is bound for the whole ingest, so every line emitted below - including
    # from the tailer and from a dependency failure - carries the identifier that ties
    # this ingest to the agent's detection and recovery lines for the same run. It is
    # bound here and NOT used as a metric label: obs.declare_counter refuses it,
    # because one series per run is one series per training job forever.
    with obs.bind(run_id=run_id):
        client = get_client()
        path = Path(metrics_file)
        if not path.exists():
            obs.incr("argus_ingest_rows_total", outcome="missing_file")
            obs.incr("argus_ingest_failures_total",
                     dependency="metrics_file", reason="connection")
            obs.log("ingest.metrics_file_missing", level="error",
                    dependency="metrics_file")
            return {"error": "metrics file not found"}

        with obs.span("ingest.batch", dependency="supabase"):
            tailer = _tailer_for(run_id, metrics_file)
            entries, next_offset = tailer.read_batch()
            if not entries:
                tailer.commit(next_offset)
                obs.incr("argus_ingest_rows_total", outcome="empty")
                return {"inserted": 0}

            rows = []
            for entry in entries:
                entry["run_id"] = run_id
                rows.append(entry)

            try:
                client.table("metrics").upsert(rows, on_conflict="run_id,step").execute()
            except Exception as e:
                # The cursor is deliberately NOT committed here, so the batch replays.
                # The classification is what makes the log line answer "why did ingest
                # stop", rather than only "ingest stopped".
                reason = _ingest_failure_reason(e)
                obs.incr("argus_ingest_failures_total",
                         dependency="supabase", reason=reason)
                obs.log("ingest.sink_failed", level="error", dependency="supabase",
                        reason=reason, error_type=type(e).__name__, detail=str(e),
                        batch_rows=len(rows), cursor_committed=False)
                raise
            tailer.commit(next_offset)  # only after the sink accepted the batch

            obs.incr("argus_ingest_rows_total", outcome="ok")
            # `batch_rows`, not `rows`. A count of rows is a size and says how much work
            # the batch was; `rows` is a denied field name in obs.sanitise precisely so
            # that passing the actual row list there is summarised rather than logged,
            # and using that name for the count got the count summarised too.
            obs.log("ingest.committed", batch_rows=len(rows), offset=next_offset)
            return {"inserted": len(rows)}


def _page(rows, limit, offset):
    """Slice a result set. `limit=None` means the whole series.

    Applied here rather than pushed into the query because the ordering the callers
    depend on is applied by the same query, and a range() pushed down would have to
    reproduce it. The unbounded read is the pre-existing behaviour and stays the
    default; see routes/metrics.py for why.
    """
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _ingest_failure_reason(exc):
    """Bucket a sink failure into the closed set the metric label permits.

    Closed because the alternative is a label whose values are exception strings, and
    an unbounded label value set is the same cardinality bomb as an unbounded label
    name. Anything unrecognised becomes "protocol" rather than a new series.
    """
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timeout" in text:
        return "timeout"
    if "connect" in name or "connection" in text or "dns" in text:
        return "connection"
    if "auth" in name or "401" in text or "403" in text or "apikey" in text \
            or "jwt" in text or "unauthorized" in text:
        return "auth"
    return "protocol"


def get_metrics(run_id, limit=None, offset=0):
    client = get_client()
    response = client.table("metrics").select("*").eq("run_id", run_id).order("step").execute()
    return _page(response.data, limit, offset)


# ── decisions ──────────────────────────────────────────────────────────────────
def insert_decision(run_id, decision_payload):
    """Store one decision under the run named by the CALLER, not by the payload.

    This read `decision_payload["anomalies"]`, which is the key agent/logger.py uses in
    its own internal payload, while the route hands it a `Decision` whose field is
    `anomaly_types` - the name in the published schema and in the database column. So
    every request that satisfied the documented contract raised KeyError and came back
    to the client as a 500 carrying the string 'anomalies'. The endpoint had never
    worked; the agent writes its decisions to Supabase directly from logger.py and
    never exercises this path.

    `id` and `run_id` are generated and taken from the path respectively, so a
    client-supplied id cannot collide with or overwrite a stored decision.
    """
    client = get_client()
    row = {
        "id": str(uuid.uuid4()),
        "run_id": run_id,
        "timestamp": decision_payload["timestamp"],
        "anomaly_types": decision_payload["anomaly_types"],
        "tools_used": decision_payload["tools_used"],
        "agent_response": decision_payload["agent_response"],
        "fixed": decision_payload.get("fixed"),
        "status": decision_payload.get("status"),
        "attempt": decision_payload.get("attempt"),
    }
    client.table("decisions").insert(row).execute()
    return row


def get_decisions(run_id, limit=None, offset=0):
    client = get_client()
    response = client.table("decisions").select("*").eq("run_id", run_id).order("timestamp", desc=True).execute()
    return _page(response.data, limit, offset)
