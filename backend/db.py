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
    from backend import tailer as tailer_defaults
except ImportError:  # pragma: no cover - exercised only by the flat layout
    from tailer import MetricsTailer
    import tailer as tailer_defaults

try:  # same two layouts; see the note above
    from backend.argus_secrets import get_secret
except ImportError:  # pragma: no cover - exercised only by the flat layout
    from argus_secrets import get_secret

# Plain, and deliberately NOT under the try/except above. limits reads its numbers at
# import time, and tests/test_abuse_limits.py configures a different bound by dropping
# "limits" from sys.modules and re-importing, which is the code path an operator gets
# by setting the variable and restarting. Binding `backend.limits` here would leave
# this module holding the first import's numbers while the test re-imported the other
# name, so the re-import would appear to work and change nothing.
import limits

# obs.py lives at the repository root and is copied to /app by both Dockerfiles, so it
# is flat in the image and importable from the repository root. It is not inside either
# package, which is why this one needs no try/except.
import obs

# ── client ─────────────────────────────────────────────────────────────────────
def get_client() -> Client:
    # SUPABASE_URL is not a credential and stays plain configuration. SUPABASE_KEY
    # goes through the resolver: Secret Manager when ARGUS_SECRET_PROJECT is set, the
    # environment otherwise. Nothing is memoised here, so a rotation takes effect on
    # the next call rather than at the next restart.
    url = os.environ.get("SUPABASE_URL")
    if not url:
        raise ValueError("SUPABASE_URL must be set in environment")
    return create_client(url, get_secret("SUPABASE_KEY"))


# ── runs ───────────────────────────────────────────────────────────────────────
# Default page sizes for the two collections that have no whole-object read.
DEFAULT_RUN_PAGE = 200
DEFAULT_DECISION_PAGE = 500


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
    """The run index grows without bound as runs accumulate; page it.

    Unlike a metrics series, this has no natural whole-object read: nothing charts
    every run that ever existed. So an absent `limit` takes a default page here
    rather than meaning "all of them", and the default is still checked against the
    work bound so the two cannot disagree.
    """
    if limit is None:
        limit = min(limits.MAX_WORK_UNITS or DEFAULT_RUN_PAGE, DEFAULT_RUN_PAGE)
    limits.check_work(limit, "rows")
    client = get_client()
    response = client.table("runs").select("*").order("created_at", desc=True).execute()
    return _page(response.data, limit, offset)


def get_run(run_id):
    client = get_client()
    # Spanned because it is on the ingest path: every /metrics/sync checks the run
    # exists before reading a byte of the file. A slow database showed up here first,
    # and without a span the request span was simply slow for no visible reason.
    with obs.span("db.select", kind="client", dependency="supabase", table="runs"):
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
        # The two per-poll caps are the ingest bound, and an operator tunes them
        # through the same variables as every other limit. They can only tighten the
        # tailer's own defaults, never loosen them: a bound that an environment
        # variable can raise without limit is not a bound. 0 means "not configured",
        # which leaves the default in force rather than removing it.
        t = MetricsTailer(
            metrics_file,
            state_path=state,
            max_lines=min(limits.MAX_WORK_UNITS or tailer_defaults.DEFAULT_MAX_LINES,
                          tailer_defaults.DEFAULT_MAX_LINES),
            max_bytes=min(limits.MAX_INGEST_BYTES or tailer_defaults.DEFAULT_MAX_BYTES,
                          tailer_defaults.DEFAULT_MAX_BYTES),
        )
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

        # Read outside the ingest span, because the parent of that span is a header on
        # the rows this read returns: the trainer wrote them in another process,
        # usually before this one started, so nothing is inherited from an ambient
        # context. Whichever row is newest in the batch is the caller the ingest is
        # acting for.
        tailer = _tailer_for(run_id, metrics_file)
        entries, next_offset = tailer.read_batch()
        rows, queue_parent = [], None
        for entry in entries:
            # The `_trace` header comes off here and goes no further. The metrics
            # table stores measurements; a telemetry header in it would be a column
            # the schema does not have and a copy of the trace nobody would read.
            row, traceparent = obs.take_trace(entry)
            row["run_id"] = run_id
            rows.append(row)
            if traceparent:
                queue_parent = traceparent

        # An empty poll opens no span. The agent polls every 10 seconds whether or not
        # the trainer wrote anything, so a span here would be several thousand a day
        # per run that all say "nothing arrived".
        if not rows:
            tailer.commit(next_offset)
            obs.incr("argus_ingest_rows_total", outcome="empty")
            return {"inserted": 0}

        with obs.span("ingest.batch", kind="consumer",
                      parent=obs.inherited_or(queue_parent),
                      dependency="supabase", batch_rows=len(rows)):
            try:
                # The database call itself, and its own span: the one place in this
                # function that leaves the process for something other than a file.
                # Without it a slow ingest is attributable only to "ingest", and the
                # question is always whether the sink or the file was slow.
                with obs.span("db.upsert", kind="client", dependency="supabase",
                              table="metrics", batch_rows=len(rows)):
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
    """Read a metrics series, bounded by rows.

    `limit=None` still means the whole series, because the dashboard charts a whole
    run and a silent default page size would truncate every chart. What it no longer
    means is "however many rows there are": above MAX_WORK_UNITS the request is
    refused with a 413 that names `limit` and `offset`, so a caller asking for a
    million rows is told how to page rather than served or silently cut short.
    """
    if limit is not None:
        # Checked before the query, so an oversized explicit ask costs nothing.
        limits.check_work(limit, "rows")
    client = get_client()
    with obs.span("db.select", kind="client", dependency="supabase", table="metrics"):
        response = client.table("metrics").select("*").eq("run_id", run_id).order("step").execute()
    if limit is None:
        limits.check_work(len(response.data), "rows")
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
    """One decision row per agent intervention; bounded the same way as the runs index."""
    if limit is None:
        limit = min(limits.MAX_WORK_UNITS or DEFAULT_DECISION_PAGE, DEFAULT_DECISION_PAGE)
    limits.check_work(limit, "rows")
    client = get_client()
    response = client.table("decisions").select("*").eq("run_id", run_id).order("timestamp", desc=True).execute()
    return _page(response.data, limit, offset)
