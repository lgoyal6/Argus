from fastapi import APIRouter, Query
from schemas import MetricEntry
from errors import ERROR_RESPONSES, internal_error, not_found
import db

router = APIRouter()

# A metrics series grows one row per emitted step, which is the same growth that made
# the old whole-file ingest quadratic. The read side had no bound at all: a request
# about a 20,000-step run returned 20,000 rows. The bound is opt-in rather than a
# default page size because the dashboard charts a whole run and passes no limit, and
# silently truncating every chart to fix an unbounded read would be a worse bug than
# the one being fixed.
MAX_PAGE = 5000


# ── get metrics for a run ──────────────────────────────────────────────────────
@router.get("/{run_id}/metrics", response_model=list[MetricEntry], responses=ERROR_RESPONSES)
def get_metrics(
    run_id: str,
    limit: int | None = Query(None, ge=1, le=MAX_PAGE,
                              description="maximum rows to return; omit for the whole series"),
    offset: int = Query(0, ge=0, description="rows to skip, oldest first"),
):
    try:
        run = db.get_run(run_id)
    except Exception as e:
        raise internal_error(e, "get_metrics")
    if not run:
        raise not_found()
    try:
        return db.get_metrics(run_id, limit=limit, offset=offset)
    except Exception as e:
        raise internal_error(e, "get_metrics")


# ── sync metrics from file into supabase ───────────────────────────────────────
@router.post("/{run_id}/metrics/sync", responses=ERROR_RESPONSES)
def sync_metrics(run_id: str):
    try:
        run = db.get_run(run_id)
    except Exception as e:
        raise internal_error(e, "sync_metrics")
    if not run:
        raise not_found()
    try:
        result = db.insert_metrics(run_id, run["metrics_file"])
    except Exception as e:
        raise internal_error(e, "sync_metrics")
    # insert_metrics reports a missing file in its return value. Passing that straight
    # through gave a 200 whose body was an error: the status line said the ingest
    # succeeded and nothing in the declared response told a client to look for an
    # `error` key.
    if "error" in result:
        raise not_found(result["error"])
    return result
