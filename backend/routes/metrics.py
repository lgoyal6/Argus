from fastapi import APIRouter, Query
from schemas import MetricEntry
from errors import ERROR_RESPONSES, SYNC_RESPONSES, conflict, internal_error, not_found
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
    # `int | None` is the natural Python annotation and the wrong contract: FastAPI
    # renders it as `anyOf: [integer, null]`, so the published document said `null` was
    # a permitted value of the parameter while the query parser answered 422 to
    # `?limit=null`. The parameter is optional, meaning it may be *absent*; it is not
    # nullable, and there is no string a client can put after `limit=` that means null.
    # Annotating the value type and leaving the default as None keeps the handler
    # behaviour identical - omitted is still None - and drops the null branch from the
    # schema. Schemathesis found this by generating the value the document permitted.
    limit: int = Query(None, ge=1, le=MAX_PAGE,
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
@router.post("/{run_id}/metrics/sync", responses=SYNC_RESPONSES)
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
        # Not a 404. The addressed resource - the run - exists and was resolved three
        # lines above; what is missing is the file its row names. Answering 404 here
        # gave one status two meanings on one route, and the two detail strings that
        # told them apart turned the refusal into an existence oracle: a caller could
        # sweep run ids and read "run not found" against "metrics file not found" to
        # learn which ones were real. Schemathesis found the collision by creating a
        # run and immediately getting a 404 for it (ensure_resource_availability).
        raise conflict(result["error"])
    return result
