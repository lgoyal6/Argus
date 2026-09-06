from fastapi import APIRouter, Query
from schemas import Decision
from errors import (DECISION_WRITE_RESPONSES, ERROR_RESPONSES, conflict,
                    internal_error, not_found)
import db

router = APIRouter()

MAX_PAGE = 5000


# ── get decisions for a run ────────────────────────────────────────────────────
@router.get("/{run_id}/decisions", response_model=list[Decision], responses=ERROR_RESPONSES)
def get_decisions(
    run_id: str,
    # See routes/metrics.py: optional is not nullable, and `int | None` published a
    # `null` branch the query parser rejects.
    limit: int = Query(None, ge=1, le=MAX_PAGE,
                       description="maximum rows to return; omit for all of them"),
    offset: int = Query(0, ge=0, description="rows to skip, newest first"),
):
    try:
        run = db.get_run(run_id)
    except Exception as e:
        raise internal_error(e, "get_decisions")
    if not run:
        raise not_found()
    try:
        return db.get_decisions(run_id, limit=limit, offset=offset)
    except Exception as e:
        raise internal_error(e, "get_decisions")


# ── insert a decision ──────────────────────────────────────────────────────────
@router.post("/{run_id}/decisions", response_model=Decision,
             responses=DECISION_WRITE_RESPONSES)
def insert_decision(run_id: str, payload: Decision):
    try:
        run = db.get_run(run_id)
    except Exception as e:
        raise internal_error(e, "insert_decision")
    if not run:
        raise not_found()
    # The run is addressable twice in one request: in the path and in the body. They
    # must not be allowed to disagree. Resolving the ambiguity silently in favour of
    # either one means a caller can be wrong about where its decision was filed and
    # never find out; refusing says so at the point the mistake is still cheap.
    if payload.run_id != run_id:
        raise conflict(
            "run_id in the body does not match the run_id in the path"
        )
    try:
        return db.insert_decision(run_id, payload.model_dump())
    except Exception as e:
        raise internal_error(e, "insert_decision")
