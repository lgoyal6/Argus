from fastapi import APIRouter, HTTPException, Query
from schemas import Decision
import db
import limits

router = APIRouter()


# ── get decisions for a run ────────────────────────────────────────────────────
@router.get("/{run_id}/decisions", response_model=list[Decision])
def get_decisions(
    run_id: str,
    limit: int | None = Query(None, ge=1, description="rows to return, newest first"),
    offset: int = Query(0, ge=0),
):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    try:
        return db.get_decisions(run_id, limit=limit, offset=offset)
    except limits.WorkLimitExceeded:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── insert a decision ──────────────────────────────────────────────────────────
@router.post("/{run_id}/decisions")
def insert_decision(run_id: str, payload: Decision):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    try:
        return db.insert_decision(run_id, payload.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
