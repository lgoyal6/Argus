from fastapi import APIRouter, HTTPException, Query
from schemas import MetricEntry
import db
import limits

router = APIRouter()


# ── get metrics for a run ──────────────────────────────────────────────────────
@router.get("/{run_id}/metrics", response_model=list[MetricEntry])
def get_metrics(
    run_id: str,
    limit: int | None = Query(None, ge=1, description="rows to return; omit for the whole series"),
    offset: int = Query(0, ge=0, description="rows to skip, oldest first"),
):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    try:
        return db.get_metrics(run_id, limit=limit, offset=offset)
    except limits.WorkLimitExceeded:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── sync metrics from file into supabase ───────────────────────────────────────
@router.post("/{run_id}/metrics/sync")
def sync_metrics(run_id: str):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    try:
        result = db.insert_metrics(run_id, run["metrics_file"])
        return result
    except limits.WorkLimitExceeded:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
