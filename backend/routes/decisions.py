from fastapi import APIRouter, HTTPException
from schemas import Decision
import db

router = APIRouter()


# ── get decisions for a run ────────────────────────────────────────────────────
@router.get("/{run_id}/decisions", response_model=list[Decision])
def get_decisions(run_id: str):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    try:
        return db.get_decisions(run_id)
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
