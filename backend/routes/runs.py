from fastapi import APIRouter, HTTPException
from schemas import RunCreate, Run
import db

router = APIRouter()


# ── get all runs ───────────────────────────────────────────────────────────────
@router.get("/", response_model=list[Run])
def get_runs():
    try:
        return db.get_runs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── get single run ─────────────────────────────────────────────────────────────
@router.get("/{run_id}", response_model=Run)
def get_run(run_id: str):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


# ── create run ─────────────────────────────────────────────────────────────────
@router.post("/", response_model=Run)
def create_run(body: RunCreate):
    try:
        return db.create_run(
            name=body.name,
            config_path=body.config_path,
            metrics_file=body.metrics_file,
            training_dir=body.training_dir
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── update run status ──────────────────────────────────────────────────────────
@router.patch("/{run_id}/status")
def update_status(run_id: str, status: str):
    valid = ["running", "completed", "failed"]
    if status not in valid:
        raise HTTPException(status_code=400, detail=f"status must be one of {valid}")
    try:
        db.update_run_status(run_id, status)
        return {"status": "updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
