from typing import Literal

from fastapi import APIRouter
from schemas import RunCreate, Run
from errors import (COLLECTION_RESPONSES, COLLECTION_WRITE_RESPONSES, ERROR_RESPONSES,
                    STATUS_RESPONSES, bad_request, internal_error, not_found)
import db

router = APIRouter()

VALID_STATUSES = ["running", "completed", "failed"]


# ── get all runs ───────────────────────────────────────────────────────────────
@router.get("/", response_model=list[Run], responses=COLLECTION_RESPONSES)
def get_runs():
    try:
        return db.get_runs()
    except Exception as e:
        raise internal_error(e, "get_runs")


# ── get single run ─────────────────────────────────────────────────────────────
@router.get("/{run_id}", response_model=Run, responses=ERROR_RESPONSES)
def get_run(run_id: str):
    try:
        run = db.get_run(run_id)
    except Exception as e:
        raise internal_error(e, "get_run")
    if not run:
        raise not_found()
    return run


# ── create run ─────────────────────────────────────────────────────────────────
@router.post("/", response_model=Run, responses=COLLECTION_WRITE_RESPONSES)
def create_run(body: RunCreate):
    try:
        return db.create_run(
            name=body.name,
            config_path=body.config_path,
            metrics_file=body.metrics_file,
            training_dir=body.training_dir
        )
    except Exception as e:
        raise internal_error(e, "create_run")


# ── update run status ──────────────────────────────────────────────────────────
# The permitted values are declared on the parameter so they reach the OpenAPI
# document as an enum. They were previously enforced only in the body of the handler,
# which meant the contract described `status` as any string at all: a client reading
# the document had no way to learn the three legal values, and Schemathesis reported
# the resulting 400 as the API rejecting a schema-compliant request. The handler check
# is kept below as a belt-and-braces guard rather than deleted, since it is the thing
# that answers if the annotation is ever loosened.
@router.patch("/{run_id}/status", responses=STATUS_RESPONSES)
def update_status(run_id: str, status: Literal["running", "completed", "failed"]):
    if status not in VALID_STATUSES:  # pragma: no cover - unreachable while the enum holds
        raise bad_request(f"status must be one of {VALID_STATUSES}")
    # Every other run-scoped route resolves the run before acting. This one did not, so
    # a PATCH against an id that was never created answered 200 {"status": "updated"}
    # and a caller with a stale or mistyped id was told its write had landed. The
    # underlying update is `.eq("id", run_id)` over zero rows, which no database
    # reports as an error, so nothing downstream could have caught it either.
    try:
        run = db.get_run(run_id)
    except Exception as e:
        raise internal_error(e, "update_status")
    if not run:
        raise not_found()
    try:
        db.update_run_status(run_id, status)
        return {"status": "updated"}
    except Exception as e:
        raise internal_error(e, "update_status")
