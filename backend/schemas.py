from pydantic import BaseModel, ConfigDict
from typing import Optional, Any
import time


# ── training run ───────────────────────────────────────────────────────────────
class RunCreate(BaseModel):
    # A field the server ignores is a field the client believes it set. With pydantic's
    # default extra="ignore", a body carrying `trainng_dir` created a run whose real
    # training_dir came from somewhere else and answered 200, so the typo was
    # undiscoverable from the response. Refusing it turns a silent wrong value into a
    # 422 that names the offending key.
    model_config = ConfigDict(extra="forbid")

    name: str
    config_path: str
    metrics_file: str
    training_dir: str


class Run(BaseModel):
    id: str
    name: str
    config_path: str
    metrics_file: str
    training_dir: str
    status: str                 # running, completed, failed
    created_at: float
    updated_at: float


# ── metrics ────────────────────────────────────────────────────────────────────
class MetricEntry(BaseModel):
    step: int
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    grad_norm: float
    timestamp: float
    anomaly_injected: Optional[str] = None


# ── agent decisions ───────────────────────────────────────────────────────────

class Decision(BaseModel):
    id: str
    run_id: str
    timestamp: float
    anomaly_types: list[Any]
    tools_used: list[str]
    agent_response: str
    fixed: Optional[bool] = None
    status: Optional[str] = None  # "fixed" | "patched" | "failed"
    # The four-rung evidence ladder from agent/attempt.py. agent/logger.py writes this
    # on every decision row; without the field here, response_model=list[Decision]
    # dropped it on the way out and the API served a recovery claim with the evidence
    # behind it silently removed.
    attempt: Optional[Any] = None
