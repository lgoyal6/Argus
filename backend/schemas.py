from pydantic import BaseModel
from typing import Optional, Any
import time


# ── training run ───────────────────────────────────────────────────────────────
class RunCreate(BaseModel):
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
