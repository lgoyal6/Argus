from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Annotated, Optional, Any
import time


def _must_survive_a_round_trip(value: str, field: str) -> str:
    """Reject a string the service cannot store or serialise.

    A lone surrogate is a valid `str` in Python and an invalid UTF-8 sequence on the
    wire. Accepting one into a run row made `POST /runs/` crash *after* the route
    returned, inside FastAPI's response serialisation, where the route's own
    `except Exception` cannot see it: the client got a bare `Internal Server Error`
    in text/plain rather than the service's declared error shape, and no error_id was
    logged because no handler ran.
    """
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(
            f"{field} contains characters that are not encodable as UTF-8"
        ) from None
    return value


# A path that no later operation can act on. `metrics_file=""` was accepted at creation
# and only failed one call later, inside POST /runs/{id}/metrics/sync, as
# `PosixPath('.') has an empty name` - a 500 on a value the API itself had approved,
# leaving a run that was permanently un-syncable.
#
# The constraint is declared as `min_length` and `pattern` rather than enforced in a
# validator so that it reaches the OpenAPI document. A validator would reject `""`
# while the published schema still said any string was acceptable, which is the same
# contract lie in the opposite direction: Schemathesis reported exactly that as "API
# rejected schema-compliant request" when the first version of this fix used a
# validator. `\S` is an unanchored search, so it means "contains a non-whitespace
# character" and rejects "   " as well as "".
UsablePath = Annotated[str, Field(min_length=1, pattern=r"\S")]


# ── training run ───────────────────────────────────────────────────────────────
class RunCreate(BaseModel):
    # A field the server ignores is a field the client believes it set. With pydantic's
    # default extra="ignore", a body carrying `trainng_dir` created a run whose real
    # training_dir came from somewhere else and answered 200, so the typo was
    # undiscoverable from the response. Refusing it turns a silent wrong value into a
    # 422 that names the offending key.
    model_config = ConfigDict(extra="forbid")

    name: str
    config_path: UsablePath
    metrics_file: UsablePath
    training_dir: UsablePath

    # Encodability is not expressible in JSON Schema - it is a property of the wire
    # encoding, not of the value - so unlike the length and pattern constraints above
    # this one stays a validator. Nothing generated against the document can violate
    # it either, because a generator that emits UTF-8 cannot produce a lone surrogate;
    # it is reachable only from a hand-written client, which is how it was found.
    @field_validator("name", "config_path", "metrics_file", "training_dir")
    @classmethod
    def _encodable(cls, v: str, info) -> str:
        return _must_survive_a_round_trip(v, info.field_name)


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

    @field_validator("timestamp", mode="before")
    @classmethod
    def _not_a_bool(cls, v):
        """`bool` is a subclass of `int`, so pydantic coerces it to a number.

        `{"timestamp": false}` was accepted and stored as `0.0`, and `true` as `1.0`:
        the contract declares a number, the client sent a boolean by mistake, and the
        service answered 200 with a decision timestamped at the Unix epoch instead of
        refusing it. The hand-rolled corpus does send `("bool", True)` for every field,
        but its assertion is only that the status is one the contract declares - and
        200 is declared - so a wrong value accepted with a right status passed.
        """
        if isinstance(v, bool):
            raise ValueError("timestamp must be a number, not a boolean")
        return v

    anomaly_types: list[Any]
    tools_used: list[str]
    agent_response: str
    fixed: Optional[bool] = None
    status: Optional[str] = None  # "fixed" | "patched" | "failed"

    @field_validator("fixed", mode="before")
    @classmethod
    def _not_a_number(cls, v):
        """The mirror of `_not_a_bool`, and the same class of silent coercion.

        `{"fixed": 0}` was accepted where the contract declares a boolean and stored as
        `False`, so a client sending a count or a status code by mistake recorded a
        definite "this run was not fixed" instead of being refused. Found by
        Schemathesis' `negative_data_rejection` check on the round after `timestamp`
        was fixed; the two are the same bug on two fields, which is a good argument for
        why one hand-written case per field is not the same thing as generation.
        """
        if v is not None and not isinstance(v, bool):
            raise ValueError("fixed must be a boolean")
        return v
    # The four-rung evidence ladder from agent/attempt.py. agent/logger.py writes this
    # on every decision row; without the field here, response_model=list[Decision]
    # dropped it on the way out and the API served a recovery claim with the evidence
    # behind it silently removed.
    attempt: Optional[Any] = None
