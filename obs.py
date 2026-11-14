"""Correlated logs and bounded metrics for the ingest, detection and recovery path.

Argus is three processes that only ever meet through a run id: the backend ingests a
metrics file, the agent detects anomalies in it, and a recovery attempt acts on what it
found. When one of those stalls, the question is always "which dependency, on which
run" - and until now the answer had to be reconstructed from `print()` output with no
identifiers in it at all.

This module is deliberately about 200 lines with no dependencies. The project already
rejected Temporal for costing a server and 51 MB to buy one property it could write
itself; importing OpenTelemetry to correlate three processes over a shared run id would
be the same trade. What is here is the part that actually earns its cost.

Three rules it enforces rather than documents:

**Identifiers go in logs, never in metric labels.** `run_id` as a Prometheus label is a
new time series per training run, forever. `declare_counter` refuses identifier-shaped
label names outright, and every label must be declared with the finite set of values it
may take, so a metric's maximum series count is computable at declaration time
(`max_series()`) instead of discovered in production.

**Measurements and configuration never become attributes.** A log or span attribute
carrying `train_loss=0.213` or a whole config dict is duplicated telemetry: those values
already exist in the metrics stream and the attempt ledger, which are the auditable
copies. Worse, config contents are exactly where credentials live. `log()` replaces the
value of any measurement- or configuration-shaped field with a shape summary and keeps
the field name, which is the part that carries the information.

**Nothing that looks like a credential is emitted.** Scrubbing lives here because this
is the only choke point every message passes through.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import re
import sys
import time
import uuid

# -- correlation ---------------------------------------------------------------
# The identifiers that tie a line in the backend's ingest log to a line in the agent's
# recovery log. They ride in a contextvar so a call three frames down does not have to
# thread them through every signature, which is what stops them being dropped.
_CONTEXT: contextvars.ContextVar[dict] = contextvars.ContextVar("argus_obs_context", default={})

CORRELATION_KEYS = ("run_id", "attempt_id", "span_id")


@contextlib.contextmanager
def bind(**ids):
    """Attach correlation identifiers to everything logged inside the block."""
    unknown = set(ids) - set(CORRELATION_KEYS)
    if unknown:
        raise ValueError(f"bind() takes correlation identifiers only, not {sorted(unknown)}")
    merged = dict(_CONTEXT.get())
    merged.update({k: v for k, v in ids.items() if v is not None})
    token = _CONTEXT.set(merged)
    try:
        yield merged
    finally:
        _CONTEXT.reset(token)


def context() -> dict:
    return dict(_CONTEXT.get())


# -- what may not become an attribute ------------------------------------------
# Measurement names emitted by training_job/train.py and read by agent/detector.py.
# Their values belong in the metrics stream; repeating them in a log attribute doubles
# the storage and makes the log the thing people trust instead of the stream.
_MEASUREMENT_FIELDS = frozenset({
    "train_loss", "val_loss", "val_acc", "grad_norm", "loss", "accuracy",
    "zscore", "z_score", "learning_rate", "lr", "gradient_clip", "batch_size",
    "weight_decay", "dropout", "momentum", "epochs", "value", "values",
})

# Configuration and payload names. A config dict is where SUPABASE_KEY and
# ANTHROPIC_API_KEY would arrive if anyone ever logged one.
_CONFIG_FIELDS = frozenset({
    "config", "cfg", "resulting_config", "patches", "changes", "payload",
    "body", "row", "rows", "entry", "entries", "env", "environ", "headers",
})

_SECRET_PATTERNS = (
    # Deliberately looser than the tree scanner in tests/test_secret_lifecycle.py.
    # Over-redacting a log line costs a reader one lookup; under-redacting one puts a
    # credential in a retained log, so a short or truncated token still matches.
    re.compile(r"eyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}"),  # JWT
    re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}"),                                    # Anthropic
    re.compile(r"https://[a-z0-9]{12,}\.supabase\.co[^\s\"']*"),                # project URL
    re.compile(r"(?i)(api[_-]?key|apikey|authorization|bearer|password|secret|token)"
               r"[\"'\s:=]+[A-Za-z0-9_\-.]{8,}"),
)

_SECRET_ENV_VARS = ("SUPABASE_KEY", "SUPABASE_URL", "ANTHROPIC_API_KEY")

REDACTED = "[redacted]"
MAX_VALUE_CHARS = 512


def _shape(value) -> str:
    """What a forbidden value was, without being it."""
    if isinstance(value, dict):
        return f"<dict keys={len(value)}>"
    if isinstance(value, (list, tuple)):
        return f"<{type(value).__name__} len={len(value)}>"
    return f"<{type(value).__name__}>"


def scrub_text(text: str) -> str:
    """Remove credential-shaped substrings, and any live credential by exact match."""
    if not isinstance(text, str):
        return text
    for var in _SECRET_ENV_VARS:
        live = os.environ.get(var)
        # A one-character value would turn every message into redaction confetti.
        if live and len(live) >= 8:
            text = text.replace(live, REDACTED)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def sanitise(fields: dict) -> dict:
    """Apply both rules to one attribute mapping.

    A forbidden field keeps its NAME and loses its value: "a config was involved" is
    the useful half and "here is the config" is the dangerous half.
    """
    out = {}
    for key, value in fields.items():
        low = key.lower()
        if low in _MEASUREMENT_FIELDS or low in _CONFIG_FIELDS:
            out[key] = _shape(value)
            continue
        if isinstance(value, (dict, list, tuple)):
            # An undeclared container is a config or a metric batch often enough that
            # summarising it by default is the safer direction to be wrong in.
            out[key] = _shape(value)
            continue
        if isinstance(value, str):
            value = scrub_text(value)
            if len(value) > MAX_VALUE_CHARS:
                value = value[:MAX_VALUE_CHARS] + f"...<truncated {len(value)} chars>"
        out[key] = value
    return out


# -- log emission ---------------------------------------------------------------
_LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40, "off": 100}
_records: list[dict] = []
_capture = False


def _threshold() -> int:
    return _LEVELS.get(os.environ.get("ARGUS_LOG_LEVEL", "info").lower(), 20)


def log(event: str, level: str = "info", **fields) -> dict | None:
    """Emit one structured line carrying the bound correlation identifiers."""
    if _LEVELS.get(level, 20) < _threshold() and not _capture:
        return None
    record = {"ts": round(time.time(), 6), "level": level, "event": event}
    record.update(_CONTEXT.get())
    record.update(sanitise(fields))
    if _capture:
        _records.append(record)
    if _LEVELS.get(level, 20) >= _threshold():
        print(json.dumps(record, default=str), file=sys.stderr, flush=True)
    return record


@contextlib.contextmanager
def capture():
    """Collect records in memory instead of only writing them. For tests."""
    global _capture
    _records.clear()
    _capture = True
    try:
        yield _records
    finally:
        _capture = False


class Span:
    """The handle a span body uses to say how the work actually turned out.

    Needed because the interesting failures here do not propagate as exceptions.
    agent/tools.py catches a dependency timeout and returns a result dict, so a span
    that inferred its outcome from "did this block raise" recorded outcome="ok" for a
    call that timed out after 400 ms. A span that reports success for a failed
    dependency is worse than no span: it is a dashboard that says the system is fine.
    """

    __slots__ = ("id", "outcome")

    def __init__(self, span_id):
        self.id = span_id
        self.outcome = None

    def failed(self, outcome):
        self.outcome = outcome


@contextlib.contextmanager
def span(name: str, **fields):
    """A timed unit of work, correlated and measured.

    Not an OpenTelemetry span: it emits a start and an end record sharing a span id,
    with the duration on the end record. That is the part of tracing this system can
    use, and it costs one import of the standard library.
    """
    span_id = uuid.uuid4().hex[:12]
    handle = Span(span_id)
    started = time.perf_counter()
    with bind(span_id=span_id):
        log(f"{name}.start", level="debug", **fields)
        try:
            yield handle
        except BaseException as exc:
            log(f"{name}.end", level="error", outcome="exception",
                error_type=type(exc).__name__,
                duration_ms=round((time.perf_counter() - started) * 1000, 3), **fields)
            raise
        else:
            outcome = handle.outcome or "ok"
            log(f"{name}.end", level="info" if outcome == "ok" else "error",
                outcome=outcome,
                duration_ms=round((time.perf_counter() - started) * 1000, 3), **fields)


# -- metrics with a cardinality bound proved at declaration time ----------------
class CardinalityError(ValueError):
    """Raised for a label that would make the series count unbounded."""


# Identifier-shaped names. Each of these takes a new value per run, per attempt or per
# request, so one of them as a label multiplies the series count by the number of runs
# the system will ever do. They belong on the log record, where they cost one field.
FORBIDDEN_LABELS = frozenset({
    "run_id", "attempt_id", "span_id", "trace_id", "error_id", "request_id",
    "id", "uuid", "step", "timestamp", "ts", "path", "file", "metrics_file",
    "config_path", "training_dir", "name", "url", "user", "message",
})

MAX_SERIES_PER_METRIC = 64

_metrics: dict[str, dict] = {}
_series: dict[str, dict[tuple, float]] = {}

OTHER = "other"


def declare_counter(name: str, labels: dict[str, tuple] | None = None) -> None:
    """Declare a metric and the complete set of values each label may take.

    Declaring the value set, rather than only the label name, is what makes the bound
    real: the maximum series count is the product of those set sizes plus one for the
    `other` bucket per label, and it is knowable now rather than after a dashboard
    stops loading.
    """
    labels = labels or {}
    for label in labels:
        if label.lower() in FORBIDDEN_LABELS:
            raise CardinalityError(
                f"{name}: label {label!r} is an identifier. One series per value of it "
                f"means one series per run, forever. Put it on the log record instead."
            )
        if not isinstance(labels[label], (tuple, frozenset)):
            raise CardinalityError(
                f"{name}: label {label!r} must declare its permitted values as a tuple"
            )
    spec = {"labels": {k: tuple(v) for k, v in labels.items()}}
    projected = 1
    for values in spec["labels"].values():
        projected *= len(values) + 1  # +1 for the OTHER bucket
    if projected > MAX_SERIES_PER_METRIC:
        raise CardinalityError(
            f"{name}: {projected} possible series exceeds the cap of {MAX_SERIES_PER_METRIC}"
        )
    spec["max_series"] = projected
    _metrics[name] = spec
    _series.setdefault(name, {})


def max_series(name: str) -> int:
    return _metrics[name]["max_series"]


def incr(name: str, amount: float = 1.0, **labels) -> tuple:
    """Increment a declared counter, folding any undeclared label value into `other`."""
    spec = _metrics.get(name)
    if spec is None:
        raise CardinalityError(f"{name} was never declared")
    if set(labels) != set(spec["labels"]):
        raise CardinalityError(
            f"{name}: expected labels {sorted(spec['labels'])}, got {sorted(labels)}"
        )
    key = tuple(
        str(labels[label]) if str(labels[label]) in spec["labels"][label] else OTHER
        for label in sorted(spec["labels"])
    )
    series = _series[name]
    if key not in series and len(series) >= spec["max_series"]:  # pragma: no cover
        # Unreachable while declare_counter enforces the product bound. Kept as a
        # backstop because a silently unbounded metric is worse than a dropped one.
        log("obs.cardinality_capped", level="error", metric=name)
        return key
    series[key] = series.get(key, 0.0) + amount
    return key


def read_counter(name: str, **labels) -> float:
    spec = _metrics[name]
    key = tuple(
        str(labels[label]) if str(labels[label]) in spec["labels"][label] else OTHER
        for label in sorted(spec["labels"])
    )
    return _series[name].get(key, 0.0)


def snapshot() -> dict:
    return {name: dict(series) for name, series in _series.items()}


def reset() -> None:
    for name in _series:
        _series[name].clear()


def declared() -> dict:
    return dict(_metrics)


# -- the metrics this system actually keeps ------------------------------------
# Every label value set below is closed, so the whole registry's series count is
# bounded and the bound is asserted in tests/test_observability.py.
declare_counter("argus_ingest_rows_total", {"outcome": ("ok", "empty", "missing_file")})
declare_counter("argus_ingest_failures_total",
                {"dependency": ("supabase", "metrics_file"),
                 "reason": ("timeout", "connection", "auth", "protocol")})
declare_counter("argus_detection_anomalies_total",
                {"type": ("loss_spike", "grad_explosion", "val_plateau", "overfitting")})
declare_counter("argus_recovery_attempts_total",
                {"outcome": ("requested", "restarted", "progressed",
                             "corrected", "not_corrected")})
declare_counter("argus_dependency_calls_total",
                {"dependency": ("training_server", "supabase", "anthropic"),
                 "outcome": ("ok", "timeout", "connection", "http_error")})
