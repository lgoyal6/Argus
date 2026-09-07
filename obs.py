"""Correlated logs and bounded metrics for the ingest, detection and recovery path.

Argus is three processes that only ever meet through a run id: the backend ingests a
metrics file, the agent detects anomalies in it, and a recovery attempt acts on what it
found. When one of those stalls, the question is always "which dependency, on which
run" - and until now the answer had to be reconstructed from `print()` output with no
identifiers in it at all.

The correlation core below has no dependencies and is always on: a deployment that
installs nothing extra still gets identifiers on every record. On top of it sits an
opt-in OpenTelemetry layer that stays inert until `OTEL_EXPORTER_OTLP_ENDPOINT` is
set.

This module used to argue against that layer, by analogy with the project's decision
to reject Temporal for costing a server and 51 MB to buy one property it could write
itself. The analogy was wrong on the fact that decides it. Temporal bought durable
retry, which this code could implement; OpenTelemetry buys the assembled cross-process
view, which the records provably cannot produce, because a span id with no parent link
says a stage was slow and never which caller it was slow for. It also costs no server
of its own. Measured here at 20,000 spans per sample, 7 samples, median: 3.94us per
span with no OTel wiring at all, 4.51us wired with no collector configured, 38.84us
wired and exporting. So the deployment that does not opt in pays 0.57us per span, and
the one that does pays about 10x, which is what `ARGUS_TRACE_SAMPLE_RATIO` is for: the
trainer opens a span per emitted step, so at ratio 1.0 a 10,000-step run is 10,000
traces.

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

CORRELATION_KEYS = ("run_id", "attempt_id", "span_id", "trace_id")


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


# -- optional OpenTelemetry export ---------------------------------------------
# Why this exists on top of the records above, and why it is not a replacement for
# them.
#
# The records answer "what happened, on which run". They cannot answer "which of the
# four processes was slow", because Argus's processes only ever meet through a file:
# the trainer appends to metrics.jsonl, the agent polls it, and the backend ingests
# it into the database. Correlating across that hop means agreeing on a wire format
# for the parent context, and W3C `traceparent` is that format already, with an
# extractor, an injector and a viewer that draws the assembled result. The viewer is
# most of the value and is the part a hand-rolled span id cannot reach: a span id
# with no parent link tells you a stage was slow, never which caller it was slow for.
#
# The measured cost of being wired but off, and of being on, is in the module
# docstring above.
#
# Two rules carried over unchanged rather than restated:
#   * every attribute goes through `sanitise` first, so a config or a credential
#     cannot reach a span any more than it can reach a log line;
#   * nothing here emits a metric, so the series bound proved by `declare_counter`
#     is untouched. Identifiers belong on spans; that is the whole point of both.

_provider = None
_otel_on = False

_SPAN_KINDS = ("internal", "server", "client", "producer", "consumer")


def setup_tracing(service_name: str, endpoint: str | None = None, exporter=None):
    """Install a tracer provider if a collector is configured, else nothing.

    Returns a shutdown callable in both cases, so callers have no branch. No
    `OTEL_EXPORTER_OTLP_ENDPOINT` means every span below stays exactly what it was
    before this section existed: two log records and a `time.perf_counter()`.

    `exporter` exists so a test can read back the spans this module actually emits.
    The redaction rules are only worth anything at the point telemetry LEAVES the
    process, and asserting on `sanitise` instead asserts on the input to the wiring
    rather than on its output - which is exactly the gap that let an unscrubbed
    exception message reach a span in the first place.
    """
    global _provider, _otel_on

    endpoint = endpoint if endpoint is not None else os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint and exporter is None:
        _otel_on = False
        return lambda: None

    from opentelemetry import trace as _trace
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    set_global_textmap(TraceContextTextMapPropagator())
    # The trainer opens a span per emitted step, so at ratio 1.0 a 10,000-step run is
    # 10,000 traces. ParentBased is what keeps a sampled trace whole: once the root is
    # in, every downstream process keeps its part of it rather than each rolling its
    # own dice and leaving the trace with holes.
    ratio = float(os.environ.get("ARGUS_TRACE_SAMPLE_RATIO", "1.0"))
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name}),
        sampler=ParentBased(TraceIdRatioBased(ratio)),
    )
    if exporter is None:
        # Imported here rather than above, because a caller that supplies its own sink
        # has no use for a network exporter and should not have to install one. That
        # is not hypothetical: it is how the tests read back the spans this module
        # emits, and importing it unconditionally made every one of them an error on
        # an install that had opentelemetry-sdk and nothing else.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces"))
        )
    else:
        # An explicitly supplied sink is one the caller means to read back, so it is
        # attached synchronously. Batching it would make every read a race with the
        # exporter's own timer.
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider.add_span_processor(SimpleSpanProcessor(exporter))
    _trace.set_tracer_provider(provider)
    _provider = provider
    _otel_on = True

    def shutdown() -> None:
        global _otel_on
        provider.shutdown()
        _otel_on = False

    return shutdown


def tracing_enabled() -> bool:
    return _otel_on


def flush(timeout_ms: int = 5_000) -> None:
    """Push queued spans now. The agent loop and the trainer are killed rather than
    returned from, and a batch processor that only flushes on its own timer loses the
    last trace of every run - which is the one somebody is looking for."""
    if _provider is not None:
        _provider.force_flush(timeout_ms)


def _otel_attrs(fields: dict) -> dict:
    """Sanitised fields, narrowed to what the SDK will accept.

    `sanitise` is the same function the log records go through, so an attribute cannot
    carry anything a log line could not. The narrowing on top of it drops None and any
    residual container, which the SDK would otherwise warn about once per span.
    """
    return {
        key: value
        for key, value in sanitise(fields).items()
        if isinstance(value, (str, bool, int, float))
    }


@contextlib.contextmanager
def _otel_span(name: str, kind: str, parent: str | None, fields: dict):
    if not _otel_on:
        yield None
        return

    from opentelemetry import trace as _trace
    from opentelemetry.trace import SpanKind

    kinds = {
        "internal": SpanKind.INTERNAL,
        "server": SpanKind.SERVER,
        "client": SpanKind.CLIENT,
        "producer": SpanKind.PRODUCER,
        "consumer": SpanKind.CONSUMER,
    }
    context = _context_from(parent) if parent else None
    tracer = _trace.get_tracer("argus")
    with tracer.start_as_current_span(
        name, context=context, kind=kinds[kind], attributes=_otel_attrs(fields),
        record_exception=False, set_status_on_exception=False,
    ) as raw:
        yield raw


def current_trace_id() -> str | None:
    if not _otel_on:
        return None
    from opentelemetry import trace as _trace

    ctx = _trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


def current_traceparent() -> str | None:
    """The ambient context as one header value, or None when tracing is off.

    This is what travels: into an HTTP header on the way to the backend or the
    training server, and into the `_trace` field of a metrics row on the way through
    the file that stands in for a queue.
    """
    if not _otel_on:
        return None
    from opentelemetry.propagate import inject

    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier.get("traceparent")


# The header name a queued message carries its parent context under. metrics.jsonl
# is the only channel between the trainer, the agent and the backend, so a row on it
# is a message and this is its one header. It is stripped by `take_trace` before the
# row reaches the metrics table: the database stores measurements, not telemetry.
TRACE_FIELD = "_trace"


def attach_trace(payload: dict) -> dict:
    """The row as it goes onto the queue, carrying the current context if there is one."""
    traceparent = current_traceparent()
    return {**payload, TRACE_FIELD: traceparent} if traceparent else payload


def inherited_or(traceparent: str | None) -> str | None:
    """The parent to use for a consumer span: whoever asked, or the message itself.

    A queue consumer has two candidate parents and must not take both. When something
    upstream is already in scope - the agent's poll, or an inbound request that
    carries context - that caller IS the parent and the span inherits it, so passing
    the message's own context as well would move the span into a different trace from
    its own caller and split the request in two. Only when nothing is in scope does
    the message's header become the parent, which is the honest reading of "this work
    exists because that row arrived".
    """
    return None if current_trace_id() else traceparent


def take_trace(payload: dict) -> tuple[dict, str | None]:
    """Split a queued row into the row itself and the context it travelled with."""
    if TRACE_FIELD not in payload:
        return payload, None
    row = {k: v for k, v in payload.items() if k != TRACE_FIELD}
    return row, payload[TRACE_FIELD]


def _context_from(traceparent: str):
    from opentelemetry.propagate import extract

    return extract({"traceparent": traceparent})


class Span:
    """The handle a span body uses to say how the work actually turned out.

    Needed because the interesting failures here do not propagate as exceptions.
    agent/tools.py catches a dependency timeout and returns a result dict, so a span
    that inferred its outcome from "did this block raise" recorded outcome="ok" for a
    call that timed out after 400 ms. A span that reports success for a failed
    dependency is worse than no span: it is a dashboard that says the system is fine.
    """

    __slots__ = ("id", "outcome", "_otel")

    def __init__(self, span_id, otel=None):
        self.id = span_id
        self.outcome = None
        self._otel = otel

    def failed(self, outcome):
        self.outcome = outcome
        if self._otel is not None:
            from opentelemetry.trace import Status, StatusCode

            self._otel.set_status(Status(StatusCode.ERROR, outcome))
            self._otel.set_attribute("outcome", outcome)

    def set(self, **fields):
        """Attach late-known attributes, if a collector is listening."""
        if self._otel is not None:
            for key, value in _otel_attrs(fields).items():
                self._otel.set_attribute(key, value)

    def rename(self, name):
        """Rename after the fact, for a request span that only learns its route
        template once routing has happened. `GET /runs/9f2c1a/metrics` as an
        operation name makes every run its own operation in the viewer's dropdown,
        and the list stops being usable after a week; the concrete path stays on
        `url.path`, where high cardinality is free."""
        if self._otel is not None:
            self._otel.update_name(name)


@contextlib.contextmanager
def span(name: str, *, kind: str = "internal", parent: str | None = None, **fields):
    """A timed unit of work, correlated and measured.

    Always emits the start and end records that this module has always emitted, so a
    deployment with no collector is unchanged. When one IS configured it additionally
    opens a real OpenTelemetry span around the same block, and binds that span's trace
    id into the correlation context - which is what makes a log line and a span in the
    viewer findable from each other.

    `parent` is a W3C traceparent string, for the consumer side of a queue hop where
    there is no ambient context to inherit from.
    """
    span_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    with _otel_span(name, kind, parent, fields) as raw:
        handle = Span(span_id, raw)
        ids = {"span_id": span_id}
        trace_id = current_trace_id()
        if trace_id:
            ids["trace_id"] = trace_id
        with bind(**ids):
            log(f"{name}.start", level="debug", **fields)
            try:
                yield handle
            except BaseException as exc:
                if raw is not None:
                    from opentelemetry.trace import Status, StatusCode

                    raw.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                    # Through the scrubber, like everything else. The SDK's own
                    # exception recorder is turned off in _otel_span because it puts
                    # `str(exc)` on the span untouched, and an exception message here
                    # is a string this module has never seen: a Supabase URL, a signed
                    # request, a config echoed back by a dependency.
                    raw.add_event("exception", _otel_attrs({
                        "exception.type": type(exc).__name__,
                        "exception.message": str(exc),
                    }))
                log(f"{name}.end", level="error", outcome="exception",
                    error_type=type(exc).__name__,
                    duration_ms=round((time.perf_counter() - started) * 1000, 3), **fields)
                raise
            else:
                outcome = handle.outcome or "ok"
                if raw is not None:
                    raw.set_attribute("outcome", outcome)
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
