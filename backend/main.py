from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import Match

from errors import InternalError
from limits import AdmissionMiddleware, BodyLimitMiddleware, WorkLimitExceeded
from routes import runs, metrics, decisions

# Plain, and separately from the names above, so that `limits.MAX_BODY_BYTES` is read
# from the module rather than copied at import: tests/test_abuse_limits.py reconfigures
# by re-importing, which is what an operator does by restarting the server.
import limits
import obs

app = FastAPI(title="AutoDebug API")

# No-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set, so the default deployment is
# unchanged and nobody needs a collector to serve the dashboard.
obs.setup_tracing("argus-backend")


# ── error shaping ──────────────────────────────────────────────────────────────
@app.exception_handler(InternalError)
async def _internal_error(_request: Request, exc: InternalError) -> JSONResponse:
    """Write the flat `ErrorResponse` shape the committed contract declares.

    Without this, FastAPI's default HTTPException handler nests the whole payload
    under `detail` and the response stops matching the schema the document publishes
    for 500. Starlette resolves handlers by walking `type(exc).__mro__`, so this one
    is found before the generic HTTPException handler below.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_id": exc.error_id},
    )


def _utf8_safe(value):
    """Replace anything in a validation error that cannot be encoded as UTF-8.

    FastAPI's 422 body echoes the offending input back to the caller. When the input
    is a lone surrogate - a valid `str`, an invalid UTF-8 sequence - rendering that
    body raises inside the response encoder, so the refusal the service correctly
    decided on never reaches the client and a bare text/plain 500 goes out instead.
    Rejecting the value in the schema is therefore not enough on its own: the error
    path has to be able to describe what it rejected.
    """
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {_utf8_safe(k): _utf8_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_utf8_safe(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content=_utf8_safe(jsonable_encoder({"detail": exc.errors()})))


def _leaf_routes(node):
    """Every method-bearing route under `node`.

    `app.routes` is not a flat list. This FastAPI version keeps an `include_router`
    call as a single nested `_IncludedRouter` entry that exposes its children through
    `effective_candidates()`, and Starlette's own `Mount` exposes them through
    `routes`; older FastAPI flattened everything into `app.routes` instead. Walking
    both shapes keeps the Allow header correct across all three rather than silently
    reporting only the handful of routes that happen to sit at the top level - which
    is what the first version of this did, reporting `Allow: GET` for a path that also
    accepts POST.
    """
    methods = getattr(node, "methods", None)
    if methods and hasattr(node, "matches"):
        yield node
        return
    children = getattr(node, "routes", None)
    if children is None:
        candidates = getattr(node, "effective_candidates", None)
        if callable(candidates):
            try:
                children = candidates()
            except Exception:  # pragma: no cover - defensive against internals moving
                children = None
    for child in children or ():
        yield from _leaf_routes(child)


def _methods_for(request: Request) -> list[str]:
    """The methods that would have matched this path, for the `Allow` header."""
    allowed: set[str] = set()
    for route in _leaf_routes(request.app.router):
        try:
            match, _ = route.matches(request.scope)
        except Exception:  # pragma: no cover - defensive
            continue
        # PARTIAL is Starlette's "the path matched, the method did not", which is
        # exactly the set a 405 has to report.
        if match in (Match.FULL, Match.PARTIAL):
            allowed |= set(route.methods)
    return sorted(allowed)


@app.exception_handler(StarletteHTTPException)
async def _http_exception(request: Request, exc: StarletteHTTPException):
    """Make `Allow` on a 405 list every method the resource takes, not just one.

    RFC 9110 requires a 405 to name the supported set. Starlette does send an `Allow`,
    but it is the method list of the *first* route whose path matched, so
    `OPTIONS /runs/` answered `Allow: GET` on a resource that also accepts POST. A
    client that follows the header to find the right method is told the wrong thing,
    which is worse than being told nothing. Schemathesis' `allow_header_conformance`
    check compares the header against the documented operations and caught it; the
    hand-rolled corpus sends no OPTIONS request and reads no response header, so it
    could not have.
    """
    response = await http_exception_handler(request, exc)
    if exc.status_code == 405:
        methods = _methods_for(request)
        if methods:
            response.headers["Allow"] = ", ".join(methods)
    return response



# ── abuse limits ───────────────────────────────────────────────────────────────
# Order is the point, not an accident. add_middleware inserts each new layer
# outermost, so this file reads inside-out and the running order is
#   CORS -> trace_request -> BodyLimit -> Admission -> routes.
#
# BodyLimit sits outside Admission so an oversized or zip-bombed body is refused
# before it can occupy a concurrency slot. Otherwise "send huge bodies" would be a
# cheaper denial of service than sending real work.
#
# trace_request sits outside both so a 413 or a 503 is a span with a status, rather
# than a refusal that never appears in a trace at all.
#
# CORS sits outside everything so that a 413 or a 503 still carries the CORS headers.
# A refusal a browser cannot read is reported to the dashboard as a network error
# instead of as the status the server actually chose, which hides exactly the
# behaviour these limits exist to make visible. The cost is that a preflight is not
# counted against the concurrency bound, which is acceptable: a preflight does no
# database work.
app.add_middleware(AdmissionMiddleware)
app.add_middleware(BodyLimitMiddleware)


# ── work bound ─────────────────────────────────────────────────────────────────
@app.exception_handler(WorkLimitExceeded)
def work_limit_exceeded(_request: Request, exc: WorkLimitExceeded) -> JSONResponse:
    """413 in the flat shape the contract declares, like every other error here.

    error_id is null rather than absent: this refusal is fully described by its own
    message, and there is no process-log line to correlate it with.
    """
    return JSONResponse(status_code=413,
                        content={"detail": str(exc), "error_id": None})

# ── request entry ──────────────────────────────────────────────────────────
@app.on_event("shutdown")
def flush_telemetry():
    """Push queued spans before the process goes away.

    The exporter batches on its own timer, so a backend that is stopped between
    timer ticks loses the spans of the requests it just served - which are the ones
    somebody restarting it is about to go looking for.
    """
    obs.flush()


@app.middleware("http")
async def trace_request(request: Request, call_next):
    """The entry span, and the only place a caller's own trace context is picked up.

    The agent calls /runs/{run_id}/metrics/sync every poll, so this is where an
    ingest stops being the agent's problem and becomes the backend's - and, without
    this, where the trace ended.
    """
    with obs.span(f"{request.method} {request.url.path}", kind="server",
                  parent=request.headers.get("traceparent"),
                  **{"http.request.method": request.method,
                     "url.path": request.url.path}) as span:
        response = await call_next(request)
        route = request.scope.get("route")
        if getattr(route, "path", None):
            span.rename(f"{request.method} {route.path}")
            span.set(**{"http.route": route.path})
        span.set(**{"http.response.status_code": response.status_code})
        if response.status_code >= 500:
            span.failed(f"http_{response.status_code}")
        # So an operator reading a 500 in the access log can go straight to the
        # trace instead of guessing which one it was.
        trace_id = obs.context().get("trace_id")
        if trace_id:
            response.headers["x-trace-id"] = trace_id
        return response

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://3.23.64.179:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── routers ────────────────────────────────────────────────────────────────────
app.include_router(runs.router, prefix="/runs", tags=["runs"])
app.include_router(metrics.router, prefix="/runs", tags=["metrics"])
app.include_router(decisions.router, prefix="/runs", tags=["decisions"])


# ── health check ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "AutoDebug API"}


# ── the limits in force, so an operator can read them off a running server ─────
@app.get("/limits")
def limits_in_force():
    return {
        "max_body_bytes": limits.MAX_BODY_BYTES,
        "max_decompressed_bytes": limits.MAX_DECOMPRESSED_BYTES,
        "max_work_units": limits.MAX_WORK_UNITS,
        "max_ingest_bytes": limits.MAX_INGEST_BYTES,
        "request_timeout_s": limits.REQUEST_TIMEOUT_S,
        "max_queue": limits.MAX_QUEUE,
        "max_concurrency": limits.MAX_CONCURRENCY,
        "reserved_light": limits.RESERVED_LIGHT,
    }
