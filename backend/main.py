from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from routes import runs, metrics, decisions

import obs

app = FastAPI(title="AutoDebug API")

# No-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set, so the default deployment is
# unchanged and nobody needs a collector to serve the dashboard.
obs.setup_tracing("argus-backend")


# ── request entry ──────────────────────────────────────────────────────────────
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
