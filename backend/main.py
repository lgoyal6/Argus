from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routes import runs, metrics, decisions
from limits import (
    AdmissionMiddleware,
    BodyLimitMiddleware,
    WorkLimitExceeded,
)
import limits as limit_settings

app = FastAPI(title="AutoDebug API")

# ── abuse limits ───────────────────────────────────────────────────────────────
# Order is the point, not an accident. add_middleware inserts each new layer
# outermost, so this file reads inside-out: the running order is
# CORS -> BodyLimit -> Admission -> routes. That was checked rather than assumed;
# the first version of this comment had it backwards.
#
# BodyLimit sits outside Admission so an oversized or zip-bombed body is refused
# before it can occupy a concurrency slot. Otherwise "send huge bodies" would be a
# cheaper denial of service than sending real work.
#
# CORS sits outside both so that a 413 or a 503 still carries the CORS headers. A
# refusal a browser cannot read is reported to the dashboard as a network error
# instead of as the status the server actually chose, which hides exactly the
# behaviour these limits exist to make visible. The cost is that a preflight is not
# counted against the concurrency bound, which is acceptable: a preflight does no
# database work.
app.add_middleware(AdmissionMiddleware)
app.add_middleware(BodyLimitMiddleware)

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


# ── work bound ────────────────────────────────────────────────────────────────
@app.exception_handler(WorkLimitExceeded)
def work_limit_exceeded(request: Request, exc: WorkLimitExceeded):
    return JSONResponse(status_code=413, content={"detail": str(exc)})


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
        "max_body_bytes": limit_settings.MAX_BODY_BYTES,
        "max_decompressed_bytes": limit_settings.MAX_DECOMPRESSED_BYTES,
        "max_work_units": limit_settings.MAX_WORK_UNITS,
        "max_ingest_bytes": limit_settings.MAX_INGEST_BYTES,
        "request_timeout_s": limit_settings.REQUEST_TIMEOUT_S,
        "max_queue": limit_settings.MAX_QUEUE,
        "max_concurrency": limit_settings.MAX_CONCURRENCY,
        "reserved_light": limit_settings.RESERVED_LIGHT,
    }
