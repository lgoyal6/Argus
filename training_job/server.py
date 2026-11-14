from fastapi import FastAPI, Request
import subprocess, threading, time
from pathlib import Path

from cli import build_training_argv
from launches import EXITED, LAUNCH_FAILED, RUNNING, TIMEOUT, LaunchRegistry

# Flat in the image, beside train.py: docker/Dockerfile.training copies obs.py in.
import obs

app = FastAPI()

# No-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set.
obs.setup_tracing("argus-training")


@app.on_event("shutdown")
def flush_telemetry():
    """Push queued spans before the process goes away.

    The exporter batches on its own timer, so a training server that is stopped between
    timer ticks loses the spans of the requests it just served - which are the ones
    somebody restarting it is about to go looking for.
    """
    obs.flush()


@app.middleware("http")
async def trace_request(request: Request, call_next):
    """The entry span for a recovery. The agent injects its context on the way in,
    so a rerun request and the detection cycle that decided to make it are one
    trace rather than two."""
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
        return response

# ── launch bookkeeping ─────────────────────────────────────────────────────────
# /rerun answers before the trainer has done anything, so its response can only
# honestly say the request was accepted. It used to say {"status": "started"} and
# discard the exit status inside the thread, which meant a trainer that died on its
# first line was indistinguishable from one that trained. Exit statuses now land in
# the registry and can be asked for at /launches/{id}.
#
# This is still only evidence that a process ran. Whether the run made progress is a
# question for the metrics the trainer writes, not for this endpoint - see
# agent/attempt.py, which deliberately does not trust anything on this page.
_registry = LaunchRegistry()

TRAINING_DIR = Path(__file__).resolve().parent


def run_training(launch_id, max_steps, traceparent=None):
    argv = ["python3", *build_training_argv(max_steps)]
    _registry.update(launch_id, state=RUNNING, argv=argv, started_at=time.time())
    # The trainer is a subprocess, so it inherits no ambient context. TRACEPARENT in
    # the environment is the conventional channel for exactly this hop, and without
    # it the rows the trainer writes carry a context unrelated to the request that
    # asked for them.
    env = None
    if traceparent:
        import os as _os

        env = {**_os.environ, "TRACEPARENT": traceparent}
    try:
        completed = subprocess.run(
            argv, cwd=TRAINING_DIR, capture_output=True, text=True, timeout=3600,
            env=env,
        )
    except subprocess.TimeoutExpired:
        _registry.update(launch_id, state=TIMEOUT, finished_at=time.time())
        return
    except Exception as e:  # the interpreter or the script is missing
        _registry.update(launch_id, state=LAUNCH_FAILED, error=str(e),
                         finished_at=time.time())
        return
    _registry.update(
        launch_id,
        state=EXITED,
        returncode=completed.returncode,
        # Truncated: this is for diagnosing a launch that died, not a log sink.
        stderr_tail=completed.stderr[-2000:],
        finished_at=time.time(),
    )


@app.post("/rerun")
def rerun(max_steps: int = 50, attempt_id: str = None):
    """Start a trainer for `attempt_id`, or return the one already running for it.

    Keyed on the attempt rather than on the request: the caller's timeout is shorter
    than the work, so a retried request must not start a second trainer writing into
    the same metrics file.
    """
    launch, created = _registry.request(attempt_id, max_steps)
    if created:
        threading.Thread(target=run_training,
                         args=(launch["launch_id"], max_steps,
                               obs.current_traceparent()),
                         daemon=True).start()
    # "requested", not "started": at this point nothing has been observed to run.
    return {"status": "requested", "launch_id": launch["launch_id"],
            "duplicate": not created}


@app.get("/launches/{launch_id}")
def launch_status(launch_id):
    return _registry.get(launch_id)


@app.get("/health")
def health():
    return {"status": "ok"}
