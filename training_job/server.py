from fastapi import FastAPI
import subprocess, threading, time
from pathlib import Path

from cli import build_training_argv
from launches import EXITED, LAUNCH_FAILED, RUNNING, TIMEOUT, LaunchRegistry

app = FastAPI()

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


def run_training(launch_id, max_steps):
    argv = ["python3", *build_training_argv(max_steps)]
    _registry.update(launch_id, state=RUNNING, argv=argv, started_at=time.time())
    try:
        completed = subprocess.run(
            argv, cwd=TRAINING_DIR, capture_output=True, text=True, timeout=3600
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
        threading.Thread(target=run_training, args=(launch["launch_id"], max_steps),
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
