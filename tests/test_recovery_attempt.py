"""What a recovery attempt is allowed to claim.

The agent's old success condition was `patched and rerun_succeeded`, where
`rerun_succeeded` meant the rerun server had returned HTTP 200 from a handler that
answers before its subprocess does anything. Under that rule a trainer that died on
its first line was logged as "fixed".

Each test below is one of the four claims being kept apart: that a request was
accepted, that a process ran, that it trained, and that the anomaly is gone. The
scenarios are written as the situations that used to be scored identically.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.attempt import (
    CANCELLED,
    CORRECTED,
    NOT_CORRECTED,
    PROGRESSED,
    REQUESTED,
    RESOLVED,
    RESTARTED,
    TIMED_OUT,
    Attempt,
    load_ledger,
)


# ── fixtures ───────────────────────────────────────────────────────────────────
def row(step, train_loss=0.5, val_loss=0.6, grad_norm=1.0, val_acc=0.5):
    return {"step": step, "epoch": 0, "train_loss": train_loss, "val_loss": val_loss,
            "val_acc": val_acc, "grad_norm": grad_norm, "timestamp": 1000.0 + step}


def write(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def metrics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    # A healthy prefix, then the grad explosion that triggers the attempt.
    write(path, [row(s, grad_norm=1.0 + s * 0.01) for s in range(1, 21)])
    write(path, [row(21, grad_norm=150.0)])
    return path


@pytest.fixture
def attempt(tmp_path, metrics):
    trigger = {"type": "grad_explosion", "step": 21, "description": "grad norm 150.0"}
    a = Attempt(run_id="run-1", trigger=trigger, ledger_path=tmp_path / "attempts.jsonl")
    a.capture_baseline(metrics)
    return a


# ── the four claims ────────────────────────────────────────────────────────────
def test_an_accepted_request_with_a_silent_trainer_is_not_a_restart(attempt, metrics):
    """The exact shape of the old bug: the launch died, nothing was written.

    The rerun server said "requested" and the trainer never appended a row, because
    the argument vector it was launched with killed it. This must stay at REQUESTED.
    """
    attempt.record_action({"tool": "patch_config", "changes": {"training.learning_rate": 1e-5}})

    assert attempt.observe(metrics) == REQUESTED
    assert attempt.fixed is False
    assert attempt.status() == "patched"


def test_a_single_written_row_is_a_restart_but_not_progress(attempt, metrics):
    """A trainer that came up, emitted once and died has not trained.

    One row proves a process existed. It does not prove optimizer steps happened
    between two emissions, which is the cheapest evidence of actual training.
    """
    attempt.record_action({"tool": "patch_config", "changes": {}})
    write(metrics, [row(10, grad_norm=2.0)])

    assert attempt.observe(metrics) == RESTARTED
    assert attempt.fixed is False


def test_progress_with_the_trigger_still_firing_is_not_a_fix(attempt, metrics):
    """It trained, and the anomaly is still there. That is progress, not correction."""
    attempt.record_action({"tool": "patch_config", "changes": {}})
    write(metrics, [row(s, grad_norm=1.0) for s in range(10, 40, 10)])
    write(metrics, [row(40, grad_norm=180.0)])  # explodes again

    assert attempt.observe(metrics) == NOT_CORRECTED
    assert attempt.fixed is False
    assert attempt.status() == "patched"


def test_only_observed_correction_counts_as_fixed(attempt, metrics):
    """Trained, and the triggering anomaly no longer fires in the new rows."""
    attempt.record_action({"tool": "patch_config", "changes": {"training.gradient_clip": 0.5}})
    write(metrics, [row(s, grad_norm=1.0) for s in range(10, 60, 10)])

    assert attempt.observe(metrics) == CORRECTED
    assert attempt.terminal == RESOLVED
    assert attempt.fixed is True
    assert attempt.status() == "fixed"


# ── the evidence channel ───────────────────────────────────────────────────────
def test_the_rerun_servers_own_report_cannot_raise_the_outcome(attempt, metrics):
    """Nothing the training server says is an input to the ladder.

    The component under judgement does not get to grade itself. Feeding in the most
    emphatic possible success report still leaves the attempt where the metrics put
    it, because observe() never reads it.
    """
    attempt.record_action(
        {"tool": "rerun_training", "response": {"status": "started", "ok": True,
                                                "returncode": 0, "fixed": True}}
    )

    assert attempt.observe(metrics) == REQUESTED
    assert attempt.fixed is False


def test_a_restarted_run_that_renumbers_steps_from_zero_still_counts_as_progress(
    attempt, metrics
):
    """The trainer resets global_step to 0 on a rerun.

    Its rows therefore carry LOWER step numbers than the tail they follow. A watermark
    keyed on "the step went up" would read a genuine restart as no progress at all,
    which is why the baseline is a row count.
    """
    assert attempt.baseline["last_step"] == 21
    write(metrics, [row(s, grad_norm=1.0) for s in (10, 20, 30)])

    assert attempt.observe(metrics) == CORRECTED


# ── termination is a separate axis ─────────────────────────────────────────────
def test_timeout_and_cancellation_are_distinct_outcomes(tmp_path, metrics):
    """Both end uncorrected, and a retry policy must tell them apart.

    "we stopped asking" and "it never answered" call for opposite responses.
    """
    trigger = {"type": "grad_explosion", "step": 21}
    ledger = tmp_path / "attempts.jsonl"

    timed_out = Attempt("run-1", trigger, ledger_path=ledger)
    timed_out.capture_baseline(metrics)
    ticks = iter([0, 1, 2, 3])
    timed_out.observe_until(metrics, deadline=2, clock=lambda: next(ticks))

    cancelled = Attempt("run-1", trigger, ledger_path=ledger)
    cancelled.capture_baseline(metrics)
    cancelled.cancel(reason="operator stopped the run")

    assert timed_out.terminal == TIMED_OUT
    assert cancelled.terminal == CANCELLED
    assert timed_out.terminal != cancelled.terminal
    assert timed_out.fixed is False and cancelled.fixed is False


def test_observe_until_stops_as_soon_as_it_is_corrected(attempt, metrics):
    polls = []

    def poll():
        polls.append(1)
        write(metrics, [row(s, grad_norm=1.0) for s in (10, 20, 30)])

    ticks = iter([0, 1, 2, 3, 4])
    assert attempt.observe_until(
        metrics, deadline=4, poll=poll, clock=lambda: next(ticks)
    ) == CORRECTED
    assert attempt.terminal == RESOLVED
    assert len(polls) == 1  # stopped on the first corrected observation


# ── provenance and durability ──────────────────────────────────────────────────
def test_the_ledger_survives_a_crash_at_the_rung_it_had_earned(attempt, metrics, tmp_path):
    """An attempt in flight when the agent dies is on disk, uncorrected.

    The failure mode this guards against is an attempt that is lost entirely, or one
    that is recovered at the rung it was hoping for rather than the one it reached.
    """
    attempt.record_action({"tool": "patch_config", "changes": {"training.gradient_clip": 0.5}},
                          resulting_config={"training": {"gradient_clip": 0.5}})
    attempt.observe(metrics)  # nothing written yet -> REQUESTED

    del attempt  # the agent process goes away here

    (recovered,) = load_ledger(tmp_path / "attempts.jsonl")
    assert recovered["outcome"] == REQUESTED
    assert recovered["fixed"] is False
    assert recovered["trigger"]["type"] == "grad_explosion"
    assert recovered["approved_action"]["changes"] == {"training.gradient_clip": 0.5}
    assert recovered["resulting_config"] == {"training": {"gradient_clip": 0.5}}
    assert recovered["baseline"]["row_count"] == 21


def test_the_ledger_survives_a_real_sigkill_mid_attempt(tmp_path, metrics):
    """The same durability, against a signal the process cannot handle.

    The test above simulates the crash with `del`, which still unwinds normally: file
    objects are finalised, buffers flush, atexit runs. None of that happens under
    SIGKILL, and SIGKILL is what an OOM kill or a `docker stop` timeout actually
    sends. Since the decision to keep this state machine instead of adopting a durable
    workflow engine rests on the ledger surviving exactly that, it is worth asserting
    against the real signal rather than against a polite stand-in.
    """
    import subprocess

    ledger = tmp_path / "attempts.jsonl"
    script = f"""
import os, signal, sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
from agent.attempt import Attempt

a = Attempt(run_id="run-1", trigger={{"type": "grad_explosion", "step": 21}},
            ledger_path={str(ledger)!r})
a.capture_baseline({str(metrics)!r})
a.record_action({{"tool": "patch_config", "changes": {{"training.gradient_clip": 0.5}}}},
                resulting_config={{"training": {{"gradient_clip": 0.5}}}})
a.observe({str(metrics)!r})          # nothing new written yet -> REQUESTED
os.kill(os.getpid(), signal.SIGKILL)  # uncatchable: no flush, no atexit, no unwind
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    # -9 confirms the process really was killed rather than exiting cleanly, which
    # would make this test pass for the wrong reason.
    assert proc.returncode == -9, f"expected SIGKILL, got {proc.returncode}: {proc.stderr}"

    (recovered,) = load_ledger(ledger)
    assert recovered["outcome"] == REQUESTED      # the rung it earned, not the one it wanted
    assert recovered["fixed"] is False
    assert recovered["approved_action"]["changes"] == {"training.gradient_clip": 0.5}
    assert recovered["baseline"]["row_count"] == 21


def test_an_outcome_never_walks_back_down_the_ladder(attempt, metrics):
    """A later, quieter observation cannot retract evidence already earned."""
    write(metrics, [row(s, grad_norm=1.0) for s in (10, 20, 30)])
    assert attempt.observe(metrics) == CORRECTED

    attempt.baseline["row_count"] = 99  # a window with nothing in it
    assert attempt.observe(metrics) == CORRECTED


def test_observing_before_a_baseline_is_a_programming_error(tmp_path, metrics):
    """Without a pre-action watermark, "new rows" has no meaning."""
    a = Attempt("run-1", {"type": "loss_spike"}, ledger_path=tmp_path / "l.jsonl")
    with pytest.raises(ValueError, match="capture_baseline"):
        a.observe(metrics)


def test_progress_alone_never_reports_a_fix_for_a_different_anomaly(attempt, metrics):
    """Correction is judged against the trigger, not against "any anomaly".

    A run that stops exploding but starts overfitting has corrected the thing this
    attempt was for; a run still exploding has not, whatever else improved.
    """
    write(metrics, [row(s, grad_norm=1.0, train_loss=0.1, val_loss=0.9) for s in (10, 20, 30)])
    assert attempt.observe(metrics) == CORRECTED
    assert "grad_explosion" not in attempt.observations[-1]["anomalies_after"]
