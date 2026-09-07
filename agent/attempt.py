"""What a recovery attempt actually achieved, and how we know.

The agent used to decide it had fixed a training run like this:

    if tool == "rerun_training" and result.get("status") == "started":
        rerun_succeeded = True
    ...
    status = "fixed" if (patched and rerun_succeeded) else ...

Every term in that is a claim nobody checked. "started" was the rerun server's HTTP
response, returned from a handler that had already answered before the subprocess it
spawned did anything, and whose exit status was discarded in a daemon thread. The
launch in fact died every time on an argument-vector mismatch (see training_job/cli.py),
so "fixed" was recorded for runs where the trainer never completed a single step.

So the vocabulary here separates four claims that were being collapsed into one, in
increasing order of what they cost to earn:

    REQUESTED   we sent an action and something accepted it
    RESTARTED   a process wrote to the metrics stream after we asked
    PROGRESSED  it wrote enough, with advancing steps, to have actually trained
    CORRECTED   and the anomaly that triggered this attempt no longer fires

A requested action is not a restart, a restarted process is not progress, and progress
is not a corrected outcome. Only CORRECTED may be reported as fixed.

The evidence for every rung above REQUESTED comes from the metrics file the trainer
writes, never from the rerun server's reply. That is the point: the component being
judged does not get to grade itself. If the training server were to answer "started"
for a process that never existed, the ladder stops at REQUESTED and says so.

Attempts are appended to a JSONL ledger as they change, so an attempt in flight when
the agent dies is still on disk afterwards, at the last rung it had earned rather than
at the one it was hoping for.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

try:  # imported as a package by the tests, flat by agent/loop.py
    from agent.detector import detect_anomalies_in
except ImportError:  # pragma: no cover - exercised only by the flat layout
    from detector import detect_anomalies_in

# Flat in both layouts; see the note in agent/detector.py.
import obs


# ── outcome vocabulary ─────────────────────────────────────────────────────────
REQUESTED = "requested"
RESTARTED = "restarted"
PROGRESSED = "progressed"
CORRECTED = "corrected"
NOT_CORRECTED = "not_corrected"

# Ordered so a later observation can only ever raise the rung, never quietly lower a
# claim we already had evidence for. NOT_CORRECTED sits level with PROGRESSED: it is
# the same amount of evidence, plus a negative answer on the last question.
_LADDER = {REQUESTED: 0, RESTARTED: 1, PROGRESSED: 2, NOT_CORRECTED: 2, CORRECTED: 3}

# How an attempt stopped, which is a different axis from how far it got. A cancelled
# attempt and one that ran out of observation budget are both "not corrected", and
# treating them alike hides the difference between "we stopped asking" and "it did not
# answer" - the two need opposite responses from a retry policy.
RESOLVED = "resolved"
TIMED_OUT = "timed_out"
CANCELLED = "cancelled"
REFUSED = "refused"
ERROR = "error"

# A launch that dies immediately still appends nothing; one that trains appends on a
# fixed step interval. Two rows with a strictly advancing step is the cheapest
# evidence that optimizer steps happened between two emissions.
MIN_PROGRESS_ROWS = 2

DEFAULT_LEDGER = "logs/attempts.jsonl"


class Attempt:
    """One recovery attempt, from the trigger through to what was observed.

    Carries the identity and provenance a later reader needs to judge the claim:
    which anomaly triggered it, what action was approved, what configuration that
    produced, and what was independently seen afterwards.
    """

    def __init__(self, run_id, trigger, ledger_path=DEFAULT_LEDGER):
        self.attempt_id = str(uuid.uuid4())
        self.run_id = run_id
        self.trigger = trigger
        self.created_at = time.time()
        self.approved_action = None
        self.resulting_config = None
        self.baseline = None
        self.observations = []
        self.outcome = REQUESTED
        self.terminal = None
        self.ledger_path = Path(ledger_path)

    @classmethod
    def from_record(cls, record, ledger_path=DEFAULT_LEDGER):
        """Rehydrate one durable attempt without creating a second identity."""
        attempt = cls.__new__(cls)
        attempt.attempt_id = record["attempt_id"]
        attempt.run_id = record["run_id"]
        attempt.trigger = record["trigger"]
        attempt.created_at = record["created_at"]
        attempt.approved_action = record.get("approved_action")
        attempt.resulting_config = record.get("resulting_config")
        attempt.baseline = record.get("baseline")
        attempt.observations = list(record.get("observations", []))
        attempt.outcome = record.get("outcome", REQUESTED)
        attempt.terminal = record.get("terminal")
        attempt.ledger_path = Path(ledger_path)
        return attempt

    # ── recording ──────────────────────────────────────────────────────────────
    def record_action(self, action, resulting_config=None):
        """The action that was actually approved and applied, post-sandbox.

        What the model asked for and what the sandbox let through are not always the
        same thing, and the ledger has to hold the second one.
        """
        self.approved_action = action
        self.resulting_config = resulting_config
        self._append()

    def capture_baseline(self, metrics_file):
        """The metrics watermark to measure "new" against, taken BEFORE the action.

        Row count rather than step number on purpose: the trainer restarts global_step
        at zero on a rerun, so a fresh run's rows carry LOWER step numbers than the
        tail they follow. Anything keyed on "step went up" would read a real restart as
        no progress at all.
        """
        rows = _read_rows(metrics_file)
        self.baseline = {
            "row_count": len(rows),
            "last_step": rows[-1].get("step") if rows else None,
            "captured_at": time.time(),
        }
        self._append()
        return self.baseline

    def cancel(self, reason=""):
        self.terminal = CANCELLED
        self.observations.append({"source": "operator", "reason": reason, "at": time.time()})
        self._append()
        return self.outcome

    def fail(self, terminal, reason=""):
        self.terminal = terminal
        self.observations.append({"source": "agent", "reason": reason, "at": time.time()})
        self._append()
        return self.outcome

    # ── observation ────────────────────────────────────────────────────────────
    def observe(self, metrics_file, now=None):
        """Raise the outcome to whatever the metrics stream currently supports.

        Reads the file directly. Nothing here consults the rerun server, because the
        rerun server's opinion of its own subprocess is the thing that was wrong.
        """
        if self.baseline is None:
            raise ValueError("capture_baseline must run before the action, not after")

        rows = _read_rows(metrics_file)
        fresh = rows[self.baseline["row_count"]:]
        evidence = {
            "source": "metrics_file",
            "at": now if now is not None else time.time(),
            "fresh_rows": len(fresh),
        }

        if not fresh:
            # Accepted request, silent trainer. This is exactly the state the old code
            # reported as "fixed".
            evidence["verdict"] = "no rows written since the request"
            self._raise_to(REQUESTED, evidence)
            return self.outcome

        steps = [r.get("step") for r in fresh if isinstance(r.get("step"), (int, float))]
        advanced = len(steps) >= MIN_PROGRESS_ROWS and steps[-1] > steps[0]
        if not advanced:
            evidence["verdict"] = "a process wrote, but not enough rows to show it trained"
            self._raise_to(RESTARTED, evidence)
            return self.outcome

        still_firing = {a["type"] for a in detect_anomalies_in(fresh)}
        evidence["steps_observed"] = [steps[0], steps[-1]]
        evidence["anomalies_after"] = sorted(still_firing)

        if self.trigger.get("type") in still_firing:
            evidence["verdict"] = "trained, but the triggering anomaly still fires"
            self._raise_to(NOT_CORRECTED, evidence)
        else:
            evidence["verdict"] = "trained, and the triggering anomaly no longer fires"
            self.terminal = RESOLVED
            # Persist the terminal marker in the same append as the corrected
            # rung. Writing the rung first leaves a restarted worker seeing a
            # supposedly unfinished attempt even though this process resolved it.
            self._raise_to(CORRECTED, evidence)
        return self.outcome

    def observe_until(self, metrics_file, deadline, poll=lambda: None, clock=time.time):
        """Observe until CORRECTED or the budget runs out.

        Reaching the deadline is recorded as TIMED_OUT and never as a corrected
        outcome: not having seen a fix is not the same as having seen a failure, and
        neither is a fix.
        """
        while clock() < deadline:
            if self.observe(metrics_file) == CORRECTED:
                return self.outcome
            poll()
        self.terminal = TIMED_OUT
        self._append()
        return self.outcome

    # ── derived ────────────────────────────────────────────────────────────────
    @property
    def fixed(self):
        """The only claim allowed to be called a fix."""
        return self.outcome == CORRECTED and self.terminal == RESOLVED

    def status(self):
        """The decision-log status, derived from evidence rather than from intent.

        "patched" now means the configuration changed and nothing after that was
        observed to work - which is a real and useful thing to report, and is what the
        overwhelming majority of the old "fixed" rows actually were.
        """
        if self.fixed:
            return "fixed"
        if self.approved_action:
            return "patched"
        return "failed"

    def to_dict(self):
        return {
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "trigger": self.trigger,
            "approved_action": self.approved_action,
            "resulting_config": self.resulting_config,
            "baseline": self.baseline,
            "observations": self.observations,
            "outcome": self.outcome,
            "terminal": self.terminal,
            "fixed": self.fixed,
            "created_at": self.created_at,
        }

    # ── internals ──────────────────────────────────────────────────────────────
    def _raise_to(self, rung, evidence):
        self.observations.append(evidence)
        before = self.outcome
        if _LADDER[rung] > _LADDER[self.outcome]:
            self.outcome = rung
        elif rung == NOT_CORRECTED and self.outcome == PROGRESSED:
            # Same rung, but a definite negative answer beats the neutral one.
            self.outcome = NOT_CORRECTED
        if self.outcome != before:
            # The run and attempt ids ride on the log record. They are refused as
            # metric labels by obs.declare_counter, which is the point: one series per
            # attempt is one series per recovery, forever, and the rung is the part
            # anyone actually aggregates.
            with obs.bind(run_id=self.run_id, attempt_id=self.attempt_id):
                obs.incr("argus_recovery_attempts_total", outcome=self.outcome)
                obs.log("recovery.rung_earned", outcome=self.outcome,
                        previous=before, trigger=self.trigger.get("type"),
                        fresh_rows=evidence.get("fresh_rows"),
                        verdict=evidence.get("verdict"))
        self._append()

    def _append(self):
        """Append the current state to the ledger.

        Append-only and rewritten in full each time: the ledger is a history of what
        was believed and when, so a crash mid-attempt leaves the last earned rung on
        disk instead of losing the attempt entirely.
        """
        path = self.ledger_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")


# ── ledger reads ───────────────────────────────────────────────────────────────
def load_ledger(ledger_path=DEFAULT_LEDGER):
    """Latest state of every attempt in the ledger, in first-seen order."""
    path = Path(ledger_path)
    if not path.exists():
        return []
    latest = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a torn final line must not lose the attempts before it
            latest[row["attempt_id"]] = row
    return list(latest.values())


def resume_attempt(run_id, trigger, ledger_path=DEFAULT_LEDGER):
    """Return the newest non-terminal attempt for the same triggering event."""
    trigger_key = (trigger.get("type"), trigger.get("step"))
    for record in reversed(load_ledger(ledger_path)):
        recorded = record.get("trigger") or {}
        if record.get("run_id") != run_id:
            continue
        if (recorded.get("type"), recorded.get("step")) != trigger_key:
            continue
        if record.get("terminal") is None:
            return Attempt.from_record(record, ledger_path=ledger_path)
        return None
    return None


def reconcile_open_attempts(run_id, metrics_file, ledger_path=DEFAULT_LEDGER):
    """Re-observe every unfinished attempt after a worker restart.

    The metrics file is append-only. If its fresh-row count has not changed
    since the last observation, there is no new evidence to append.
    """
    row_count = len(_read_rows(metrics_file))
    reconciled = []
    for record in load_ledger(ledger_path):
        if record.get("run_id") != run_id or record.get("terminal") is not None:
            continue
        attempt = Attempt.from_record(record, ledger_path=ledger_path)
        if attempt.baseline is None:
            continue
        fresh_rows = max(0, row_count - attempt.baseline["row_count"])
        last_metrics = next(
            (item for item in reversed(attempt.observations)
             if item.get("source") == "metrics_file"),
            None,
        )
        if last_metrics is None or last_metrics.get("fresh_rows") != fresh_rows:
            attempt.observe(metrics_file)
        reconciled.append(attempt)
    return reconciled


def _read_rows(metrics_file):
    path = Path(metrics_file)
    if not path.exists():
        return []
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # the trainer may be mid-write on the last line
    return rows
