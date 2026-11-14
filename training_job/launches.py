"""Which rerun requests actually start a trainer, and which are already running.

Separate from server.py so the policy is testable without a web framework installed,
and because it is policy rather than transport.

The rule it enforces is that a rerun is identified by the recovery attempt that asked
for it, not by the arrival of an HTTP request. Retries are not hypothetical here: the
agent's tool call has a 10s timeout, the handler answers before the trainer has done
anything, and a request that times out in transit looks identical to one that was
never received. Retrying it without a key starts a SECOND trainer against the same
config, and both append to one metrics file - which corrupts the exact signal the
recovery is judged by, since interleaved rows from two runs read as a step sequence
that jumps around.

So a launch carries its attempt's id and a repeat request for a live attempt returns
the launch already in flight instead of starting another. This is the reconciliation
policy that a durable-workflow engine would have forced us to write anyway: activities
in any at-least-once executor can run more than once, and the fix is the same key
either way. Writing it here rather than adopting an engine to get it is the point of
the comparison in RECORD_argus_lifecycle.md.
"""

from __future__ import annotations

import threading
import time
import uuid

# States a launch can be in. "requested" means only that we have accepted the job;
# whether a trainer ran is a question for the metrics stream, not for this registry.
REQUESTED = "requested"
RUNNING = "running"
EXITED = "exited"
LAUNCH_FAILED = "launch_failed"
TIMEOUT = "timeout"

LIVE_STATES = (REQUESTED, RUNNING)


class LaunchRegistry:
    """In-memory record of launches, keyed by attempt.

    In-memory is a deliberate limit and not a hidden one: the trainer is a child of
    this process, so a restart of this process loses the children too, and there is
    nothing for a persisted registry to reconcile against. The durable record of what
    was attempted lives in the agent's ledger (agent/attempt.py), which survives
    independently and re-derives the outcome from the metrics file.
    """

    def __init__(self, clock=time.time):
        self._by_id = {}
        self._by_attempt = {}
        self._lock = threading.Lock()
        self._clock = clock

    def request(self, attempt_id, max_steps):
        """Claim a launch for `attempt_id`.

        Returns (launch, created). `created` is False when a live launch for this
        attempt already exists, which is the duplicate-suppression case.
        """
        with self._lock:
            existing = self._by_attempt.get(attempt_id)
            if existing is not None and existing["state"] in LIVE_STATES:
                return dict(existing), False

            launch = {
                "launch_id": str(uuid.uuid4()),
                "attempt_id": attempt_id,
                "max_steps": max_steps,
                "state": REQUESTED,
                "requested_at": self._clock(),
            }
            self._by_id[launch["launch_id"]] = launch
            if attempt_id is not None:
                self._by_attempt[attempt_id] = launch
            return dict(launch), True

    def update(self, launch_id, **fields):
        with self._lock:
            launch = self._by_id.get(launch_id)
            if launch is None:
                return None
            launch.update(fields)
            return dict(launch)

    def get(self, launch_id):
        with self._lock:
            launch = self._by_id.get(launch_id)
            return dict(launch) if launch else {"state": "unknown", "launch_id": launch_id}

    def for_attempt(self, attempt_id):
        with self._lock:
            launch = self._by_attempt.get(attempt_id)
            return dict(launch) if launch else None
