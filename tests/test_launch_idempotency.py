"""A retried rerun request must not start a second trainer.

This is the requirement that came out of comparing the persisted attempt ledger
against a Temporal implementation of the same flow. Temporal was not adopted, but the
question it forces - what happens when this activity runs twice? - had two real
answers here, and this is the dangerous one.

The agent's rerun tool call has a 10 second timeout while a training run takes
minutes, and the handler answers before the trainer has done anything. A request whose
response is lost is indistinguishable from one that never arrived, so it gets retried.
Without a key, the retry starts a second trainer against the same config, and both
append to one metrics file. Interleaved rows from two runs read as a step sequence
that jumps backwards, which corrupts the exact signal agent/attempt.py uses to decide
whether the recovery worked.

The key is the recovery attempt's id, so "the same attempt asking again" and "a new
attempt" are distinguishable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training_job"))

from agent import tools
from agent.sandbox import SandboxError
from launches import EXITED, LaunchRegistry


# ── the registry policy ────────────────────────────────────────────────────────
def test_a_retry_for_a_live_attempt_returns_the_launch_already_running():
    reg = LaunchRegistry()
    first, created_first = reg.request("attempt-1", max_steps=100)
    second, created_second = reg.request("attempt-1", max_steps=100)

    assert created_first is True
    assert created_second is False
    assert second["launch_id"] == first["launch_id"]


def test_a_different_attempt_gets_its_own_launch():
    reg = LaunchRegistry()
    first, _ = reg.request("attempt-1", max_steps=100)
    second, created = reg.request("attempt-2", max_steps=100)

    assert created is True
    assert second["launch_id"] != first["launch_id"]


def test_an_attempt_may_launch_again_once_its_previous_run_has_finished():
    """Suppression covers in-flight duplicates, not the whole history.

    A second attempt at the same fix after the first run has exited is a legitimate
    retry of the recovery, not a duplicate of one request.
    """
    reg = LaunchRegistry()
    first, _ = reg.request("attempt-1", max_steps=100)
    reg.update(first["launch_id"], state=EXITED, returncode=0)

    second, created = reg.request("attempt-1", max_steps=100)
    assert created is True
    assert second["launch_id"] != first["launch_id"]


def test_a_launch_records_the_exit_status_it_used_to_discard():
    """The old handler spawned a thread and dropped the return code on the floor.

    A trainer that died on its first line was then indistinguishable from one that
    trained for an hour.
    """
    reg = LaunchRegistry()
    launch, _ = reg.request("attempt-1", max_steps=100)
    reg.update(launch["launch_id"], state=EXITED, returncode=1, stderr_tail="ValueError")

    stored = reg.get(launch["launch_id"])
    assert stored["state"] == EXITED
    assert stored["returncode"] == 1
    assert "ValueError" in stored["stderr_tail"]


def test_an_unknown_launch_id_is_reported_as_unknown_not_invented():
    assert LaunchRegistry().get("nope")["state"] == "unknown"


# ── the key may not come from the model ────────────────────────────────────────
def test_the_model_cannot_supply_its_own_idempotency_key(monkeypatch, tmp_path):
    """Model output choosing the key would launder a retry into a fresh launch.

    The agent reads training logs it does not control, so a model that can be talked
    into varying this argument could be talked into starting unlimited trainers.
    """
    monkeypatch.setenv("ARGUS_WORKSPACE", str(tmp_path))

    result = tools.run_tool(
        "rerun_training",
        {"training_dir": str(tmp_path), "max_steps": 50, "attempt_id": "chosen-by-model"},
    )
    assert result.get("refused") is True
    assert "attempt_id" in result["error"]


def test_the_caller_injected_key_reaches_the_training_server(monkeypatch, tmp_path):
    monkeypatch.setenv("ARGUS_WORKSPACE", str(tmp_path))
    sent = {}

    class FakeResponse:
        def json(self):
            return {"status": "requested", "launch_id": "L1", "duplicate": False}

    class FakeRequests:
        @staticmethod
        def post(url, json=None, timeout=None):
            sent.update(json)
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", FakeRequests)

    result = tools.run_tool(
        "rerun_training", {"training_dir": str(tmp_path), "max_steps": 50},
        attempt_id="attempt-abc",
    )
    assert sent["attempt_id"] == "attempt-abc"
    # And never "started": the server has not observed a trainer at this point.
    assert result["status"] == "requested"
