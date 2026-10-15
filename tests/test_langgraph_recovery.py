"""Deterministic end-to-end tests for the checkpointed LangGraph workflow."""

from agent.langgraph_recovery import build_recovery_graph


class FakeModel:
    def __init__(self):
        self.calls = 0

    def next_action(self, state):
        self.calls += 1
        if not state.get("tool_result"):
            return {"thought": "inspect the fixture", "action": {"kind": "read_metrics", "input": {"last_n": 2}}}
        return {"thought": "the fixture is healthy", "action": {"kind": "finish", "input": {}}}


class RetryModel:
    def next_action(self, state):
        if state.get("tool_result", {}).get("status") == "error":
            return {"thought": "retry the transient read", "action": {"kind": "read_metrics", "input": {}}}
        if not state.get("tool_result"):
            return {"action": {"kind": "read_metrics", "input": {}}}
        return {"action": {"kind": "finish", "input": {}}}


def test_react_workflow_runs_fixture_to_completion():
    seen = []

    def tool(name, payload):
        seen.append((name, payload))
        return {"status": "ok", "rows": [1, 2]}

    graph = build_recovery_graph(FakeModel(), tool)
    state = graph.invoke({"anomaly": {"type": "loss_spike"}, "budget": 3},
                         config={"configurable": {"thread_id": "fixture-complete"}})
    assert state["status"] == "completed"
    assert [x[0] for x in seen] == ["read_metrics"]
    assert state["history"][-1]["result"]["status"] == "finished"


def test_checkpoint_pause_and_resume_preserves_history():
    model = FakeModel()
    graph = build_recovery_graph(model, lambda name, payload: {"status": "ok"})
    config = {"configurable": {"thread_id": "fixture-resume"}}
    paused = graph.invoke({"anomaly": {"type": "grad_explosion"}, "budget": 1}, config=config)
    assert paused["status"] == "paused"
    assert len(paused["history"]) == 1

    resumed = graph.invoke({"resume": True, "budget": 2}, config=config)
    assert resumed["status"] == "completed"
    assert len(resumed["history"]) == 2
    assert model.calls == 2
    terminal = graph.invoke({"resume": True, "budget": 2}, config=config)
    assert terminal["status"] == "completed"
    assert model.calls == 2


def test_failed_terminal_checkpoint_does_not_call_model_again():
    class BrokenModel:
        calls = 0

        def next_action(self, state):
            self.calls += 1
            raise RuntimeError("fixture model failure")

    model = BrokenModel()
    graph = build_recovery_graph(model, lambda name, payload: {"status": "ok"})
    config = {"configurable": {"thread_id": "failed-terminal-reuse"}}
    failed = graph.invoke({"anomaly": {"type": "loss_spike"}}, config=config)
    reused = graph.invoke({"resume": True}, config=config)
    assert failed["status"] == reused["status"] == "failed"
    assert model.calls == 1


def test_tool_error_is_retried_and_then_completes():
    calls = 0

    def flaky_tool(name, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"status": "error", "error": "temporary fixture failure"}
        return {"status": "ok", "rows": [1]}

    graph = build_recovery_graph(RetryModel(), flaky_tool)
    state = graph.invoke({"anomaly": {"type": "loss_spike"}, "budget": 4, "max_retries": 2},
                         config={"configurable": {"thread_id": "fixture-retry"}})
    assert state["status"] == "completed"
    assert calls == 2
    assert state["retries"] == 1


def test_action_gate_refuses_model_action_before_tool_effect():
    effects = []
    graph = build_recovery_graph(
        FakeModel(),
        lambda name, payload: effects.append((name, payload)) or {"status": "ok"},
        action_gate=lambda state, action: False,
    )
    state = graph.invoke(
        {"anomaly": {"type": "loss_spike"}, "budget": 2},
        config={"configurable": {"thread_id": "gate-refusal"}},
    )
    assert state["status"] == "failed"
    assert "action gate" in state["error"]
    assert effects == []
