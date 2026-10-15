from agent.langgraph_recovery import build_recovery_graph


class FinishModel:
    def next_action(self, state):
        return {"action": {"kind": "finish", "input": {}}}


def test_terminal_checkpoint_does_not_repeat_tool_effects():
    calls = []
    graph = build_recovery_graph(FinishModel(), lambda name, payload: calls.append(name) or {"status": "ok"})
    config = {"configurable": {"thread_id": "holdout-terminal-reuse"}}
    first = graph.invoke({"anomaly": {"type": "loss_spike"}, "budget": 2}, config=config)
    second = graph.invoke({"resume": True, "budget": 2}, config=config)
    assert first["status"] == second["status"] == "completed"
    assert calls == []


def test_tool_call_budget_pauses_without_losing_history():
    calls = []
    class OneReadThenFinish:
        def next_action(self, state):
            return {"action": {"kind": "finish" if state.get("tool_result") else "read_metrics", "input": {}}}
    graph = build_recovery_graph(OneReadThenFinish(), lambda name, payload: calls.append(name) or {"status": "ok"})
    config = {"configurable": {"thread_id": "holdout-budget"}}
    state = graph.invoke({"anomaly": {"type": "loss_spike"}, "budget": 1}, config=config)
    assert state["status"] == "paused"
    assert len(state["history"]) == 1 and calls == ["read_metrics"]


def test_recovery_context_reaches_model_and_survives_checkpoint_resume(tmp_path):
    resources = {"config_path": str(tmp_path / "config.yaml"),
                 "baseline_config": {"training.learning_rate": 0.001}}
    policy = "Read the exact config path; finish without a write."
    contract = {"read_config": {"required": {"config_path": "string"}}}
    seen = []
    effects = []

    class ContextModel:
        def next_action(self, state):
            assert state["resources"] == resources
            assert state["recovery_policy"] == policy
            assert state["action_contract"] == contract
            seen.append(state["resources"])
            if state.get("tool_result"):
                return {"action": {"kind": "finish", "input": {}}}
            return {"action": {"kind": "read_config",
                               "input": {"config_path": resources["config_path"]}}}

    def tool(name, payload):
        effects.append((name, payload))
        return {"status": "ok", "config": {"training": {"learning_rate": 0.001}}}

    graph = build_recovery_graph(ContextModel(), tool)
    thread = {"configurable": {"thread_id": "context-resume"}}
    first = graph.invoke({"anomaly": {}, "resources": resources,
                          "recovery_policy": policy, "action_contract": contract,
                          "budget": 1}, config=thread)
    assert first["status"] == "paused"
    second = graph.invoke({"resume": True, "budget": 2}, config=thread)
    assert second["status"] == "completed"
    assert second["resources"] == resources
    assert len(seen) == 2
    assert effects == [("read_config", {"config_path": resources["config_path"]})]
