"""A checkpointed LangGraph recovery workflow for deterministic anomaly repair.

This module is deliberately separate from ``agent.loop``.  The production loop
still owns Anthropic calls, while this graph gives the recovery protocol an
explicit state machine that can be replayed with a fake model in tests.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Protocol, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class RecoveryState(TypedDict, total=False):
    """State persisted after every graph node."""

    anomaly: dict[str, Any]
    action: dict[str, Any]
    thought: str
    tool_result: dict[str, Any]
    history: list[dict[str, Any]]
    round: int
    budget: int
    retries: int
    max_retries: int
    status: Literal["running", "paused", "completed", "failed"]
    resume: bool
    error: str


class RecoveryModel(Protocol):
    """Small model boundary, easy to replace with a real tool-calling model."""

    def next_action(self, state: RecoveryState) -> dict[str, Any]: ...


ToolRunner = Callable[[str, dict[str, Any]], dict[str, Any]]


def build_recovery_graph(
    model: RecoveryModel,
    tool_runner: ToolRunner,
    *,
    checkpointer: Any | None = None,
):
    """Build the LangGraph ReAct loop.

    The model chooses one tool action, the executor runs it, and the router sends
    the result back to the model.  ``budget`` is a deliberate pause boundary:
    callers can invoke the same thread again with ``resume=True`` and a new budget,
    preserving the prior history and tool results from the MemorySaver checkpoint.
    """

    def prepare(state: RecoveryState) -> dict[str, Any]:
        if state.get("status") == "paused" and state.get("resume"):
            return {"status": "running", "resume": False, "error": ""}
        if state.get("status") in {"completed", "failed"}:
            return {}
        return {
            "status": "running",
            "round": state.get("round", 0),
            "budget": max(1, int(state.get("budget", 5))),
            "retries": int(state.get("retries", 0)),
            "max_retries": max(0, int(state.get("max_retries", 2))),
            "history": list(state.get("history", [])),
        }

    def think(state: RecoveryState) -> dict[str, Any]:
        try:
            decision = model.next_action(state)
            action = dict(decision.get("action", decision))
            return {"action": action, "thought": str(decision.get("thought", ""))}
        except Exception as exc:
            return {"status": "failed", "error": f"model error: {exc}"}

    def execute(state: RecoveryState) -> dict[str, Any]:
        action = state.get("action", {})
        kind = action.get("kind", "")
        history = list(state.get("history", []))
        if kind == "finish":
            history.append({"action": action, "result": {"status": "finished"}})
            return {"history": history, "status": "completed"}
        if not kind:
            return {"status": "failed", "error": "model returned no action"}
        try:
            result = tool_runner(kind, dict(action.get("input", {})))
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}
        history.append({"action": action, "result": result})
        next_retries = int(state.get("retries", 0)) + (1 if result.get("status") == "error" else 0)
        exhausted = result.get("status") == "error" and next_retries > int(state.get("max_retries", 2))
        return {
            "tool_result": result,
            "history": history,
            "round": int(state.get("round", 0)) + 1,
            "retries": next_retries,
            "status": "failed" if exhausted else "running",
            "error": "retry budget exhausted" if exhausted else "",
        }

    def route_after_think(state: RecoveryState) -> str:
        if state.get("status") == "failed":
            return END
        if state.get("action", {}).get("kind") == "finish":
            return "execute"
        if int(state.get("round", 0)) >= int(state.get("budget", 5)):
            return "pause"
        return "execute"

    def pause(state: RecoveryState) -> dict[str, Any]:
        return {"status": "paused"}

    def route_after_execute(state: RecoveryState) -> str:
        if state.get("status") in {"completed", "failed"}:
            return END
        if state.get("tool_result", {}).get("status") == "error" and int(state.get("retries", 0)) > int(state.get("max_retries", 2)):
            return END
        if int(state.get("round", 0)) >= int(state.get("budget", 5)):
            return "pause"
        return "think"

    graph = StateGraph(RecoveryState)
    graph.add_node("prepare", prepare)
    graph.add_node("think", think)
    graph.add_node("execute", execute)
    graph.add_node("pause", pause)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "think")
    graph.add_conditional_edges("think", route_after_think, {"execute": "execute", "pause": "pause", END: END})
    graph.add_conditional_edges("execute", route_after_execute, {"think": "think", "pause": "pause", END: END})
    graph.add_edge("pause", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())
