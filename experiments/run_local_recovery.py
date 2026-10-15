"""Run the pre-frozen local Ollama recovery action-following study."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from agent.langgraph_recovery import build_recovery_graph
from agent.local_ollama import OllamaRecoveryModel


class RecordingModel:
    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def next_action(self, state):
        started = time.monotonic()
        try:
            decision = self.inner.next_action(state)
            result = {"decision": decision, "metadata": self.inner.last_response_metadata,
                      "raw_response": self.inner.last_raw_response,
                      "elapsed_s": round(time.monotonic() - started, 3), "error": None}
            self.calls.append(result)
            return decision
        except Exception as exc:
            self.calls.append({"decision": None, "metadata": self.inner.last_response_metadata,
                               "raw_response": self.inner.last_raw_response,
                               "elapsed_s": round(time.monotonic() - started, 3),
                               "error": f"{type(exc).__name__}: {exc}"})
            raise


def run_case(case, model):
    expected = case["expected_actions"]
    cursor = 0
    gate_events = []
    effects = []

    def gate(state, action):
        nonlocal cursor
        accepted = cursor < len(expected) and action == expected[cursor]
        gate_events.append({"index": cursor, "action": action, "accepted": accepted})
        if accepted:
            cursor += 1
        return accepted

    def fixture_tool(name, payload):
        # This runs only after the graph's deterministic action gate accepts the action.
        effects.append({"tool": name, "input": payload})
        return case["tool_results"].get(name, {"status": "error", "error": "unlisted fixture tool"})

    graph = build_recovery_graph(model, fixture_tool, action_gate=gate)
    config = {"configurable": {"thread_id": f"local-{case['id']}"}}
    initial = dict(case["initial_state"])
    budget = int(initial.get("budget", 1))
    started = time.monotonic()
    first = graph.invoke(initial, config=config)
    first_status = first.get("status")
    resumed = None
    if first_status == "paused":
        resumed = graph.invoke({"resume": True, "budget": 3}, config=config)
    terminal = resumed or first
    before_terminal_reuse = len(effects)
    calls_before_terminal_reuse = len(model.calls)
    reused = graph.invoke({"resume": True, "budget": 3}, config=config)
    return {
        "case_id": case["id"],
        "initial_budget": budget,
        "first_status": first_status,
        "resumed_status": resumed.get("status") if resumed else None,
        "terminal_status": terminal.get("status"),
        "terminal_reuse_status": reused.get("status"),
        "terminal_reuse_added_effects": len(effects) - before_terminal_reuse,
        "terminal_reuse_added_model_calls": len(model.calls) - calls_before_terminal_reuse,
        "elapsed_s": round(time.monotonic() - started, 3),
        "expected_actions": expected,
        "gate_events": gate_events,
        "gate_completed": cursor == len(expected),
        "fixture_effects": effects,
        "history": terminal.get("history", []),
        "error": terminal.get("error"),
        "model_calls": model.calls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="llama3.2:3b")
    args = parser.parse_args()

    raw_protocol = args.protocol.read_bytes()
    protocol = json.loads(raw_protocol)
    cases = protocol["scenarios"]
    if len(cases) > 2:
        raise SystemExit("bounded study accepts at most two frozen scenarios")
    results = []
    for case in cases:
        local_model = OllamaRecoveryModel(model=args.model)
        recording = RecordingModel(local_model)
        results.append(run_case(case, recording))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(raw_protocol).hexdigest(),
        "model_tag": args.model,
        "model_manifest_digest": "sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff",
        "ollama_cloud_disabled_by_runner": True,
        "local_endpoint": "http://127.0.0.1:11434",
        "cases": results,
        "summary": {
            "cases": len(results),
            "exact_action_sequences": sum(r["gate_completed"] for r in results),
            "terminal_reuse_without_effects": all(r["terminal_reuse_added_effects"] == 0 for r in results),
            "all_effects_gate_approved": all(
                len(r["fixture_effects"]) == sum(e["accepted"] and e["action"]["kind"] != "finish" for e in r["gate_events"])
                for r in results
            ),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
