"""Export one deterministic pause/resume trace from Argus's LangGraph workflow.

The fake model and tool make the saved public example repeatable without a model
provider or live training job. The graph source hash binds the artifact to code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.langgraph_recovery import build_recovery_graph  # noqa: E402


class FixtureModel:
    """Read once, then finish after the graph resumes from its checkpoint."""

    def next_action(self, state):
        if state.get("tool_result"):
            return {"action": {"kind": "finish", "input": {}}}
        return {"action": {"kind": "read_metrics", "input": {"last_n": 2}}}


def export() -> dict:
    graph_source = ROOT / "agent" / "langgraph_recovery.py"
    graph = build_recovery_graph(
        FixtureModel(), lambda _name, _payload: {"status": "ok", "rows": 2}
    )
    config = {"configurable": {"thread_id": "public-recovery-replay"}}
    paused = graph.invoke(
        {"anomaly": {"type": "loss_spike"}, "budget": 1}, config=config
    )
    resumed = graph.invoke({"resume": True, "budget": 2}, config=config)
    if paused["status"] != "paused" or resumed["status"] != "completed":
        raise RuntimeError("fixture graph did not pause and complete after resume")

    return {
        "thread_id": config["configurable"]["thread_id"],
        "source": "agent/langgraph_recovery.py",
        "source_sha256": hashlib.sha256(graph_source.read_bytes()).hexdigest(),
        "first_status": paused["status"],
        "resumed_status": resumed["status"],
        "actions": [entry["action"]["kind"] for entry in resumed["history"]],
        "history": resumed["history"],
        "scope": "Real LangGraph execution with a deterministic fake model and fixture tool; no provider or live training action.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dashboard/public/recovery-replay.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(export(), indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
