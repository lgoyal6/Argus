#!/usr/bin/env python3
"""Run Argus's real detector on bundled training metrics without credentials."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.detector import detect_anomalies  # noqa: E402

EXPECTED = {"loss_spike", "grad_explosion"}


def main() -> int:
    metrics = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name(
        "sample_metrics.jsonl"
    )
    anomalies = detect_anomalies(metrics)
    observed = {item["type"] for item in anomalies}
    report = {
        "metrics": str(metrics),
        "rows": sum(1 for line in metrics.read_text().splitlines() if line.strip()),
        "detected": sorted(observed),
        "expected": sorted(EXPECTED),
    }
    print(json.dumps(report, indent=2))
    if observed != EXPECTED:
        print("offline walkthrough result changed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
