import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "offline_walkthrough.py"
SAMPLE = ROOT / "examples" / "sample_metrics.jsonl"


def run(path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_bundled_walkthrough_executes_the_real_detector():
    result = run(SAMPLE)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["rows"] == 21
    assert report["detected"] == ["grad_explosion", "loss_spike"]


def test_walkthrough_rejects_a_sample_that_no_longer_contains_the_fault(tmp_path):
    rows = [json.loads(line) for line in SAMPLE.read_text().splitlines()]
    rows[-1]["train_loss"] = 0.95
    rows[-1]["grad_norm"] = 1.0
    changed = tmp_path / "changed.jsonl"
    changed.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = run(changed)
    assert result.returncode == 1
    assert "offline walkthrough result changed" in result.stderr
