"""Detector reads a bounded tail, not the whole run.

Reading the entire metrics file to inspect the last twenty rows makes every
detection cycle cost O(run length). The results must be identical either way -
a cheaper read that changes a verdict is not an optimisation, it is a bug.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import detector


def write_run(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def row(step, train=1.0, val=1.0, grad=0.5):
    return {"step": step, "train_loss": train, "val_loss": val, "grad_norm": grad}


def test_bounded_read_gives_the_same_verdicts_as_the_full_read(tmp_path):
    import random

    rng = random.Random(5)
    m = tmp_path / "metrics.jsonl"
    for scenario, rows in {
        "quiet": [row(i, train=rng.gauss(1.0, 0.05)) for i in range(400)],
        "loss spike": [row(i, train=rng.gauss(1.0, 0.05)) for i in range(400)]
        + [row(400, train=8.0)],
        "grad explosion": [row(i, grad=rng.gauss(1.0, 0.05)) for i in range(400)]
        + [row(400, grad=9.0)],
        "overfitting": [row(i, train=1.0, val=1.0 + i * 0.01) for i in range(400)],
    }.items():
        write_run(m, rows)
        full = detector.detect_anomalies(str(m), )
        # Same file, forced whole-file read, same detectors.
        whole = []
        metrics = detector.load_metrics(str(m), history=None)
        for check in (detector.detect_loss_spike, detector.detect_grad_explosion,
                      detector.detect_val_plateau, detector.detect_overfitting):
            r = check(metrics)
            if r:
                whole.append(r)
        assert [a["type"] for a in full] == [a["type"] for a in whole], (
            f"{scenario}: bounded read changed the verdict"
        )


def test_work_does_not_grow_with_run_length(tmp_path):
    """Bytes read per detection cycle must not scale with the run.

    The counter wraps read AND readlines: an earlier version of this test wrapped
    only read(), so the unbounded whole-file path (which uses readlines) counted
    zero bytes and the test passed against the very regression it exists to catch.
    """
    import builtins

    m = tmp_path / "metrics.jsonl"
    sizes = {}
    returned = {}
    for n in (100, 2000, 20000):
        write_run(m, [row(i) for i in range(n)])
        read = {"bytes": 0}
        real_open = builtins.open

        def counting_open(file, mode="r", *a, **kw):
            fh = real_open(file, mode, *a, **kw)
            if str(file) != str(m):
                return fh
            real_read, real_readlines = fh.read, fh.readlines

            def counted_read(*args):
                data = real_read(*args)
                read["bytes"] += len(data)
                return data

            def counted_readlines(*args):
                lines = real_readlines(*args)
                read["bytes"] += sum(len(l) for l in lines)
                return lines

            fh.read = counted_read
            fh.readlines = counted_readlines
            return fh

        builtins.open = counting_open
        try:
            returned[n] = len(detector.load_metrics(str(m)))
        finally:
            builtins.open = real_open
        sizes[n] = read["bytes"]

    # The property is that the read is CONSTANT once the file exceeds one block,
    # not that it is below some ratio: a 10x longer run must read the same bytes.
    assert sizes[20000] == sizes[2000], (
        f"read grows with run length: {sizes} bytes for 100/2000/20000 steps"
    )
    assert m.stat().st_size > 500_000  # the file really did get big
    assert sizes[20000] < m.stat().st_size / 10
    # And the caller is handed a bounded window, not the whole run.
    assert returned[20000] <= detector.REQUIRED_HISTORY, (
        f"load_metrics returned {returned[20000]} records for a 20,000-step run"
    )


def test_a_torn_final_line_does_not_break_detection(tmp_path):
    m = tmp_path / "metrics.jsonl"
    write_run(m, [row(i) for i in range(50)])
    with open(m, "a") as f:
        f.write('{"step": 50, "train_los')  # trainer caught mid-write
    metrics = detector.load_metrics(str(m))
    assert len(metrics) >= 20
    assert metrics[-1]["step"] == 49, "a torn line was parsed as a record"


def test_short_runs_still_read_everything(tmp_path):
    m = tmp_path / "metrics.jsonl"
    write_run(m, [row(i) for i in range(5)])
    metrics = detector.load_metrics(str(m))
    assert [r["step"] for r in metrics] == [0, 1, 2, 3, 4]
