"""Rolling z-score behaviour.

The first test documents a real defect in the original implementation: the sample
being tested was included in the window it was measured against, which bounds the
achievable z-score at (n-1)/sqrt(n) no matter how extreme the anomaly. With
ROLLING_WINDOW = 20 that ceiling is 4.249, and GRAD_EXPLOSION_ZSCORE is 4.0 - so the
gradient detector's statistical branch could only ever fire in a 6% sliver of its
range, and not at all before 19 samples had accumulated.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import detector


def series(n_normal, spike, seed=3):
    """A realistically noisy baseline plus one spike.

    A perfectly flat baseline has zero spread and legitimately yields no z-score at
    all, so it cannot demonstrate anything about the ceiling.
    """
    import random

    rng = random.Random(seed)
    return [rng.gauss(1.0, 0.05) for _ in range(n_normal)] + [spike]


def test_an_extreme_spike_is_not_capped_by_its_own_window():
    # A spike twelve orders of magnitude above the baseline must not score the same
    # as a merely large one. Under the old self-including formula both saturate at
    # (n-1)/sqrt(n).
    big = detector.rolling_zscore(series(19, 1e6), window=20)
    huge = detector.rolling_zscore(series(19, 1e12), window=20)
    assert big is not None and huge is not None
    ceiling = (20 - 1) / math.sqrt(20)  # 4.249, the old hard bound
    assert big > ceiling, (
        f"z={big} is still capped at the self-inclusion ceiling {ceiling}"
    )
    assert huge > big, "a larger anomaly must score higher, not saturate"


def test_grad_explosion_threshold_is_reachable_early_in_a_run():
    # Before the fix, fewer than 19 samples made GRAD_EXPLOSION_ZSCORE = 4.0
    # mathematically unreachable, so an explosion in the first steps of a run could
    # only be caught by the hard threshold.
    import random

    rng = random.Random(11)
    metrics = [
        {"step": i, "train_loss": 1.0, "val_loss": 1.0, "grad_norm": rng.gauss(1.0, 0.05)}
        for i in range(6)
    ]
    # Well under GRAD_EXPLOSION_THRESHOLD (10), so only the z-score branch can fire.
    metrics.append({"step": 6, "train_loss": 1.0, "val_loss": 1.0, "grad_norm": 9.0})
    got = detector.detect_grad_explosion(metrics)
    assert got is not None, "a 9x gradient jump on a flat baseline went undetected"
    assert "z-score" in got["description"]


def test_reference_window_excludes_the_tested_sample():
    # Baseline of 20 ones, then a 2.0. Mean/std of the REFERENCE (the ones) are
    # 1.0 and 0.0 -> undefined, so this must return None rather than a made-up
    # number. Zero-variance history is "cannot say", not "not anomalous".
    assert detector.rolling_zscore([1.0] * 20 + [2.0], window=20) is None


def test_warmup_returns_none_rather_than_a_fabricated_score():
    assert detector.rolling_zscore([], window=20) is None
    assert detector.rolling_zscore([1.0], window=20) is None
    # One reference point has no spread to measure against.
    assert detector.rolling_zscore([1.0, 5.0], window=20) is None


def test_a_normal_fluctuation_does_not_trigger():
    import random

    random.seed(7)
    vals = [random.gauss(1.0, 0.1) for _ in range(40)]
    z = detector.rolling_zscore(vals, window=20)
    assert z is not None
    assert abs(z) < 3.0, f"noise scored {z}, would be a false alarm"
