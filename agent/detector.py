import json
from pathlib import Path
from statistics import mean, stdev


# ── thresholds ─────────────────────────────────────────────────────────────────
GRAD_EXPLOSION_THRESHOLD = 10  # hard threshold on grad norm
OVERFIT_RATIO = 2.0            # val/train loss ratio threshold

LOSS_SPIKE_ZSCORE = 3.0        # z-score threshold for loss spike
GRAD_EXPLOSION_ZSCORE = 4.0    # z-score threshold for grad norm spike
ROLLING_WINDOW = 20            # window size for z-score calculations

VAL_PLATEAU_STEPS = 10         # CUSUM window for plateau detection
CUSUM_MIN_DROP = 0.01          # CUSUM must drop by at least this to avoid plateau flag

OVERFIT_TREND_STEPS = 5        # gap must be increasing over this many steps


# ── read metrics file ──────────────────────────────────────────────────────────
def load_metrics(metrics_file):
    path = Path(metrics_file)
    if not path.exists():
        return []
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return [json.loads(l) for l in lines]


# ── statistical helpers ────────────────────────────────────────────────────────
def rolling_zscore(values, window=20):
    """Return the z-score of the last value against the rolling window mean/std.

    Requires at least 2 values in the window; returns None if there is
    insufficient data or zero standard deviation.
    """
    window_vals = values[-window:] if len(values) >= window else values[:]
    if len(window_vals) < 2:
        return None
    mu = mean(window_vals)
    sigma = stdev(window_vals)
    if sigma == 0:
        return None
    return (window_vals[-1] - mu) / sigma


# ── individual detectors ───────────────────────────────────────────────────────
def detect_loss_spike(metrics):
    if len(metrics) < 2:
        return None
    train_losses = [m["train_loss"] for m in metrics]
    z = rolling_zscore(train_losses, window=ROLLING_WINDOW)
    if z is None or z <= LOSS_SPIKE_ZSCORE:
        return None
    curr = metrics[-1]["train_loss"]
    prev = metrics[-2]["train_loss"]
    return {
        "type": "loss_spike",
        "step": metrics[-1]["step"],
        "prev_loss": prev,
        "curr_loss": curr,
        "ratio": round(curr / prev, 3) if prev > 0 else None,
        "zscore": round(z, 3),
        "description": f"train loss jumped from {prev} to {curr} (z-score {round(z, 2)})"
    }


def detect_grad_explosion(metrics):
    if len(metrics) < 1:
        return None
    curr = metrics[-1]
    grad = curr["grad_norm"]

    hard_trigger = grad > GRAD_EXPLOSION_THRESHOLD

    zscore_trigger = False
    if len(metrics) >= 2:
        grad_norms = [m["grad_norm"] for m in metrics]
        z = rolling_zscore(grad_norms, window=ROLLING_WINDOW)
        if z is not None and z > GRAD_EXPLOSION_ZSCORE:
            zscore_trigger = True

    if not (hard_trigger or zscore_trigger):
        return None

    reason = []
    if hard_trigger:
        reason.append(f"exceeds threshold {GRAD_EXPLOSION_THRESHOLD}")
    if zscore_trigger:
        grad_norms = [m["grad_norm"] for m in metrics]
        z = rolling_zscore(grad_norms, window=ROLLING_WINDOW)
        reason.append(f"z-score {round(z, 2)}")

    return {
        "type": "grad_explosion",
        "step": curr["step"],
        "grad_norm": grad,
        "description": f"grad norm {grad} — {', '.join(reason)}"
    }


def detect_val_plateau(metrics):
    if len(metrics) < VAL_PLATEAU_STEPS:
        return None
    recent = metrics[-VAL_PLATEAU_STEPS:]
    val_losses = [m["val_loss"] for m in recent]

    # CUSUM: cumulative sum of (val_loss - target), where target = first value in window
    target = val_losses[0]
    cusum = []
    s = 0.0
    for v in val_losses:
        s += v - target
        cusum.append(s)

    cusum_max = max(cusum)
    cusum_final = cusum[-1]
    # plateau if CUSUM hasn't dropped by at least CUSUM_MIN_DROP from its peak
    if cusum_max - cusum_final < CUSUM_MIN_DROP:
        return {
            "type": "val_plateau",
            "step": metrics[-1]["step"],
            "val_losses": val_losses,
            "cusum_drop": round(cusum_max - cusum_final, 5),
            "description": (
                f"val loss plateau detected over {VAL_PLATEAU_STEPS} steps "
                f"(CUSUM drop {round(cusum_max - cusum_final, 4)} < {CUSUM_MIN_DROP})"
            )
        }
    return None


def detect_overfitting(metrics):
    if len(metrics) < 1:
        return None
    curr = metrics[-1]
    if curr["train_loss"] <= 0:
        return None

    ratio = curr["val_loss"] / curr["train_loss"]
    if ratio < OVERFIT_RATIO:
        return None

    # also require the val/train gap to be increasing over the last N steps
    if len(metrics) >= OVERFIT_TREND_STEPS:
        gaps = [
            m["val_loss"] - m["train_loss"]
            for m in metrics[-OVERFIT_TREND_STEPS:]
            if m["train_loss"] > 0
        ]
        gap_increasing = len(gaps) >= 2 and all(
            gaps[i] < gaps[i + 1] for i in range(len(gaps) - 1)
        )
        if not gap_increasing:
            return None

    return {
        "type": "overfitting",
        "step": curr["step"],
        "train_loss": curr["train_loss"],
        "val_loss": curr["val_loss"],
        "ratio": round(ratio, 3),
        "description": (
            f"val loss ({curr['val_loss']}) is {round(ratio, 2)}x train loss "
            f"({curr['train_loss']}) with increasing gap"
        )
    }


# ── main detector ──────────────────────────────────────────────────────────────
def detect_anomalies(metrics_file):
    metrics = load_metrics(metrics_file)
    if not metrics:
        return []

    anomalies = []

    checks = [
        detect_loss_spike,
        detect_grad_explosion,
        detect_val_plateau,
        detect_overfitting,
    ]

    for check in checks:
        result = check(metrics)
        if result:
            anomalies.append(result)

    return anomalies


# ── cli ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    metrics_file = sys.argv[1] if len(sys.argv) > 1 else "metrics/metrics.jsonl"
    anomalies = detect_anomalies(metrics_file)
    if anomalies:
        print(f"detected {len(anomalies)} anomaly/anomalies:")
        for a in anomalies:
            print(f"  [{a['type']}] {a['description']}")
    else:
        print("no anomalies detected")
