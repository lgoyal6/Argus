import json
from pathlib import Path


# ── thresholds ─────────────────────────────────────────────────────────────────
LOSS_SPIKE_FACTOR = 1.5        # train loss increased by 50% in one step
GRAD_EXPLOSION_THRESHOLD = 10  # grad norm above this is an explosion
VAL_PLATEAU_STEPS = 10         # val loss hasn't improved in this many steps
OVERFIT_RATIO = 2.0            # val loss is this many times higher than train loss


# ── read metrics file ──────────────────────────────────────────────────────────
def load_metrics(metrics_file):
    path = Path(metrics_file)
    if not path.exists():
        return []
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return [json.loads(l) for l in lines]


# ── individual detectors ───────────────────────────────────────────────────────
def detect_loss_spike(metrics):
    if len(metrics) < 2:
        return None
    prev = metrics[-2]["train_loss"]
    curr = metrics[-1]["train_loss"]
    if prev > 0 and curr / prev >= LOSS_SPIKE_FACTOR:
        return {
            "type": "loss_spike",
            "step": metrics[-1]["step"],
            "prev_loss": prev,
            "curr_loss": curr,
            "ratio": round(curr / prev, 3),
            "description": f"train loss jumped from {prev} to {curr} (x{round(curr/prev,2)})"
        }
    return None


def detect_grad_explosion(metrics):
    if len(metrics) < 1:
        return None
    curr = metrics[-1]
    if curr["grad_norm"] > GRAD_EXPLOSION_THRESHOLD:
        return {
            "type": "grad_explosion",
            "step": curr["step"],
            "grad_norm": curr["grad_norm"],
            "description": f"grad norm {curr['grad_norm']} exceeds threshold {GRAD_EXPLOSION_THRESHOLD}"
        }
    return None


def detect_val_plateau(metrics):
    if len(metrics) < VAL_PLATEAU_STEPS:
        return None
    recent = metrics[-VAL_PLATEAU_STEPS:]
    val_losses = [m["val_loss"] for m in recent]
    best = min(val_losses)
    worst = max(val_losses)
    # plateau if val loss range is less than 1% of the best value
    if (worst - best) / (best + 1e-8) < 0.005:
        return {
            "type": "val_plateau",
            "step": metrics[-1]["step"],
            "val_losses": val_losses,
            "description": f"val loss has not improved in {VAL_PLATEAU_STEPS} steps (range: {round(best,4)} - {round(worst,4)})"
        }
    return None


def detect_overfitting(metrics):
    if len(metrics) < 1:
        return None
    curr = metrics[-1]
    if curr["train_loss"] > 0 and curr["val_loss"] / curr["train_loss"] >= OVERFIT_RATIO:
        return {
            "type": "overfitting",
            "step": curr["step"],
            "train_loss": curr["train_loss"],
            "val_loss": curr["val_loss"],
            "ratio": round(curr["val_loss"] / curr["train_loss"], 3),
            "description": f"val loss ({curr['val_loss']}) is {round(curr['val_loss']/curr['train_loss'],2)}x train loss ({curr['train_loss']})"
        }
    return None


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
