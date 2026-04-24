import json
import random
import time
from pathlib import Path


# ── anomaly types ──────────────────────────────────────────────────────────────
ANOMALY_TYPES = ["loss_spike", "grad_explosion", "val_plateau", "overfitting"]


# ── inject into metrics stream ─────────────────────────────────────────────────
def inject_anomaly(metrics_file, anomaly_type=None):
    """
    Reads the last line of the metrics file, corrupts the values
    to simulate an anomaly, and appends the corrupted line.
    """
    path = Path(metrics_file)
    if not path.exists():
        print("metrics file not found, nothing to inject into")
        return

    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if not lines:
        print("metrics file is empty")
        return

    last = json.loads(lines[-1])

    if anomaly_type is None:
        anomaly_type = random.choice(ANOMALY_TYPES)

    print(f"injecting anomaly: {anomaly_type}")

    if anomaly_type == "loss_spike":
        # multiply train and val loss by a large factor
        last["train_loss"] = round(last["train_loss"] * random.uniform(8.0, 15.0), 6)
        last["val_loss"] = round(last["val_loss"] * random.uniform(6.0, 12.0), 6)
        last["anomaly_injected"] = "loss_spike"

    elif anomaly_type == "grad_explosion":
        # push grad norm way above the clipping threshold
        last["grad_norm"] = round(random.uniform(50.0, 200.0), 6)
        last["train_loss"] = round(last["train_loss"] * random.uniform(3.0, 6.0), 6)
        last["anomaly_injected"] = "grad_explosion"

    elif anomaly_type == "val_plateau":
        # freeze val accuracy and val loss across 5 synthetic steps
        base_val_loss = last["val_loss"]
        base_val_acc = last["val_acc"]
        base_step = last["step"]

        for i in range(1, 6):
            plateau_entry = {
                "step": base_step + i,
                "epoch": last["epoch"],
                "train_loss": round(last["train_loss"] * random.uniform(0.95, 1.05), 6),
                "val_loss": round(base_val_loss + random.uniform(-0.001, 0.001), 6),
                "val_acc": round(base_val_acc + random.uniform(-0.001, 0.001), 6),
                "grad_norm": round(last["grad_norm"] * random.uniform(0.9, 1.1), 6),
                "timestamp": time.time(),
                "anomaly_injected": "val_plateau"
            }
            with open(path, "a") as f:
                f.write(json.dumps(plateau_entry) + "\n")
        print("val plateau injected across 5 steps")
        return

    elif anomaly_type == "overfitting":
        # train loss keeps dropping but val loss rises
        last["train_loss"] = round(last["train_loss"] * random.uniform(0.3, 0.5), 6)
        last["val_loss"] = round(last["val_loss"] * random.uniform(2.5, 4.0), 6)
        last["val_acc"] = round(max(0.0, last["val_acc"] - random.uniform(0.1, 0.2)), 6)
        last["anomaly_injected"] = "overfitting"

    last["step"] = last["step"] + 1
    last["timestamp"] = time.time()

    with open(path, "a") as f:
        f.write(json.dumps(last) + "\n")

    print(f"injected: {json.dumps(last, indent=2)}")


# ── cli ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    anomaly = sys.argv[1] if len(sys.argv) > 1 else None
    inject_anomaly("metrics/metrics.jsonl", anomaly)
