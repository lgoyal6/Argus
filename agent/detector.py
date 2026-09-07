import json
from pathlib import Path
from statistics import mean, stdev

# obs.py sits at the repository root and is copied into /app by docker/Dockerfile.agent,
# so it is flat beside this module in the image. tests/test_flat_layout.py enforces the
# COPY, because losing it starts the container and kills it on the first import.
import obs


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
# The deepest lookback any detector below needs: rolling_zscore reads
# ROLLING_WINDOW values plus the sample under test, and every other check is
# shallower. Reading more than this per cycle is work that grows with the run
# while the answer does not.
REQUIRED_HISTORY = max(ROLLING_WINDOW + 1, VAL_PLATEAU_STEPS, OVERFIT_TREND_STEPS)


def _tail_lines(path, count, chunk_size=8 * 1024):
    """Last `count` non-empty lines, read backwards from the end of the file.

    Reading the whole file to look at the last twenty rows makes each detection
    cycle cost O(run length); by step 10,000 that is 10,000 rows parsed to decide
    something the last twenty already determine. One 8 KiB block holds far more
    than REQUIRED_HISTORY rows at any realistic row size; the loop only reads a
    second block if it does not.
    """
    with open(path, "rb") as f:
        f.seek(0, 2)
        end = f.tell()
        blocks = []
        newlines = 0
        # One extra newline: the first line in the window is usually partial.
        while end > 0 and newlines <= count:
            size = min(chunk_size, end)
            end -= size
            f.seek(end)
            block = f.read(size)
            newlines += block.count(b"\n")
            blocks.append(block)
    data = b"".join(reversed(blocks))
    lines = [l.strip() for l in data.decode("utf-8", "replace").splitlines() if l.strip()]
    return lines[-count:]


def load_metrics(metrics_file, history=REQUIRED_HISTORY):
    """Recent metric entries, newest last.

    history=None reads the whole file, which the CLI and any caller that wants a
    full view can still ask for; the detectors do not need it.
    """
    path = Path(metrics_file)
    if not path.exists():
        return []
    if history is None:
        with open(path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
    else:
        lines = _tail_lines(path, history)
    out = []
    for line in lines:
        try:
            row, _ = obs.take_trace(json.loads(line))
            out.append(row)
        except ValueError:
            # A torn last line (the trainer mid-write) or a corrupt record must
            # not stop detection on the records that did parse.
            continue
    return out


def newest_trace_context(metrics_file):
    """The parent context of the last row on the queue, if it carried one.

    This is the consumer half of the hop. The agent is a different process from the
    trainer and usually was not running when the row was written, so nothing is
    inherited from an ambient context; the row itself is the only channel. Reading
    the last row rather than all of them is deliberate: a detection cycle is
    triggered by the newest sample, so that is the caller it belongs to.
    """
    path = Path(metrics_file)
    if not path.exists():
        return None
    for line in reversed(_tail_lines(path, 1)):
        try:
            _, traceparent = obs.take_trace(json.loads(line))
        except ValueError:
            return None
        return traceparent
    return None


# ── statistical helpers ────────────────────────────────────────────────────────
def rolling_zscore(values, window=ROLLING_WINDOW):
    """Z-score of the last value against the window of values BEFORE it.

    The sample under test is deliberately excluded from its own reference
    statistics. Including it - as this did originally - bounds the achievable
    |z| at (n-1)/sqrt(n) no matter how extreme the anomaly. With window=20 that
    ceiling is 4.249, which left GRAD_EXPLOSION_ZSCORE (4.0) reachable only in a
    6% sliver of its range and mathematically unreachable below 19 samples; a 5x
    jump on two samples scored 0.707. Excluding the sample removes the bound, so
    a bigger anomaly scores higher instead of saturating.

    Returns None when there is not enough history, and when the reference window
    has zero spread. A flat history genuinely cannot say whether the next value is
    anomalous, and inventing a score there manufactures alarms on constant metrics
    (a frozen gradient, or a metric the trainer has stopped updating). The hard
    thresholds still cover those cases.
    """
    if len(values) < 2:
        return None
    reference = values[-(window + 1):-1]
    if len(reference) < 2:
        return None
    mu = mean(reference)
    sigma = stdev(reference)
    if sigma == 0:
        return None
    return (values[-1] - mu) / sigma


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
        "description": f"grad norm {grad} - {', '.join(reason)}"
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
CHECKS = (
    detect_loss_spike,
    detect_grad_explosion,
    detect_val_plateau,
    detect_overfitting,
)


def detect_anomalies_in(metrics):
    """Run every check over metric rows already in hand.

    Split out from detect_anomalies so a caller holding a specific window can ask
    about that window. agent/attempt.py needs exactly this: to decide whether a
    recovery corrected anything it has to re-run the triggering check over the rows
    written AFTER the attempt, not over whatever the file happens to end with.
    """
    if not metrics:
        return []
    return [result for check in CHECKS if (result := check(metrics))]


def detect_anomalies(metrics_file):
    """One detection cycle over the tail of the metrics file.

    Instrumented here rather than in detect_anomalies_in because agent/attempt.py calls
    that one to re-check the triggering anomaly AFTER a recovery. Counting both through
    the same metric would mix "the system found a fault" with "the verifier looked
    again", and the ratio between those two is the number the recovery work exists to
    report honestly.

    The anomaly dictionaries carry train_loss, the z-score and a rendered description.
    None of that goes into an attribute: those values are already in the metrics stream,
    which is the copy an auditor should be reading. Only the type and the step travel,
    and the type is the sole label because it is the only field with a closed value set.
    """
    with obs.span("detection.cycle", kind="consumer",
                  parent=obs.inherited_or(newest_trace_context(metrics_file)),
                  metrics_file_present=Path(metrics_file).exists()):
        results = detect_anomalies_in(load_metrics(metrics_file))
        for anomaly in results:
            obs.incr("argus_detection_anomalies_total", type=anomaly["type"])
            obs.log("detection.anomaly", type=anomaly["type"], step=anomaly.get("step"))
        return results


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
