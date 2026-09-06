"""Synthetic training-run episodes, and an explicit account of what they are not.

WHAT THIS IS NOT: real data. This repository ships no recorded training runs -
training_job/metrics/ and agent/logs/ are both gitignored and empty - so there are
zero real labelled failure episodes to learn from. Everything below is generated.

That matters for how the results may be read. A detector trained on generated faults
learns the generator, and the generator's faults are placed by rules that a threshold
can also be written against. So a learned model doing well here is evidence about this
generator, not evidence that it would beat thresholds on a real cluster. The comparison
in lifecycle.py is therefore run to find out whether a learned model beats the
deterministic detector *even on data built to be learnable*, which is the friendly
case for it. Losing that comparison is a strong result; winning it is a weak one.

To keep it from being a pure restatement of the detector's own thresholds, the faults
here vary in severity and include two that are deliberately subtle - a slow plateau and
a gradual overfit, both of which develop under the hard thresholds. If a learned model
is going to earn its place anywhere, it is on those.

Each episode is one independent run: its own seed, its own trajectory, its own fault.
Episodes never share rows, which is what makes a split by episode a real held-out split.
"""

from __future__ import annotations

import hashlib
import json
import random

FAULTS = ("none", "loss_spike", "grad_explosion", "val_plateau", "overfitting")

# Rows per episode and how often the trainer emits, matching training_job/config.yaml.
EPISODE_ROWS = 120
EMIT_EVERY = 10


SPIKE_ROWS = 3  # a loss spike is transient; the sustained faults are not


def _episode(run_id, fault, rng):
    """One run's metric rows, healthy until the fault starts, then affected.

    The fault perturbs the EMITTED values; it never feeds back into the underlying
    trajectory. That separation is load-bearing, and getting it wrong the first time
    invalidated the whole comparison: applying the spike multiplier to the running
    loss variable compounded it over three rows to ~200x and left it there for the
    rest of the episode, decaying slowly. Every remaining row was then labelled
    faulty while carrying a level no healthy run ever reaches, so a classifier could
    score those rows by recognising which regime the episode was in and never detect
    anything. The deterministic detector was scored as missing them - correctly, since
    a high but flat loss has no anomaly signal in it - and lost on a labelling
    artifact rather than on merit.

    So a row is labelled faulty only while the fault's signal is actually present:
    the three rows of a spike, and every row after onset for the faults that really
    are ongoing states.
    """
    rows = []
    train_loss = rng.uniform(2.0, 2.4)
    val_loss = train_loss + rng.uniform(0.05, 0.15)
    grad = rng.uniform(0.8, 2.0)
    # Faults start well past the detector's 20-row reference window, so a window at
    # the onset has a healthy history to be judged against.
    onset = rng.randint(40, 80) if fault != "none" else EPISODE_ROWS + 1
    severity = rng.uniform(0.3, 1.0)
    plateau_at = None

    for i in range(EPISODE_ROWS):
        step = (i + 1) * EMIT_EVERY
        decay = rng.uniform(0.985, 0.999)
        # The clean trajectory, advanced identically whether or not a fault is active.
        train_loss = max(0.05, train_loss * decay + rng.gauss(0, 0.01))
        val_loss = max(0.05, val_loss * decay + rng.gauss(0, 0.015))
        grad = max(0.05, grad * rng.uniform(0.97, 1.03) + rng.gauss(0, 0.05))

        e_train, e_val, e_grad = train_loss, val_loss, grad
        started = i >= onset
        active = started

        if started and fault == "loss_spike":
            # Sharp and short: a few rows outside the recent spread, then recovery.
            active = (i - onset) < SPIKE_ROWS
            if active:
                e_train = train_loss * (1 + 9 * severity)
                e_val = val_loss * (1 + 7 * severity)
        elif started and fault == "grad_explosion":
            e_grad = rng.uniform(20.0, 200.0) * severity + 15.0
            e_train = train_loss * (1 + 3 * severity)
        elif started and fault == "val_plateau":
            # Subtle: val loss simply stops moving while train loss keeps falling.
            if plateau_at is None:
                plateau_at = val_loss
            e_val = plateau_at + rng.gauss(0, 0.002)
        elif started and fault == "overfitting":
            # Subtle and gradual: the gap opens a little more every step.
            drift = (i - onset) * 0.02 * severity
            e_train = max(0.02, train_loss - drift * 0.5)
            e_val = val_loss + drift

        rows.append({
            "run_id": run_id,
            "step": step,
            "epoch": step // 500,
            "train_loss": round(e_train, 6),
            "val_loss": round(e_val, 6),
            "val_acc": round(max(0.0, min(1.0, 1.0 - e_val / 3.0)), 6),
            "grad_norm": round(e_grad, 6),
            # The label. Present only in generated data; a real stream has no such
            # column, which is precisely the problem this experiment is bounded by.
            "fault": fault if active else "none",
        })
    return rows


def generate(n_episodes=40, seed=20260905):
    """`n_episodes` independent runs. Deterministic given the seed."""
    master = random.Random(seed)
    episodes = []
    for run_index in range(n_episodes):
        fault = FAULTS[run_index % len(FAULTS)]
        rng = random.Random(master.randrange(2**32))
        episodes.append({
            "run_id": f"ep-{run_index:03d}",
            "fault": fault,
            "rows": _episode(f"ep-{run_index:03d}", fault, rng),
        })
    return episodes


def dataset_hash(episodes):
    """Content hash of the generated data, for the tracking record.

    Pins the exact rows a run was scored on, so "same seed" is verifiable rather than
    asserted.
    """
    h = hashlib.sha256()
    for ep in episodes:
        h.update(json.dumps(ep, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def split_by_episode(episodes, train=0.5, val=0.2):
    """Chronological split on whole episodes.

    Whole episodes, because windows inside one run overlap: consecutive windows share
    rows, so splitting on windows would put near-copies of the same moment on both
    sides and score memorisation as generalisation. Chronological rather than shuffled,
    because a monitor is always asked about runs it has not seen yet.
    """
    n = len(episodes)
    a, b = int(n * train), int(n * (train + val))
    return episodes[:a], episodes[a:b], episodes[b:]
