"""Windows over a metric stream, and the candidate scorers built on them.

Every candidate answers the same question on the same input so the comparison is
about the decision rule and nothing else: given the last ROLLING_WINDOW+1 rows of a
run, is the final row anomalous?

The deterministic candidate is the shipped detector, wrapped rather than
reimplemented. Scoring a paraphrase of it would compare the learned models against a
strawman, and the point of the exercise is whether they beat the thing actually in
production.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.detector import ROLLING_WINDOW, detect_anomalies_in

WINDOW = ROLLING_WINDOW + 1


def windows(episodes):
    """(features, label, run_id, rows) per position, oldest first.

    Windows within one run overlap by construction. That is fine as long as a split
    never cuts through a run - see episodes.split_by_episode.
    """
    out = []
    for ep in episodes:
        rows = ep["rows"]
        for end in range(WINDOW, len(rows) + 1):
            w = rows[end - WINDOW:end]
            out.append((
                extract(w),
                1 if w[-1]["fault"] != "none" else 0,
                ep["run_id"],
                w,
            ))
    return out


def extract(w):
    """Plain summary statistics of the window: level, spread, trend and z-scores.

    Deliberately the same information the deterministic detector reads. Handing the
    learned models extra signal the detector cannot see would make any win theirs by
    construction rather than by modelling.
    """
    feats = []
    for key in ("train_loss", "val_loss", "grad_norm", "val_acc"):
        v = [r[key] for r in w]
        ref, last = v[:-1], v[-1]
        mu, sd = mean(ref), pstdev(ref)
        feats += [
            last,
            mu,
            sd,
            (last - mu) / sd if sd > 0 else 0.0,          # z against prior window
            last - ref[-1],                                # one-step delta
            v[-1] - v[0],                                  # window trend
            max(v) - min(v),                               # window range
        ]
    tl, vl = w[-1]["train_loss"], w[-1]["val_loss"]
    feats += [
        vl / tl if tl > 0 else 0.0,                        # overfit ratio
        vl - tl,                                           # overfit gap
        (vl - tl) - (w[0]["val_loss"] - w[0]["train_loss"]),  # gap trend
    ]
    return feats


def deterministic_predict(rows_list):
    """The shipped detector, as a binary scorer over windows."""
    return [1 if detect_anomalies_in(w) else 0 for w in rows_list]


FEATURE_NAMES = [
    f"{k}_{s}"
    for k in ("train_loss", "val_loss", "grad_norm", "val_acc")
    for s in ("last", "mean", "std", "zscore", "delta", "trend", "range")
] + ["overfit_ratio", "overfit_gap", "gap_trend"]
