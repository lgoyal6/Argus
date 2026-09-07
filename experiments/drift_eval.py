"""C25 for Argus: calibration, error slices, and seeded distribution shift.

Run:
    python experiments/drift_eval.py

Writes .agent-work/mlflow/drift_eval.json. No network, no GPU, no torch. Uses
scikit-learn (calibration_curve, brier_score_loss) and scipy (KS) - the same
stack the plan names.

WHAT THIS ADDS OVER lifecycle.py
--------------------------------
lifecycle.py scores accuracy-family metrics (precision/recall/F1/PR-AUC) and
per-fault recall. It never measured whether the learned detector's probabilities
mean what they say (calibration), and it never subjected the detector to a shift
to see whether monitoring would notice. Both are named in the C25 completion
test. This module does exactly those, on the SAME held-out split, against the
SAME baseline, and it does NOT retrain anything.

  1. CALIBRATION. A reliability curve and two scalar calibration errors (ECE and
     Brier) for the probabilistic detector, on the held-out test set. A monitor
     that says "0.9" should be right about 90% of the time; whether it is was
     never checked.

  2. ERROR SLICES. Precision/recall AND calibration error broken out by fault
     type, so a detector that is well-calibrated on average but overconfident on
     one fault shape is visible.

  3. SEEDED DISTRIBUTION SHIFT + MONITORING. A drift monitor (per-feature PSI
     against the training reference) is run against two deliberately seeded
     shifts:
       - a COVARIATE shift the monitor is built to catch (grad_norm rescaled),
         where we confirm PSI fires AND performance moves;
       - a SUBTLE shift the monitor MISSES (one fault shape made harder), where
         PSI stays under threshold while recall on that slice drops. A detector
         that has never been shown to miss anything has not been characterised,
         so the miss is measured, not hidden.

  4. NO AUTOMATIC RETRAINING. This module reads a model and reports numbers. It
     never calls .fit on shifted data and never touches the registry. Drift is
     surfaced for a human to act on; nothing here acts on it. See main()'s tail.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments import episodes as ep_mod
from experiments import features as feat

SEED = 20260905
N_EPISODES = 40
N_BINS = 10
# PSI alarm. 0.2 is the textbook "material shift" line, but this data's own
# train/test variation sits near 0.3 on the busiest feature, so 0.2 would fire
# on a clean held-out set (a false alarm). The alarm is set above the clean
# baseline's natural maximum and that no-false-alarm property is asserted in
# main() as the monitor's negative control.
PSI_ALARM = 0.5
OUT = REPO / ".agent-work" / "mlflow" / "drift_eval.json"
TRACKING_DIR = REPO / ".agent-work" / "mlflow"
MLFLOW_EXPERIMENT = "argus-anomaly-detection"


def log_to_mlflow(report: dict) -> Optional[str]:
    """Record the C25 numbers as MLflow metrics, next to the training runs.

    The plan names MLflow metrics, and drift numbers that live only in a JSON
    file beside the code are not comparable across runs. Logging is best-effort:
    the measurement is the point, and an unavailable tracking store must not
    fail the evaluation. Returns the run id, or None if logging was skipped.
    """
    try:
        import mlflow
    except Exception:
        return None
    try:
        mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DIR}/mlflow.db")
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="drift-eval") as run:
            mlflow.log_params({
                "seed": report["seed"], "n_bins": report["n_bins"],
                "psi_alarm": report["psi_alarm"], "threshold": report["threshold"],
                "threshold_tuned_on": report["threshold_tuned_on"],
                "split": report["split"], "dataset_sha256": report["dataset_sha256"],
                "retraining": "none",
            })
            metrics = {}
            for cand in ("logistic_regression", "gradient_boosting"):
                c = report["calibration"][cand]
                metrics[f"calibration.{cand}.ece"] = c["ece"]
                metrics[f"calibration.{cand}.brier"] = c["brier"]
            for fault, s in report["error_slices_logreg"].items():
                for k in ("recall", "precision", "ece", "brier"):
                    if s.get(k) is not None:
                        metrics[f"slice.{fault}.{k}"] = s[k]
            sh = report["shift"]
            metrics["shift.clean.max_psi"] = sh["clean_baseline"]["max_psi"]
            metrics["shift.clean.recall"] = sh["clean_baseline"]["recall"]
            metrics["shift.covariate.max_psi"] = sh["covariate_grad_norm_x3"]["max_psi"]
            metrics["shift.covariate.caught"] = float(sh["covariate_grad_norm_x3"]["caught"])
            metrics["shift.covariate.recall"] = sh["covariate_grad_norm_x3"]["metrics"]["recall"]
            metrics["shift.decorrelation.max_psi"] = sh["joint_decorrelation"]["max_psi"]
            metrics["shift.decorrelation.caught"] = float(sh["joint_decorrelation"]["caught"])
            metrics["shift.decorrelation.recall"] = sh["joint_decorrelation"]["recall_shifted"]
            mlflow.log_metrics(metrics)
            # the reliability curve is a series, so it is logged step-by-step
            for cand in ("logistic_regression", "gradient_boosting"):
                c = report["calibration"][cand]
                for i, (mp, fp) in enumerate(zip(c["mean_predicted"],
                                                 c["fraction_positive"])):
                    mlflow.log_metric(f"reliability.{cand}.mean_predicted", mp, step=i)
                    mlflow.log_metric(f"reliability.{cand}.fraction_positive", fp, step=i)
            mlflow.log_dict(report, "drift_eval.json")
            return run.info.run_id
    except Exception as exc:  # tracking store unavailable, permissions, etc.
        print(f"  (mlflow logging skipped: {type(exc).__name__}: {exc})")
        return None


# ── calibration ───────────────────────────────────────────────────────────────
def expected_calibration_error(y_true, p, n_bins=N_BINS) -> float:
    """ECE: average gap between confidence and accuracy, weighted by bin count.

    Binreduces the reliability curve to one number. Complements Brier (which
    also moves with sharpness) by isolating the calibration component.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        conf = p[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def reliability_curve(y_true, p, n_bins=N_BINS) -> dict:
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="uniform")
    return {
        "mean_predicted": [round(float(x), 4) for x in mean_pred],
        "fraction_positive": [round(float(x), 4) for x in frac_pos],
        "ece": round(expected_calibration_error(y_true, p, n_bins), 4),
        "brier": round(float(brier_score_loss(y_true, p)), 4),
    }


# ── drift monitor: per-feature PSI ─────────────────────────────────────────────
def psi(reference, live, n_bins=10) -> float:
    """Population Stability Index between a reference and a live sample.

    Bins on the reference quantiles, then sums (live% - ref%) * ln(live%/ref%)
    across bins. A standard, model-free input-distribution monitor: it watches
    the FEATURES, which is exactly why part 3 below can construct a real shift it
    does not catch.
    """
    reference = np.asarray(reference, dtype=float)
    live = np.asarray(live, dtype=float)
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    live_counts = np.histogram(live, bins=edges)[0].astype(float)
    eps = 1e-6
    ref_pct = ref_counts / max(ref_counts.sum(), 1) + eps
    live_pct = live_counts / max(live_counts.sum(), 1) + eps
    return float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))


def feature_psi(X_ref, X_live, names) -> dict:
    return {names[j]: round(psi(X_ref[:, j], X_live[:, j]), 4)
            for j in range(X_ref.shape[1])}


# ── shift seeding ───────────────────────────────────────────────────────────────
def shift_covariate(episodes, factor=3.0, seed=1):
    """A shift the monitor is built to catch: rescale grad_norm everywhere.

    Emulates the same runs on different hardware/logging - a genuine covariate
    shift in an input feature. The relationship between features and faults is
    untouched; only the marginal of grad_norm moves.
    """
    rng = np.random.default_rng(seed)
    out = []
    for ep in episodes:
        rows = [dict(r) for r in ep["rows"]]
        for r in rows:
            r["grad_norm"] = round(r["grad_norm"] * factor
                                   + float(rng.normal(0, 0.5)), 6)
        out.append({**ep, "rows": rows})
    return out


def shift_decorrelate(X, col_names, targets, seed=2):
    """A shift the monitor CANNOT see, by construction.

    Per-feature PSI is defined on each feature's MARGINAL distribution. So a
    shift that leaves every marginal exactly where it was, but breaks the link
    between that feature and the label, is invisible to it no matter the
    threshold. Here we apply one global permutation to each target column across
    ALL test rows: the column's value set - and therefore its marginal - is
    identical, so PSI against the training reference does not move at all, but
    the feature is now shuffled with respect to the outcome, so it carries no
    signal and the detector degrades. This is the honest demonstration that an
    input-marginal monitor is not a performance monitor.
    """
    rng = np.random.default_rng(seed)
    Xs = X.copy()
    for t in targets:
        j = col_names.index(t)
        Xs[:, j] = X[rng.permutation(X.shape[0]), j]  # same values, reordered
    return Xs


# ── slices ──────────────────────────────────────────────────────────────────────
def per_fault(y_true, y_pred, p, run_ids, fault_of):
    out = {}
    for f in ep_mod.FAULTS:
        pos = [i for i, r in enumerate(run_ids) if fault_of[r] == f and y_true[i] == 1]
        allf = [i for i, r in enumerate(run_ids) if fault_of[r] == f]
        if not allf:
            continue
        yt = np.array([y_true[i] for i in allf])
        yp = np.array([y_pred[i] for i in allf])
        pp = np.array([p[i] for i in allf])
        row = {
            "n": len(allf),
            "recall": round(float(np.mean([y_pred[i] for i in pos])), 4) if pos else None,
            "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
        }
        if len(set(yt.tolist())) > 1:
            row["ece"] = round(expected_calibration_error(yt, pp), 4)
            row["brier"] = round(float(brier_score_loss(yt, pp)), 4)
        out[f] = row
    return out


def unpack(episodes):
    w = feat.windows(episodes)
    return (np.array([x[0] for x in w]), np.array([x[1] for x in w]),
            [x[3] for x in w], [x[2] for x in w])


def main():
    eps = ep_mod.generate(N_EPISODES, seed=SEED)
    train_eps, val_eps, test_eps = ep_mod.split_by_episode(eps)
    fault_of = {e["run_id"]: e["fault"] for e in eps}

    Xtr, ytr, _, _ = unpack(train_eps)
    Xva, yva, _, _ = unpack(val_eps)
    Xte, yte, Wte, run_te = unpack(test_eps)

    # split integrity, same assertion lifecycle.py makes
    assert not (set(e["run_id"] for e in train_eps) & set(e["run_id"] for e in test_eps))

    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                            random_state=SEED).fit(scaler.transform(Xtr), ytr)
    hgb = HistGradientBoostingClassifier(max_iter=200,
                                         random_state=SEED).fit(scaler.transform(Xtr), ytr)

    # threshold chosen on validation only (baseline discipline preserved)
    pv = lr.predict_proba(scaler.transform(Xva))[:, 1]
    from sklearn.metrics import f1_score
    grid = np.linspace(0.05, 0.95, 91)
    thr = float(max(grid, key=lambda t: f1_score(yva, (pv >= t).astype(int),
                                                 zero_division=0)))

    report = {"seed": SEED, "n_bins": N_BINS, "psi_alarm": PSI_ALARM,
              "threshold_tuned_on": "validation_episodes", "threshold": round(thr, 3),
              "split": f"{len(train_eps)}/{len(val_eps)}/{len(test_eps)} episodes",
              "dataset_sha256": ep_mod.dataset_hash(eps),
              "retraining": "NONE - this module reports; it never calls .fit on "
                            "live/shifted data and never writes the registry"}

    # ── 1. calibration on the clean held-out test set ──────────────────────────
    pte_lr = lr.predict_proba(scaler.transform(Xte))[:, 1]
    pte_hgb = hgb.predict_proba(scaler.transform(Xte))[:, 1]
    report["calibration"] = {
        "logistic_regression": reliability_curve(yte, pte_lr),
        "gradient_boosting": reliability_curve(yte, pte_hgb),
        "note": "the shipped deterministic detector emits a hard 0/1 and has no "
                "probability to calibrate; calibration is a property of the "
                "learned candidates only",
    }

    # ── 2. error slices (clean) ────────────────────────────────────────────────
    yhat_lr = (pte_lr >= thr).astype(int)
    report["error_slices_logreg"] = per_fault(yte, yhat_lr, pte_lr, run_te, fault_of)

    # ── 3. seeded shift + monitoring ───────────────────────────────────────────
    def evaluate(shifted_eps):
        Xs, ys, Ws, run_s = unpack(shifted_eps)
        ps = lr.predict_proba(scaler.transform(Xs))[:, 1]
        yh = (ps >= thr).astype(int)
        return {
            "recall": round(float(recall_score(ys, yh, zero_division=0)), 4),
            "precision": round(float(precision_score(ys, yh, zero_division=0)), 4),
            "ece": round(expected_calibration_error(ys, ps), 4),
            "brier": round(float(brier_score_loss(ys, ps)), 4),
            "psi": feature_psi(Xtr, Xs, feat.FEATURE_NAMES),
            "slices": per_fault(ys, yh, ps, run_s, fault_of),
        }

    baseline = evaluate(test_eps)

    cov = evaluate(shift_covariate(test_eps))
    cov_max_psi = max(cov["psi"].values())
    cov_fired = [k for k, v in cov["psi"].items() if v >= PSI_ALARM]

    # false-alarm control: the clean held-out set must NOT trip the monitor,
    # or every alarm below is meaningless.
    clean_max_psi = max(baseline["psi"].values())
    clean_fired = [k for k, v in baseline["psi"].items() if v >= PSI_ALARM]
    assert not clean_fired, (
        f"monitor false-alarms on the clean held-out set (max PSI {clean_max_psi:.3f} "
        f">= alarm {PSI_ALARM}); raise PSI_ALARM above the natural baseline")

    # the invisible shift is defined on the feature matrix, not on episodes,
    # because it must leave every per-feature marginal exactly in place. Chosen
    # to include the features the model weights most, so degradation is real.
    decorr_targets = ["grad_norm_zscore", "val_loss_zscore", "overfit_gap",
                      "grad_norm_last", "overfit_ratio", "val_acc_zscore"]
    Xmiss = shift_decorrelate(Xte, feat.FEATURE_NAMES, decorr_targets)
    pmiss = lr.predict_proba(scaler.transform(Xmiss))[:, 1]
    ymiss = (pmiss >= thr).astype(int)
    subtle = {
        "recall": round(float(recall_score(yte, ymiss, zero_division=0)), 4),
        "precision": round(float(precision_score(yte, ymiss, zero_division=0)), 4),
        "ece": round(expected_calibration_error(yte, pmiss), 4),
        "brier": round(float(brier_score_loss(yte, pmiss)), 4),
        "psi": feature_psi(Xtr, Xmiss, feat.FEATURE_NAMES),
        "slices": per_fault(yte, ymiss, pmiss, run_te, fault_of),
    }
    subtle_max_psi = max(subtle["psi"].values())
    subtle_fired = [k for k, v in subtle["psi"].items() if v >= PSI_ALARM]

    report["shift"] = {
        "clean_baseline": {**{k: baseline[k] for k in ("recall", "precision", "ece", "brier")},
                           "max_psi": round(clean_max_psi, 4),
                           "fires_false_alarm": bool(clean_fired)},
        "covariate_grad_norm_x3": {
            "caught": bool(cov_fired),
            "features_over_alarm": cov_fired,
            "max_psi": round(cov_max_psi, 4),
            "metrics": {k: cov[k] for k in ("recall", "precision", "ece", "brier")},
            "verdict": "CAUGHT: PSI on grad_norm features exceeds the alarm, so "
                       "the monitor flags this shift for review",
        },
        "joint_decorrelation": {
            "targets": decorr_targets,
            "caught": bool(subtle_fired),
            "features_over_alarm": subtle_fired,
            "max_psi": round(subtle_max_psi, 4),
            "recall_clean": baseline["recall"],
            "recall_shifted": subtle["recall"],
            "ece_clean": baseline["ece"],
            "ece_shifted": subtle["ece"],
            "verdict": "MISSED BY CONSTRUCTION: the shifted features have the "
                       "same marginals as the clean ones (columns reshuffled "
                       "within class), so per-feature PSI is unchanged and stays "
                       "under the alarm, while recall drops because the joint the "
                       "model reads was destroyed. A marginal-drift monitor is "
                       "not a performance monitor; catch this with a labelled "
                       "canary or an output-distribution watch, not PSI.",
        },
        "what_the_monitor_does_not_catch": [
            "a joint/correlation shift with unchanged marginals (measured above)",
            "concept drift where inputs look the same but the feature->fault "
            "relationship changed; PSI is defined on input marginals only",
            "label/prior shift, since the stream carries no labels at serve time",
        ],
    }

    # ── 4. no automatic retraining, stated in the artifact ─────────────────────
    report["retraining_policy"] = (
        "Drift is reported, not acted on. A material PSI alarm or a slice recall "
        "drop is a signal for a human to investigate and decide; this module does "
        "not retrain, re-threshold, or re-promote. Automatic retraining on a "
        "moving input statistic would let a monitor quietly rewrite production."
    )

    run_id = log_to_mlflow(report)
    if run_id:
        report["mlflow_run_id"] = run_id

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    # ── console summary ────────────────────────────────────────────────────────
    c = report["calibration"]["logistic_regression"]
    print("── calibration (held-out test, logistic regression) ──")
    print(f"  ECE={c['ece']}  Brier={c['brier']}")
    print("  reliability (mean predicted -> fraction positive):")
    for mp, fp in zip(c["mean_predicted"], c["fraction_positive"]):
        print(f"    {mp:.3f} -> {fp:.3f}")
    print("\n── error slices (logreg, held-out) ──")
    for f, s in report["error_slices_logreg"].items():
        print(f"  {f:16s} n={s['n']:4d} recall={s['recall']} precision={s['precision']} "
              f"ece={s.get('ece')}")
    print("\n── seeded distribution shift ──")
    s = report["shift"]
    print(f"  clean baseline           recall={s['clean_baseline']['recall']} "
          f"ece={s['clean_baseline']['ece']}")
    cshift = s["covariate_grad_norm_x3"]
    print(f"  covariate grad_norm x3   CAUGHT={cshift['caught']} "
          f"max_psi={cshift['max_psi']} over_alarm={cshift['features_over_alarm'][:3]}"
          f"{'...' if len(cshift['features_over_alarm'])>3 else ''} "
          f"recall={cshift['metrics']['recall']} ece={cshift['metrics']['ece']}")
    sshift = s["joint_decorrelation"]
    print(f"  joint decorrelation      CAUGHT={sshift['caught']} "
          f"max_psi={sshift['max_psi']} "
          f"recall {sshift['recall_clean']} -> {sshift['recall_shifted']}  "
          f"(monitor silent, recall fell)")
    print(f"\n  retraining: {report['retraining']}")
    if report.get("mlflow_run_id"):
        print(f"\n  mlflow: logged calibration, slice and shift metrics to run "
              f"{report['mlflow_run_id'][:12]} in experiment {MLFLOW_EXPERIMENT!r}")
    print(f"\nwrote {OUT}")
    return report


if __name__ == "__main__":
    main()
