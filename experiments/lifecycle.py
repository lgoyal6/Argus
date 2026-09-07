"""Does a learned detector earn a place in Argus, and can we promote and undo one?

Two separable questions, run in one pass.

The first is a decision. Argus ships a deterministic detector: hard thresholds plus
rolling z-scores and a CUSUM. The claim worth testing is that a learned model beats it
on the same input. This scores three candidates - the shipped detector, logistic
regression and gradient boosting - once on a held-out set of whole runs, having tuned
only on validation runs, and applies quality AND latency gates before anything is
allowed near production. The honest outcome is whichever one the numbers give; a
learned model that does not clear the gates does not get promoted, and that result is
worth publishing rather than working around.

The second is a capability. Whatever wins, the lifecycle has to be real: every run
records the code revision, dataset hash, seed, configuration and environment it ran
under; artifacts are registered as versions of one model with one input contract; a
version is promoted, smoke-tested against known inputs, and then rolled back to its
predecessor, which is verified to still serve.

Tracking is a local SQLite file and a local artifact directory. No server: the registry
needs a database-backed store rather than a bare file store, and SQLite is one.

    python experiments/lifecycle.py
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments import artifact_guard as guard
from experiments import episodes as ep_mod
from experiments import features as feat

# ── experiment configuration ───────────────────────────────────────────────────
SEED = 20260905
N_EPISODES = 40
EXPERIMENT = "argus-anomaly-detection"
REGISTERED_MODEL = "argus-anomaly-detector"

# Promotion gates. A learned model has to be enough better to pay for itself: it adds
# a training pipeline, a feature contract and an artifact to keep current, against a
# detector that is forty lines of arithmetic. A rounding-error win is not worth that,
# so the bar is an absolute margin rather than "beats the baseline".
MIN_F1_GAIN = 0.02
MIN_RECALL = 0.80          # a monitor that misses failures is not a monitor

# Per-window decision latency, gated on the MEDIAN. p99 is the statistic you would
# normally want, and it is recorded, but neither statistic is stable enough here to
# hang a promotion on. Two runs of THIS FILE, unchanged, on the same machine at
# different system loads measured gradient boosting at:
#
#     median 27.035 ms / p99 846.749 ms   (host load average ~89)
#     median  1.483 ms / p99   3.726 ms   (host load average ~34)
#
# an 18x swing on the median and 227x on the p99, which flipped this gate from False
# to True while every quality metric reproduced to the last decimal place. So the
# median is the less bad of two unreliable numbers here, not a trustworthy one: read
# the gate as a smoke alarm for order-of-magnitude cost regressions on a quiet
# machine, and never as a certification. Against a 10-second poll interval there is
# enormous headroom either way, which is why this is not the gate that decides
# anything below.
LATENCY_BUDGET_MS = 5.0

# The gate that actually decides this comparison.
#
# Argus has no recorded training runs: training_job/metrics/ and agent/logs/ are
# gitignored and empty, so the episodes scored below are generated. A model fitted to
# a generator has learned the generator. It can be measured, registered and served -
# all of which is worth having working - but it cannot be promoted on the strength of
# a result about synthetic data, however large the margin. Encoding that as a gate
# rather than as a caveat in a write-up is the point: the pipeline refuses, and the
# refusal is recorded next to the metrics that would otherwise have justified it.
REQUIRE_REAL_EPISODES = True

TRACKING_DIR = REPO / ".agent-work" / "mlflow"
# Per-version file digests, written at registration and verified at promotion.
MANIFEST_DIR = TRACKING_DIR / "manifests"


# ── provenance ─────────────────────────────────────────────────────────────────
def code_revision():
    """The commit the experiment ran at, plus whether the tree was dirty.

    A hash alone is a half-truth when there are uncommitted edits, and a result that
    cannot be tied to code is not reproducible.
    """
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain"))}


def environment():
    import sklearn
    return {"python": platform.python_version(), "platform": platform.platform(),
            "mlflow": mlflow.__version__, "sklearn": sklearn.__version__,
            "numpy": np.__version__}


# ── the served contract ────────────────────────────────────────────────────────
# Both candidates are pyfunc models over the SAME input: the raw metric rows of a
# window, exactly what a monitor holds. Feature extraction happens inside the learned
# model, not in the caller. That is what makes the versions interchangeable, and it is
# the difference between a rollback that works and one that needs a caller change too.
INPUT_COLUMNS = ["window_id", "pos", "step", "train_loss", "val_loss", "val_acc", "grad_norm"]


def to_frame(rows_list):
    import pandas as pd
    records = []
    for wid, w in enumerate(rows_list):
        for pos, r in enumerate(w):
            records.append({"window_id": wid, "pos": pos, "step": r["step"],
                            "train_loss": r["train_loss"], "val_loss": r["val_loss"],
                            "val_acc": r["val_acc"], "grad_norm": r["grad_norm"]})
    return pd.DataFrame.from_records(records, columns=INPUT_COLUMNS)


def _regroup(df):
    out = []
    for _, g in df.sort_values(["window_id", "pos"]).groupby("window_id", sort=True):
        out.append(g.to_dict("records"))
    return out


class DeterministicDetector(mlflow.pyfunc.PythonModel):
    """The shipped detector, served through the registry like any other version."""

    def predict(self, context, model_input, params=None):
        from experiments.features import deterministic_predict
        return np.array(deterministic_predict(_regroup(model_input)))


class LearnedDetector(mlflow.pyfunc.PythonModel):
    def __init__(self, model, scaler, threshold):
        self.model, self.scaler, self.threshold = model, scaler, threshold

    def predict(self, context, model_input, params=None):
        from experiments.features import extract
        X = np.array([extract(w) for w in _regroup(model_input)])
        p = self.model.predict_proba(self.scaler.transform(X))[:, 1]
        return (p >= self.threshold).astype(int)


def log_safe_detector(spec):
    """Log the data-only representation accepted by the activation guard."""
    mlflow.log_dict(spec, "safe-model/detector.json")
    if spec.get("feature_order"):
        mlflow.log_dict({"features": spec["feature_order"]},
                        "safe-model/feature_contract.json")
    mlflow.log_text(
        "artifact_path: safe-model\n"
        "flavors:\n"
        "  argus_safe:\n"
        "    data: detector.json\n"
        "    format: argus-detector-v1\n",
        "safe-model/MLmodel",
    )


# ── scoring ────────────────────────────────────────────────────────────────────
def score(y_true, y_pred, scores=None):
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if scores is not None:
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    return out


def latency_ms(model, samples, warmup=20):
    """Median and p99 per-window decision latency through the served pyfunc contract.

    Every candidate is timed the same way, on the same inputs, including the frame
    the caller would hand it. Timing the deterministic detector on raw lists while the
    learned ones paid for a DataFrame round trip measured the harness, not the models,
    and made the gap between them mostly pandas.

    Warmed up first: the first call through a freshly loaded model pays one-off import
    and allocation costs that a monitor polling every ten seconds never pays again,
    and at 500 samples a single such outlier lands directly on the p99.
    """
    frames = [to_frame([w]) for w in samples]
    for f in frames[:warmup]:
        model.predict(None, f)
    times = []
    for f in frames:
        t = time.perf_counter()
        model.predict(None, f)
        times.append((time.perf_counter() - t) * 1000)
    return {"latency_median_ms": float(np.median(times)),
            "latency_p99_ms": float(np.percentile(times, 99))}


def per_fault_recall(y_true, y_pred, run_ids, fault_of):
    """Recall broken out by fault type.

    The aggregate hides the finding. A candidate can win overall while losing on the
    faults that matter, or - as here - win entirely on two fault shapes and tie or
    lose on the rest, which says something quite different about whether it is better.
    """
    out = {}
    for f in ep_mod.FAULTS:
        idx = [i for i, r in enumerate(run_ids) if fault_of[r] == f and y_true[i] == 1]
        if idx:
            out[f"recall_{f}"] = float(np.mean([y_pred[i] for i in idx]))
    return out


def main():
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DIR}/mlflow.db")
    mlflow.set_registry_uri(f"sqlite:///{TRACKING_DIR}/mlflow.db")
    client = MlflowClient()

    # The artifact root has to be set when the experiment is created; setting only the
    # tracking URI sends metadata to SQLite and leaves the artifacts themselves in
    # ./mlruns at whatever the working directory happens to be, which drops a few MB
    # of serialised models into the repository root.
    if client.get_experiment_by_name(EXPERIMENT) is None:
        client.create_experiment(EXPERIMENT,
                                 artifact_location=f"file://{TRACKING_DIR}/artifacts")
    mlflow.set_experiment(EXPERIMENT)

    # ── data, split by whole run ────────────────────────────────────────────────
    eps = ep_mod.generate(N_EPISODES, seed=SEED)
    train_eps, val_eps, test_eps = ep_mod.split_by_episode(eps)
    dhash = ep_mod.dataset_hash(eps)

    def unpack(e):
        w = feat.windows(e)
        return (np.array([x[0] for x in w]), np.array([x[1] for x in w]),
                [x[3] for x in w], [x[2] for x in w])

    Xtr, ytr, _, _ = unpack(train_eps)
    Xva, yva, Wva, _ = unpack(val_eps)
    Xte, yte, Wte, run_te = unpack(test_eps)
    fault_of = {e["run_id"]: e["fault"] for e in eps}

    # The split has to be airtight or every number below is fiction.
    assert not (set(r["run_id"] for r in train_eps) & set(r["run_id"] for r in test_eps))
    assert not (set(r["run_id"] for r in val_eps) & set(r["run_id"] for r in test_eps))

    common = {
        "seed": SEED, "n_episodes": N_EPISODES, "dataset_sha256": dhash,
        "window": feat.WINDOW,
        "split": f"{len(train_eps)}/{len(val_eps)}/{len(test_eps)} episodes",
        "train_windows": len(ytr), "val_windows": len(yva), "test_windows": len(yte),
        "test_positive_rate": round(float(yte.mean()), 4),
        **code_revision(),
    }
    print(json.dumps({"dataset": common}, indent=2))

    results = {}
    scaler = StandardScaler().fit(Xtr)

    # ── candidate 1: the shipped deterministic detector ─────────────────────────
    with mlflow.start_run(run_name="deterministic-thresholds") as run:
        mlflow.log_params({**common, "candidate": "deterministic",
                           "trainable_parameters": 0})
        mlflow.log_dict(environment(), "environment.json")
        yp = np.array(feat.deterministic_predict(Wte))
        m = score(yte, yp)
        m.update(latency_ms(DeterministicDetector(), Wte[:500]))
        m.update(per_fault_recall(yte, yp, run_te, fault_of))
        mlflow.log_metrics(m)
        mlflow.pyfunc.log_model(
            name="model", python_model=DeterministicDetector(),
            code_paths=[str(REPO / "agent"), str(REPO / "experiments")],
            input_example=to_frame(Wte[:2]),
        )
        log_safe_detector({"format": "argus-detector-v1", "kind": "deterministic"})
        results["deterministic"] = {**m, "run_id": run.info.run_id}

    # ── candidates 2 and 3: learned, tuned on validation only ───────────────────
    for name, build in (
        ("logistic_regression",
         lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        ("gradient_boosting",
         lambda: HistGradientBoostingClassifier(max_iter=200, random_state=SEED)),
    ):
        with mlflow.start_run(run_name=name) as run:
            model = build().fit(scaler.transform(Xtr), ytr)

            # The decision threshold is a hyperparameter and is chosen on validation
            # runs. Choosing it on the test set is the most common way a comparison
            # like this quietly becomes meaningless.
            pv = model.predict_proba(scaler.transform(Xva))[:, 1]
            grid = np.linspace(0.05, 0.95, 91)
            thr = float(max(grid, key=lambda t: f1_score(yva, (pv >= t).astype(int),
                                                         zero_division=0)))

            pt = model.predict_proba(scaler.transform(Xte))[:, 1]
            m = score(yte, (pt >= thr).astype(int), scores=pt)
            wrapped = LearnedDetector(model, scaler, thr)
            m.update(latency_ms(wrapped, Wte[:500]))
            m.update(per_fault_recall(yte, (pt >= thr).astype(int), run_te, fault_of))

            mlflow.log_params({**common, "candidate": name,
                               "decision_threshold": round(thr, 3),
                               "threshold_tuned_on": "validation_episodes",
                               "n_features": Xtr.shape[1]})
            mlflow.log_dict(environment(), "environment.json")
            mlflow.log_dict({"features": feat.FEATURE_NAMES}, "feature_contract.json")
            mlflow.log_metrics(m)
            mlflow.pyfunc.log_model(
                name="model", python_model=wrapped,
                code_paths=[str(REPO / "agent"), str(REPO / "experiments")],
                input_example=to_frame(Wte[:2]),
            )
            if name == "logistic_regression":
                log_safe_detector({
                    "format": "argus-detector-v1",
                    "kind": name,
                    "feature_order": feat.FEATURE_NAMES,
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "coef": model.coef_[0].tolist(),
                    "intercept": float(model.intercept_[0]),
                    "threshold": thr,
                })
            results[name] = {**m, "run_id": run.info.run_id}

    # ── the gates ──────────────────────────────────────────────────────────────
    base = results["deterministic"]
    verdict = {}
    for name, m in results.items():
        if name == "deterministic":
            continue
        quality = (m["f1"] >= base["f1"] + MIN_F1_GAIN) and (m["recall"] >= MIN_RECALL)
        latency = m["latency_median_ms"] <= LATENCY_BUDGET_MS
        provenance = not REQUIRE_REAL_EPISODES  # no real episodes exist to satisfy it
        serialization = name == "logistic_regression"
        verdict[name] = {
            "f1_gain_vs_baseline": round(m["f1"] - base["f1"], 4),
            "quality_gate": quality, "latency_gate": latency,
            "provenance_gate": provenance,
            "data_only_artifact_gate": serialization,
            "promotable": quality and latency and provenance and serialization,
        }

    print("\n── results (held-out episodes, scored once) ──")
    for name, m in results.items():
        print(f"  {name:24s} f1={m['f1']:.4f} precision={m['precision']:.4f} "
              f"recall={m['recall']:.4f} median={m['latency_median_ms']:.3f}ms "
              f"p99={m['latency_p99_ms']:.3f}ms")
    print("\n── recall by fault type (held-out) ──")
    for f in ep_mod.FAULTS:
        k = f"recall_{f}"
        if k in results["deterministic"]:
            print(f"  {f:16s} " + "  ".join(
                f"{n}={results[n][k]:.4f}" for n in results))
    print("\n── gates ──")
    for name, v in verdict.items():
        print(f"  {name:24s} gain={v['f1_gain_vs_baseline']:+.4f} "
              f"quality={v['quality_gate']} latency={v['latency_gate']} "
              f"provenance={v['provenance_gate']} "
              f"data_only={v['data_only_artifact_gate']} PROMOTABLE={v['promotable']}")

    # ── registry: version, promote, smoke-test, roll back ──────────────────────
    # v1 is the deterministic detector and is registered unconditionally, so a
    # baseline is always available to roll back to. The best learned candidate is
    # registered as v2 whether or not it passed. Only the deterministic and
    # logistic candidates have data-only activation artifacts. Gradient boosting
    # remains a measured experiment until it has a non-executable export format.
    deployable = "logistic_regression"
    v1 = mlflow.register_model(f"runs:/{base['run_id']}/safe-model", REGISTERED_MODEL)
    v2 = mlflow.register_model(f"runs:/{results[deployable]['run_id']}/safe-model", REGISTERED_MODEL)
    revision = common["git_commit"] + ("+dirty" if common["git_dirty"] else "")
    for v in (v1, v2):
        client.set_model_version_tag(REGISTERED_MODEL, v.version, "dataset_sha256", dhash)
        client.set_model_version_tag(REGISTERED_MODEL, v.version, "code_revision", revision)
        client.set_model_version_tag(REGISTERED_MODEL, v.version, "run_id", v.run_id)
    client.set_model_version_tag(REGISTERED_MODEL, v1.version, "kind", "deterministic")
    client.set_model_version_tag(REGISTERED_MODEL, v2.version, "kind", deployable)

    # Digest each version's files at the moment it is produced, and pin the
    # manifest's own sha256 on the version as a tag. Without this the digest
    # check inside guarded_promote has nothing to compare against and silently
    # does nothing, which is how a wired guard still ends up not guarding.
    for v in (v1, v2):
        model_version = client.get_model_version(REGISTERED_MODEL, v.version)
        guard.record_manifest(
            client, REGISTERED_MODEL, v.version,
            mlflow.artifacts.download_artifacts(v.source),
            {
                "dataset_sha256": dhash,
                "code_revision": revision,
                "run_id": model_version.run_id,
                "kind": model_version.tags["kind"],
            },
            str(MANIFEST_DIR),
        )

    smoke = to_frame(Wte[:200])
    expected = yte[:200]

    # The guarded promotion path. `guarded_promote` runs every artifact check -
    # provenance, digest, data-only format, feature order, staged load, finite
    # weights, prediction dtype/shape/domain - against the STAGED version and
    # only moves the production alias if all of them pass. A rejection raises
    # and leaves the alias untouched, so the prior version keeps serving. This
    # is the wiring the audit found missing: before, the alias moved first and
    # nothing was ever checked.
    def promote(version, label):
        expected_run_id = base["run_id"] if label == "deterministic" else results[label]["run_id"]
        report = guard.guarded_promote(
            client, REGISTERED_MODEL, version, alias="production",
            canary_frame=smoke,
            canary_expected_rows=int(smoke["window_id"].nunique()),
            expected_feature_order=feat.FEATURE_NAMES,
            expected_dataset_sha256=dhash,
            expected_code_revision=revision,
            expected_run_id=expected_run_id,
            expected_kind=label,
            manifest_dir=str(MANIFEST_DIR),
        )
        m = guard.load_registered_model(client, REGISTERED_MODEL, version)
        pred = np.asarray(m.predict(smoke)).ravel()
        agree = float((pred == expected).mean())
        passed = [c["check"] for c in report["checks"]]
        print(f"  promoted v{version} ({label}): {len(passed)} checks passed "
              f"{passed}; serves {len(pred)} windows, agreement {agree:.4f}")
        return agree

    print("\n── promotion and rollback (guarded) ──")
    if verdict[deployable]["promotable"]:
        promote(v2.version, deployable)
    else:
        # v2 did not clear its quality/latency gates, so it is never promoted:
        # the deterministic baseline is what goes to production. Validation is a
        # separate gate from quality - a well-formed artifact that is simply not
        # good enough still must not be activated.
        print(f"  v{v2.version} ({deployable}) did not clear its gates; "
              "not promoting it")
    promote(v1.version, "deterministic")
    current = client.get_model_version_by_alias(REGISTERED_MODEL, "production")
    print(f"  production alias now points at v{current.version} "
          f"({client.get_model_version(REGISTERED_MODEL, current.version).tags.get('kind')})")

    summary = {"dataset": common, "results": results, "gates": verdict,
               "promoted": current.version, "registry": REGISTERED_MODEL}
    (TRACKING_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {TRACKING_DIR / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
