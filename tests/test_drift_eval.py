"""Invariants for experiments.drift_eval (C25).

The catch/miss pair is itself a negative control: a monitor that fired on both
would be proving nothing, and one that fired on neither would be inert. We
assert it fires on the covariate shift, stays silent on the clean baseline
(false-alarm control) AND on the marginal-preserving shift, and that the silent
shift really did degrade recall.

    python -m pytest tests/test_drift_eval.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# The C25 harness fits real models, so it needs the model stack from
# experiments/requirements.txt rather than requirements-dev.txt. The `model-quality`
# CI job installs it and runs this file, so these skips are a local convenience and
# not a way for the suite to stop running.
pytest.importorskip("sklearn", reason="pip install -r experiments/requirements.txt")
pytest.importorskip("scipy", reason="pip install -r experiments/requirements.txt")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import drift_eval as d  # noqa: E402


@pytest.fixture(scope="module")
def report():
    return d.main()


def test_calibration_error_is_a_real_fraction(report):
    for cand in ("logistic_regression", "gradient_boosting"):
        c = report["calibration"][cand]
        assert 0.0 <= c["ece"] <= 1.0
        assert 0.0 <= c["brier"] <= 1.0
        assert len(c["mean_predicted"]) == len(c["fraction_positive"])


def test_error_slices_cover_every_fault(report):
    slices = report["error_slices_logreg"]
    for fault in ("loss_spike", "grad_explosion", "val_plateau", "overfitting"):
        assert fault in slices and slices[fault]["n"] > 0


def test_clean_baseline_does_not_false_alarm(report):
    base = report["shift"]["clean_baseline"]
    assert base["fires_false_alarm"] is False
    assert base["max_psi"] < d.PSI_ALARM


def test_covariate_shift_is_caught(report):
    cov = report["shift"]["covariate_grad_norm_x3"]
    assert cov["caught"] is True
    assert cov["max_psi"] >= d.PSI_ALARM


def test_marginal_preserving_shift_is_missed_but_degrades(report):
    m = report["shift"]["joint_decorrelation"]
    # the monitor stays silent...
    assert m["caught"] is False
    assert m["max_psi"] < d.PSI_ALARM
    # ...and its PSI equals the clean baseline's, because the marginals are
    # identical by construction.
    assert abs(m["max_psi"] - report["shift"]["clean_baseline"]["max_psi"]) < 1e-6
    # ...while recall really did fall: a real regression the monitor missed.
    assert m["recall_shifted"] < m["recall_clean"] - 0.05


def test_the_numbers_reach_mlflow_as_metrics(report):
    """The plan names MLflow metrics, so the report must be readable back out."""
    mlflow = pytest.importorskip("mlflow")
    run_id = report.get("mlflow_run_id")
    assert run_id, "drift_eval did not log to MLflow"
    mlflow.set_tracking_uri(f"sqlite:///{d.TRACKING_DIR}/mlflow.db")
    metrics = mlflow.tracking.MlflowClient().get_run(run_id).data.metrics
    # control: the values in MLflow are the values in the report, not placeholders
    assert metrics["calibration.logistic_regression.ece"] == pytest.approx(
        report["calibration"]["logistic_regression"]["ece"])
    assert metrics["shift.covariate.caught"] == 1.0
    assert metrics["shift.decorrelation.caught"] == 0.0


def test_module_declares_no_automatic_retraining(report):
    assert "NONE" in report["retraining"]
    assert "retrain" in report["retraining_policy"].lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
