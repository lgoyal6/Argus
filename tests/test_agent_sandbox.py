"""The recovery agent's tool boundary.

Threat model: a training run's metrics are text produced by the training job, and
the agent feeds them to a model that then chooses tool calls. Anything able to
influence that log can attempt to steer the model. These tests are written as the
attacks, not as unit assertions about helpers - each one is a thing the tools would
have done before, reachable from log content.
"""

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import tools
from agent.sandbox import SandboxError, resolve_within, validate_patches


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "training_job"
    ws.mkdir()
    cfg = {
        "training": {"learning_rate": 0.001, "gradient_clip": 5.0, "weight_decay": 0.001,
                     "epochs": 20, "optimizer": "adam"},
        "data": {"batch_size": 64, "dataset": "cifar10", "num_workers": 2},
        "checkpointing": {"checkpoint_dir": "checkpoints/", "save_every": 5},
        "monitoring": {"metrics_file": "metrics/metrics.jsonl", "emit_every_n_steps": 10},
    }
    (ws / "config.yaml").write_text(yaml.dump(cfg))
    monkeypatch.setenv("ARGUS_WORKSPACE", str(ws))
    return ws


# ── path boundary ─────────────────────────────────────────────────────────────

def test_reading_outside_the_workspace_is_refused(workspace):
    out = tools.run_tool("read_config", {"config_path": "/etc/passwd"})
    assert out.get("refused"), f"read /etc/passwd was allowed: {out}"


def test_traversal_out_of_the_workspace_is_refused(workspace):
    out = tools.run_tool("read_config", {"config_path": str(workspace / ".." / ".." / "etc" / "passwd")})
    assert out.get("refused"), f"traversal escaped the workspace: {out}"


def test_a_symlink_pointing_outside_is_refused(workspace, tmp_path):
    secret = tmp_path / "secret.yaml"
    secret.write_text("token: hunter2\n")
    link = workspace / "innocent.yaml"
    link.symlink_to(secret)
    out = tools.run_tool("read_config", {"config_path": str(link)})
    assert out.get("refused"), f"symlink escaped the workspace: {out}"


def test_patch_config_cannot_write_outside_the_workspace(workspace, tmp_path):
    """The serious one: patch_config opens for WRITE, so an unbounded path here is
    an arbitrary-file-write primitive reachable from log content."""
    target = tmp_path / "outside.yaml"
    target.write_text("original: true\n")
    out = tools.run_tool("patch_config", {
        "config_path": str(target),
        "patches": {"training.learning_rate": 0.1},
    })
    assert out.get("refused"), f"wrote outside the workspace: {out}"
    assert target.read_text() == "original: true\n", "the file outside was modified"


# ── config-key boundary ───────────────────────────────────────────────────────

def test_only_allowlisted_hyperparameters_are_editable(workspace):
    for key, value in [
        ("data.dataset", "attacker-controlled"),
        ("checkpointing.checkpoint_dir", "/tmp/exfil"),
        ("monitoring.metrics_file", "/tmp/elsewhere.jsonl"),
        ("training.optimizer", "sgd"),
    ]:
        out = tools.run_tool("patch_config", {
            "config_path": str(workspace / "config.yaml"),
            "patches": {key: value},
        })
        assert out.get("refused"), f"{key} was editable: {out}"

    cfg = yaml.safe_load((workspace / "config.yaml").read_text())
    assert cfg["data"]["dataset"] == "cifar10"
    assert cfg["checkpointing"]["checkpoint_dir"] == "checkpoints/"
    assert cfg["monitoring"]["metrics_file"] == "metrics/metrics.jsonl"


def test_values_outside_their_range_are_refused(workspace):
    for key, value in [
        ("training.learning_rate", 1e9),
        ("training.learning_rate", -1.0),
        ("data.batch_size", 10**9),
        ("data.batch_size", 0),
        ("training.epochs", 10**6),
    ]:
        with pytest.raises(SandboxError):
            validate_patches({key: value})


def test_wrong_types_are_refused(workspace):
    for key, value in [
        ("training.learning_rate", "0.1"),
        ("data.batch_size", 12.5),
        ("data.batch_size", True),  # bool is an int subclass; still not a batch size
        ("training.epochs", None),
    ]:
        with pytest.raises(SandboxError):
            validate_patches({key: value})


# ── the tools still work ──────────────────────────────────────────────────────

def test_a_legitimate_recovery_still_succeeds(workspace):
    """The boundary must not break the product. This is the actual repair an
    agent makes for a loss spike: lower the learning rate, tighten the clip."""
    out = tools.run_tool("patch_config", {
        "config_path": str(workspace / "config.yaml"),
        "patches": {"training.learning_rate": 0.0001, "training.gradient_clip": 0.5},
    })
    assert not out.get("refused"), out
    assert out["status"] == "patched"
    cfg = yaml.safe_load((workspace / "config.yaml").read_text())
    assert cfg["training"]["learning_rate"] == 0.0001
    assert cfg["training"]["gradient_clip"] == 0.5
    # and untouched keys survive
    assert cfg["training"]["optimizer"] == "adam"
    assert cfg["data"]["dataset"] == "cifar10"


def test_reading_a_config_inside_the_workspace_works(workspace):
    out = tools.run_tool("read_config", {"config_path": str(workspace / "config.yaml")})
    assert not out.get("refused"), out


# ── argument surface ──────────────────────────────────────────────────────────

def test_unexpected_tool_arguments_are_refused(workspace):
    out = tools.run_tool("read_config", {"config_path": str(workspace / "config.yaml"),
                                         "extra": "surprise"})
    assert out.get("refused"), f"unknown kwargs accepted: {out}"


def test_an_unknown_tool_is_refused(workspace):
    assert tools.run_tool("exfiltrate", {}).get("refused")


def test_refusals_are_returned_not_raised(workspace):
    """A refusal must be a tool result the model can react to, not an exception
    that kills the recovery loop."""
    out = tools.run_tool("read_config", {"config_path": "/etc/hosts"})
    assert isinstance(out, dict) and out.get("refused") and "refused" in out["error"]
