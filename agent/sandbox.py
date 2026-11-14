"""Boundaries for the recovery agent's tools.

The agent reads a training run's metrics and then chooses tool calls. Those metrics
are not trusted input: they are text written by a training job, and anything that can
influence that job's logs can attempt to steer the model. So the tools must be safe
even when the model has been talked into asking for something unreasonable.

Two boundaries, because the tools had neither:

  * Paths. read_config / read_metrics / patch_config took any path the model named.
    patch_config opens for WRITE, so that was an arbitrary-file-write primitive
    reachable from log content. Every path is now resolved (following symlinks) and
    must land inside the workspace root.
  * Config keys. patch_config would set any dot-path to any value, including
    swapping the dataset or the checkpoint directory. Only the hyperparameters a
    recovery is allowed to touch are editable, and each has a type and a range.

Both are allowlists: an unknown path or key is refused, rather than a denylist of
known-bad ones that a new tool would quietly bypass.
"""

from __future__ import annotations

import os
from numbers import Real
from pathlib import Path


class SandboxError(Exception):
    """A tool call asked for something outside its permitted boundary."""


def workspace_root() -> Path:
    """Directory every tool path must resolve inside.

    ARGUS_WORKSPACE overrides it; the default is the training job beside this repo,
    which is the only tree a recovery has any business editing.
    """
    env = os.environ.get("ARGUS_WORKSPACE")
    root = Path(env) if env else Path(__file__).resolve().parents[1] / "training_job"
    return root.resolve()


def resolve_within(candidate, root=None) -> Path:
    """Resolve `candidate` and require it to be inside `root`.

    resolve() is what makes this hold against symlinks and against ".." segments:
    a symlink inside the workspace pointing at /etc is resolved before the check, so
    it fails like any other outside path.
    """
    root = (root or workspace_root()).resolve()
    try:
        resolved = Path(candidate).resolve()
    except (OSError, RuntimeError) as exc:  # RuntimeError: symlink loop
        raise SandboxError(f"path could not be resolved: {candidate}") from exc
    if resolved != root and root not in resolved.parents:
        raise SandboxError(
            f"path escapes the agent workspace: {candidate} -> {resolved} (root {root})"
        )
    return resolved


# Editable hyperparameters: key -> (python type, minimum, maximum).
#
# Scoped to what a recovery actually needs. Notably absent: data.dataset,
# checkpointing.checkpoint_dir and monitoring.metrics_file - changing those is not
# fixing a training run, it is redirecting where the job reads and writes.
ALLOWED_CONFIG_KEYS = {
    "training.learning_rate": (Real, 1e-8, 1.0),
    "training.gradient_clip": (Real, 1e-4, 1e4),
    "training.weight_decay": (Real, 0.0, 1.0),
    "training.epochs": (int, 1, 1000),
    "data.batch_size": (int, 1, 4096),
    "data.num_workers": (int, 0, 64),
    "monitoring.emit_every_n_steps": (int, 1, 10000),
    "checkpointing.save_every": (int, 1, 10000),
}


def validate_patches(patches):
    """Return patches unchanged, or raise SandboxError naming the first violation."""
    if not isinstance(patches, dict) or not patches:
        raise SandboxError("patches must be a non-empty object")
    for key, value in patches.items():
        if key not in ALLOWED_CONFIG_KEYS:
            raise SandboxError(
                f"config key not editable by the agent: {key!r} "
                f"(allowed: {', '.join(sorted(ALLOWED_CONFIG_KEYS))})"
            )
        expected, low, high = ALLOWED_CONFIG_KEYS[key]
        # bool is a subclass of int; a boolean learning rate is a bug, not a value.
        if isinstance(value, bool) or not isinstance(value, expected):
            raise SandboxError(
                f"{key} must be {getattr(expected, '__name__', expected)}, got {type(value).__name__}"
            )
        if expected is int and isinstance(value, float):
            raise SandboxError(f"{key} must be an integer, got {value!r}")
        if not (low <= value <= high):
            raise SandboxError(f"{key}={value!r} is outside the permitted range [{low}, {high}]")
    return patches


# Tool calls arrive as **kwargs straight from the model, so an unexpected key would
# be a TypeError rather than a clear refusal.
ALLOWED_TOOL_ARGS = {
    "read_config": {"config_path"},
    "read_metrics": {"metrics_file", "last_n"},
    "patch_config": {"config_path", "patches"},
    "rerun_training": {"training_dir", "max_steps"},
}


def validate_tool_args(tool_name, tool_input):
    if tool_name not in ALLOWED_TOOL_ARGS:
        raise SandboxError(f"unknown tool: {tool_name}")
    if not isinstance(tool_input, dict):
        raise SandboxError("tool input must be an object")
    extra = set(tool_input) - ALLOWED_TOOL_ARGS[tool_name]
    if extra:
        raise SandboxError(f"unexpected arguments for {tool_name}: {sorted(extra)}")
    return tool_input
