import json
import yaml
from pathlib import Path

# Flat in both layouts; see the note in agent/detector.py.
import obs

try:  # imported as a package by the tests, flat by agent/loop.py and the image
    from agent.sandbox import (
        SandboxError,
        resolve_within,
        validate_patches,
        validate_tool_args,
    )
except ImportError:  # pragma: no cover - exercised only by the flat layout
    from sandbox import (
        SandboxError,
        resolve_within,
        validate_patches,
        validate_tool_args,
    )

# ── tool definitions (passed to Anthropic API) ─────────────────────────────────
TOOLS = [
    {
        "name": "read_config",
        "description": "Read the current training config.yaml file. Use this to understand current hyperparameters before deciding on a fix.",
        "input_schema": {
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Path to the config.yaml file"
                }
            },
            "required": ["config_path"]
        }
    },
    {
        "name": "read_metrics",
        "description": "Read the last N lines of the metrics file to understand recent training behavior.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metrics_file": {
                    "type": "string",
                    "description": "Path to the metrics.jsonl file"
                },
                "last_n": {
                    "type": "integer",
                    "description": "Number of recent metric entries to read"
                }
            },
            "required": ["metrics_file", "last_n"]
        }
    },
    {
        "name": "patch_config",
        "description": "Apply a targeted fix to config.yaml by updating specific hyperparameter values. Only change what is necessary to fix the detected anomaly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Path to the config.yaml file"
                },
                "patches": {
                    "type": "object",
                    "description": "Dict of dot-notation keys and new values. e.g. {'training.learning_rate': 0.0001, 'training.gradient_clip': 0.5}"
                }
            },
            "required": ["config_path", "patches"]
        }
    },
    {
        "name": "rerun_training",
        "description": "Rerun the training job with the current config. Call this after applying a patch to verify if the fix worked.",
        "input_schema": {
            "type": "object",
            "properties": {
                "training_dir": {
                    "type": "string",
                    "description": "Path to the training_job directory"
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Number of steps to run before stopping to check if the fix worked. Keep this small (50-100) for fast verification."
                }
            },
            "required": ["training_dir", "max_steps"]
        }
    }
]


# ── tool implementations ───────────────────────────────────────────────────────
def read_config(config_path):
    path = resolve_within(config_path)
    if not path.exists():
        return {"error": f"config not found at {config_path}"}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_metrics(metrics_file, last_n=20):
    path = resolve_within(metrics_file)
    try:
        last_n = int(last_n)
    except (TypeError, ValueError):
        raise SandboxError("last_n must be an integer")
    last_n = max(1, min(last_n, 1000))  # bound the read the model can ask for
    if not path.exists():
        return {"error": f"metrics file not found at {metrics_file}"}
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    recent = lines[-last_n:]
    return [json.loads(l) for l in recent]


def patch_config(config_path, patches):
    path = resolve_within(config_path)
    validate_patches(patches)
    if not path.exists():
        return {"error": f"config not found at {config_path}"}

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    applied = {}
    for key, value in patches.items():
        # dot notation: "training.learning_rate" -> cfg["training"]["learning_rate"]
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            if part not in node:
                return {"error": f"key {part} not found in config"}
            node = node[part]
        old_value = node.get(parts[-1], "not found")
        node[parts[-1]] = value
        applied[key] = {"old": old_value, "new": value}

    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return {"status": "patched", "changes": applied}


# The training server is the agent's one synchronous dependency, and it is the one that
# stalls: the handler answers before its subprocess does anything, and this call's
# timeout is much shorter than a training run. Classifying the failure at the call site
# is what turns "the recovery did not work" into "the training server did not answer in
# 10s", which is a different problem with a different fix.
TRAINING_SERVER = "http://training:8001/rerun"
REQUEST_TIMEOUT_S = 10


def _classify(exc):
    """Which kind of dependency failure this was, from a closed set.

    A closed set is what lets the outcome be a metric label at all. The exception's
    own message is not a label and not an attribute: it carries a URL, and on some
    clients credentials, so it goes in the log body where obs.scrub_text sees it.
    """
    name = type(exc).__name__
    if "Timeout" in name:
        return "timeout"
    if "ConnectionError" in name or "ConnectTimeout" in name:
        return "connection"
    if "HTTPError" in name or "Status" in name:
        return "http_error"
    return "connection"


def rerun_training(training_dir, max_steps=50, attempt_id=None):
    import requests

    resolve_within(training_dir)
    try:
        max_steps = int(max_steps)
    except (TypeError, ValueError):
        raise SandboxError("max_steps must be an integer")
    if not (1 <= max_steps <= 10000):
        raise SandboxError(f"max_steps={max_steps} outside the permitted range [1, 10000]")
    with obs.bind(attempt_id=attempt_id):
        with obs.span("recovery.rerun_request", dependency="training_server",
                      max_steps=max_steps, timeout_s=REQUEST_TIMEOUT_S) as span:
            try:
                response = requests.post(
                    TRAINING_SERVER,
                    # The attempt id is the idempotency key. This request's timeout is far
                    # shorter than a training run, so a retry after a lost response is normal;
                    # without the key it would start a second trainer appending to the same
                    # metrics file, and two interleaved runs make the stream unreadable.
                    json={"max_steps": max_steps, "attempt_id": attempt_id},
                    timeout=REQUEST_TIMEOUT_S
                )
            except Exception as e:
                reason = _classify(e)
                # The exception is swallowed and returned as a tool result, so the span
                # has to be told; otherwise its end record claims outcome="ok".
                span.failed(reason)
                obs.incr("argus_dependency_calls_total",
                         dependency="training_server", outcome=reason)
                obs.log("recovery.rerun_failed", level="error",
                        dependency="training_server", reason=reason,
                        error_type=type(e).__name__, detail=str(e),
                        timeout_s=REQUEST_TIMEOUT_S)
                return {"status": "error", "error": str(e)}
            obs.incr("argus_dependency_calls_total",
                     dependency="training_server", outcome="ok")
            # "requested", never "started". This is an HTTP response from a handler that
            # returns before its subprocess has done anything, so the most it can honestly
            # attest to is that the request was accepted. Whether a trainer ran, and
            # whether it fixed anything, is settled by agent/attempt.py against the
            # metrics stream - not here.
            return {"status": "requested", "response": response.json()}

# ── tool dispatcher ────────────────────────────────────────────────────────────
_HANDLERS = {
    "read_config": read_config,
    "read_metrics": read_metrics,
    "patch_config": patch_config,
    "rerun_training": rerun_training,
}


# Arguments the caller injects, which the model may not supply. attempt_id is an
# idempotency key: letting model output choose it would let a retry be laundered into
# a fresh launch, which is the thing the key exists to prevent.
_INJECTED = {"rerun_training": "attempt_id"}


def run_tool(tool_name, tool_input, attempt_id=None):
    """Dispatch one tool call, refusing anything outside the agent's boundary.

    A refusal is returned as a normal tool result rather than raised: the model
    should see "you may not do that" and pick a different action, not crash the
    recovery loop. The refusal text names the boundary so the reason is auditable
    after the fact.
    """
    try:
        # Validated before injection, so the allowlist still describes exactly what
        # the model is permitted to send.
        validate_tool_args(tool_name, tool_input)
        kwargs = dict(tool_input)
        if _INJECTED.get(tool_name) == "attempt_id":
            kwargs["attempt_id"] = attempt_id
        return _HANDLERS[tool_name](**kwargs)
    except SandboxError as e:
        return {"error": f"refused: {e}", "refused": True}
