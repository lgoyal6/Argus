import json
import yaml
import subprocess
from pathlib import Path


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
    path = Path(config_path)
    if not path.exists():
        return {"error": f"config not found at {config_path}"}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_metrics(metrics_file, last_n=20):
    path = Path(metrics_file)
    if not path.exists():
        return {"error": f"metrics file not found at {metrics_file}"}
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    recent = lines[-last_n:]
    return [json.loads(l) for l in recent]


def patch_config(config_path, patches):
    path = Path(config_path)
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


def rerun_training(training_dir, max_steps=50):
    train_script = Path(training_dir) / "train.py"
    if not train_script.exists():
        return {"error": f"train.py not found in {training_dir}"}

    # clear old metrics so the agent gets a fresh read after rerun
    metrics_file = Path(training_dir) / "metrics" / "metrics.jsonl"
    if metrics_file.exists():
        metrics_file.unlink()

    try:
        venv_python = str(Path(training_dir) / "venv/bin/python3") if (Path(training_dir) / "venv/bin/python3").exists() else "python3"
        result = subprocess.run(
            [venv_python, str(train_script), str(max_steps)],
            cwd=training_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
        return {
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout[-3000:],
            "stderr": result.stderr[-1000:] if result.stderr else ""
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "training exceeded 300 second limit"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ── tool dispatcher ────────────────────────────────────────────────────────────
def run_tool(tool_name, tool_input):
    if tool_name == "read_config":
        return read_config(**tool_input)
    elif tool_name == "read_metrics":
        return read_metrics(**tool_input)
    elif tool_name == "patch_config":
        return patch_config(**tool_input)
    elif tool_name == "rerun_training":
        return rerun_training(**tool_input)
    else:
        return {"error": f"unknown tool: {tool_name}"}
