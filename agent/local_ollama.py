"""Call loopback Ollama for the checkpointed recovery graph.

Used by langgraph_recovery.py and tests/test_local_ollama.py.
Reject redirects and proxies; validated model actions still pass the graph gate.
"""

from __future__ import annotations

import json
import math
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class LocalModelError(ValueError):
    """The local model endpoint or response violated the adapter contract."""


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LocalModelError("local Ollama endpoint must not redirect")


_ACTIONS = {"read_config", "read_metrics", "patch_config", "rerun_training", "finish"}
_ACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["thought", "action"],
    "properties": {
        "thought": {"type": "string", "maxLength": 1000},
        "action": {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "input"],
            "properties": {
                "kind": {"type": "string", "enum": sorted(_ACTIONS)},
                "input": {"type": "object"},
            },
        },
    },
}

_INPUT_TYPES = {"string": (str,), "integer": (int,), "number": (int, float),
                "boolean": (bool,), "object": (dict,), "array": (list,),
                "null": (type(None),)}


def _schema_for_contract(contract):
    """Use the caller's declared argument types for generation and decoding."""
    if contract is None or contract == {}:
        return _ACTION_SCHEMA, {}
    if not isinstance(contract, dict) or any(not isinstance(key, str) for key in contract):
        raise LocalModelError("action contract must be an object")
    inputs, variants = {}, []
    for kind, entry in sorted(contract.items()):
        if kind not in _ACTIONS or not isinstance(entry, dict) or set(entry) != {"required"}:
            raise LocalModelError("action contract contains an unsupported declaration")
        fields = entry["required"]
        if not isinstance(fields, dict) or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or value not in _INPUT_TYPES
            for key, value in fields.items()
        ):
            raise LocalModelError("action contract requires named fields with JSON types")
        inputs[kind] = dict(fields)
        variants.append({
            "type": "object", "additionalProperties": False, "required": ["kind", "input"],
            "properties": {
                "kind": {"type": "string", "enum": [kind]},
                "input": {"type": "object", "additionalProperties": False,
                          "required": sorted(fields),
                          "properties": {key: {"type": value} for key, value in sorted(fields.items())}},
            },
        })
    return {**_ACTION_SCHEMA, "properties": {
        "thought": _ACTION_SCHEMA["properties"]["thought"],
        "action": {"anyOf": variants},
    }}, inputs


class OllamaRecoveryModel:
    """A bounded JSON-action client restricted to an HTTP loopback endpoint."""

    def __init__(self, model: str = "llama3.2:3b", *, base_url: str = "http://127.0.0.1:11434",
                 timeout_s: float = 30, max_tokens: int = 256, opener=None):
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise LocalModelError("Ollama endpoint must be plain HTTP on loopback")
        if parsed.path not in {"", "/"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise LocalModelError("Ollama endpoint must not contain credentials or query data")
        if not model or len(model) > 128 or any(ch.isspace() for ch in model):
            raise LocalModelError("invalid model tag")
        if not 0 < timeout_s <= 30:
            raise LocalModelError("timeout_s must be in (0, 30]")
        if not 1 <= max_tokens <= 256:
            raise LocalModelError("max_tokens must be in [1, 256]")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        # Loopback traffic stays direct even when proxy environment variables exist.
        self._opener = opener if opener is not None else build_opener(
            ProxyHandler({}), _NoRedirects()
        ).open
        self.last_response_metadata = None
        self.last_raw_response = None

    def next_action(self, state):
        """Return one schema-checked action; tool safety remains with the graph gate."""
        schema, inputs = _schema_for_contract(state.get("action_contract"))
        payload = {
            "model": self.model,
            "system": (
                "Choose exactly one next recovery action from the listed allowed actions. "
                "Return a JSON object with exactly keys thought and action. action must "
                "have exactly keys kind and input. Do not claim a tool was run. Follow "
                "the provided recovery_policy and action_contract."
            ),
            "prompt": json.dumps({
                "anomaly": state.get("anomaly", {}),
                "resources": state.get("resources", {}),
                "recovery_policy": state.get("recovery_policy", ""),
                "action_contract": state.get("action_contract", {}),
                "last_tool_result": state.get("tool_result"),
                "history": state.get("history", []),
                "round": state.get("round", 0),
                "allowed_actions": sorted(inputs or _ACTIONS),
                "action_schema": schema,
            }, ensure_ascii=False, separators=(",", ":")),
            "format": schema,
            "stream": False,
            "options": {"temperature": 0, "num_predict": self.max_tokens},
        }
        request = Request(
            self.base_url + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout_s) as response:
                body = response.read(1_000_001)
        except (OSError, URLError, TimeoutError) as exc:
            raise LocalModelError(f"local Ollama request failed: {type(exc).__name__}") from exc
        if len(body) > 1_000_000:
            raise LocalModelError("Ollama response exceeded 1 MB")
        try:
            envelope = json.loads(body)
            generated = envelope["response"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise LocalModelError("Ollama returned invalid JSON") from exc
        if envelope.get("done") is not True or envelope.get("done_reason") != "stop":
            raise LocalModelError("Ollama generation did not complete with a stop reason")
        if not isinstance(generated, str) or len(generated) > 100_000:
            raise LocalModelError("Ollama generated response is invalid or too large")
        self.last_raw_response = generated
        self.last_response_metadata = {
            key: envelope.get(key) for key in (
                "model", "created_at", "done", "done_reason", "total_duration",
                "load_duration", "prompt_eval_count", "prompt_eval_duration",
                "eval_count", "eval_duration",
            ) if key in envelope
        }
        try:
            result = json.loads(generated)
        except json.JSONDecodeError as exc:
            raise LocalModelError("Ollama generated invalid action JSON") from exc
        if not isinstance(result, dict) or set(result) != {"thought", "action"}:
            raise LocalModelError("model response must contain exactly thought and action")
        action = result["action"]
        if not isinstance(result["thought"], str) or len(result["thought"]) > 1000:
            raise LocalModelError("thought must be a string of at most 1000 characters")
        if not isinstance(action, dict) or set(action) != {"kind", "input"}:
            raise LocalModelError("action must contain exactly kind and input")
        if not isinstance(action["kind"], str) or action["kind"] not in _ACTIONS or not isinstance(action["input"], dict):
            raise LocalModelError("action kind or input is invalid")
        if action["kind"] == "finish" and action["input"]:
            raise LocalModelError("finish action input must be empty")
        if inputs:
            fields = inputs.get(action["kind"])
            if fields is None or set(action["input"]) != set(fields):
                raise LocalModelError("action does not match declared contract fields")
            for key, typename in fields.items():
                value = action["input"][key]
                if type(value) not in _INPUT_TYPES[typename] or (
                    typename == "number" and not math.isfinite(value)
                ):
                    raise LocalModelError("action argument does not match declared contract type")
        return result
