"""Check local_ollama.py transport bounds and strict action decoding.

Fake responses cover schema controls; a loopback server exercises redirects.
A local endpoint must not redirect the client to an external destination.
"""
import json

import pytest

from agent.local_ollama import LocalModelError, OllamaRecoveryModel


class Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return self.body[:limit]


def fake_opener_for(inner, seen, *, done=True, done_reason="stop"):
    def opener(request, timeout):
        seen.append((request, timeout, json.loads(request.data)))
        return Response(json.dumps({"response": json.dumps(inner), "done": done,
                                    "done_reason": done_reason,
                                    "prompt_eval_count": 10}).encode())
    return opener


def test_local_adapter_posts_bounded_json_to_loopback_and_parses_action():
    seen = []
    model = OllamaRecoveryModel(opener=fake_opener_for(
        {"thought": "inspect", "action": {"kind": "read_metrics", "input": {"last_n": 5}}}, seen))
    state = {"anomaly": {"type": "loss_spike"}, "round": 0}
    assert model.next_action(state)["action"]["kind"] == "read_metrics"
    request, timeout, payload = seen[0]
    assert request.full_url == "http://127.0.0.1:11434/api/generate"
    assert timeout == 30
    assert payload["stream"] is False
    schema = payload["format"]
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert set(schema["required"]) == {"thought", "action"}
    action_schema = schema["properties"]["action"]
    assert action_schema["properties"]["input"]["type"] == "object"
    assert set(action_schema["properties"]["kind"]["enum"]) == {
        "read_config", "read_metrics", "patch_config", "rerun_training", "finish"}
    assert json.loads(payload["prompt"])["action_schema"] == schema
    assert payload["options"] == {"temperature": 0, "num_predict": 256}
    assert model.last_response_metadata["prompt_eval_count"] == 10


@pytest.mark.parametrize("url", ["https://127.0.0.1:11434", "http://example.com:11434",
                                  "http://127.0.0.1:11434/path", "http://u:p@127.0.0.1:11434"])
def test_local_adapter_rejects_non_loopback_or_ambiguous_endpoint(url):
    with pytest.raises(LocalModelError):
        OllamaRecoveryModel(base_url=url)


@pytest.mark.parametrize("inner", [
    {"thought": "missing action"},
    {"thought": "bad kind", "action": {"kind": "shell", "input": {}}},
    {"thought": "bad input", "action": {"kind": "finish", "input": []}},
    {"thought": "stringified input", "action": {"kind": "read_metrics", "input": "{'last_n': 3}"}},
    {"thought": 4, "action": {"kind": "finish", "input": {}}},
    {"thought": "extra", "action": {"kind": "finish", "input": {}}, "extra": 1},
])
def test_local_adapter_fails_closed_on_invalid_action_schema(inner):
    model = OllamaRecoveryModel(opener=fake_opener_for(inner, []))
    with pytest.raises(LocalModelError):
        model.next_action({})


def test_local_adapter_rejects_invalid_limits_and_non_json_response():
    with pytest.raises(LocalModelError):
        OllamaRecoveryModel(timeout_s=31)
    with pytest.raises(LocalModelError):
        OllamaRecoveryModel(max_tokens=257)
    model = OllamaRecoveryModel(opener=lambda *args, **kwargs: Response(b"not-json"))
    with pytest.raises(LocalModelError):
        model.next_action({})


@pytest.mark.parametrize("done,done_reason", [(False, "stop"), (True, "length"), (True, None)])
def test_local_adapter_rejects_incomplete_or_token_limited_generation(done, done_reason):
    model = OllamaRecoveryModel(opener=fake_opener_for(
        {"thought": "finish", "action": {"kind": "finish", "input": {}}}, [],
        done=done, done_reason=done_reason))
    with pytest.raises(LocalModelError, match="generation did not complete"):
        model.next_action({})


def test_local_adapter_retains_raw_model_response_when_action_schema_fails():
    model = OllamaRecoveryModel(opener=fake_opener_for(
        {"thought": "bad", "action": {"kind": "shell", "input": {}}}, []))
    with pytest.raises(LocalModelError):
        model.next_action({})
    assert json.loads(model.last_raw_response)["action"]["kind"] == "shell"
    assert model.last_response_metadata["done"] is True


def test_loopback_endpoint_cannot_redirect_off_host(monkeypatch):
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import urllib.request

    requests = []

    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append("POST")
            self.send_response(302)
            self.send_header("Location", "http://remote.invalid/api/generate")
            self.end_headers()

        def log_message(self, *args):
            pass

    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setattr(urllib.request, "_opener", None)
    real_connect = socket.create_connection
    blocked_connections = []

    def loopback_only(address, *args, **kwargs):
        if address[0] != "127.0.0.1":
            blocked_connections.append(address[0])
            raise OSError("test forbids non-loopback connection")
        return real_connect(address, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", loopback_only)
    server = HTTPServer(("127.0.0.1", 0), Redirect)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        model = OllamaRecoveryModel(base_url=f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(LocalModelError):
            model.next_action({})
        assert requests == ["POST"], "test did not reach the loopback redirect endpoint"
        assert blocked_connections == [], "adapter tried to follow a nonlocal redirect"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


CONTRACT = {"read_metrics": {"required": {"metrics_file": "string", "last_n": "integer"}},
            "finish": {"required": {}}}


@pytest.mark.parametrize("action", [
    {"kind": "read_metrics", "input": {"metrics_file": "/synthetic/metrics.jsonl", "last_n": "5"}},
    {"kind": "read_metrics", "input": {"metrics_file": "/synthetic/metrics.jsonl", "last_n": True}},
    {"kind": "read_metrics", "input": {"metrics_file": "/synthetic/metrics.jsonl", "last_n": 1.5}},
    {"kind": "read_metrics", "input": {"last_n": 5}},
    {"kind": "read_metrics", "input": {"metrics_file": "/synthetic/metrics.jsonl", "last_n": 5, "extra": 1}},
    {"kind": "rerun_training", "input": {}},
])
def test_declared_action_contract_rejects_wrong_argument_types_fields_or_actions(action):
    model = OllamaRecoveryModel(opener=fake_opener_for({"thought": "inspect", "action": action}, []))
    with pytest.raises(LocalModelError, match="contract"):
        model.next_action({"action_contract": CONTRACT})


def test_declared_contract_constrains_structured_generation_and_accepts_typed_arguments():
    seen = []
    action = {"kind": "read_metrics", "input": {"metrics_file": "/synthetic/metrics.jsonl", "last_n": 5}}
    model = OllamaRecoveryModel(opener=fake_opener_for({"thought": "inspect", "action": action}, seen))
    assert model.next_action({"action_contract": CONTRACT})["action"] == action
    payload = seen[0][2]
    assert json.loads(payload["prompt"])["allowed_actions"] == ["finish", "read_metrics"]
    variants = payload["format"]["properties"]["action"]["anyOf"]
    metrics = next(v for v in variants if v["properties"]["kind"]["enum"] == ["read_metrics"])
    inputs = metrics["properties"]["input"]
    assert inputs["properties"]["last_n"] == {"type": "integer"}
    assert inputs["additionalProperties"] is False
    assert set(inputs["required"]) == {"metrics_file", "last_n"}


@pytest.mark.parametrize("contract", [
    {"read_metrics": {"required": {"last_n": "timestamp"}}},
    {"read_metrics": {"required": {"last_n": 1}}},
    {"finish": {"required": {}}, 1: {"required": {}}},
    ["finish"],
])
def test_malformed_contract_is_rejected_before_any_model_call(contract):
    seen = []
    model = OllamaRecoveryModel(opener=fake_opener_for(
        {"thought": "finish", "action": {"kind": "finish", "input": {}}}, seen))
    with pytest.raises(LocalModelError, match="contract"):
        model.next_action({"action_contract": contract})
    assert seen == []
