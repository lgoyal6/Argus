"""Bounded public demo endpoint for Argus's source gradient detector."""

from __future__ import annotations

import json
import math
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.detector import detect_grad_explosion


MAX_BODY_BYTES = 2048
MAX_HISTORY_VALUES = 21
MAX_VALUE = 1_000_000


def evaluate_history(text: object) -> dict:
    if not isinstance(text, str):
        raise ValueError("Enter gradient values as text.")
    if len(text) > 512:
        raise ValueError("Input is too long. Use at most 21 values.")
    parts = [part for part in text.replace(",", " ").split() if part]
    if not 1 <= len(parts) <= MAX_HISTORY_VALUES:
        raise ValueError(f"Enter between 1 and {MAX_HISTORY_VALUES} values.")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("Values must be finite numbers between 0 and 1,000,000.") from exc
    if any(not math.isfinite(value) or value < 0 or value > MAX_VALUE for value in values):
        raise ValueError("Values must be finite numbers between 0 and 1,000,000.")

    metrics = [{"step": index + 1, "grad_norm": value} for index, value in enumerate(values)]
    return {"count": len(metrics), "finding": detect_grad_explosion(metrics)}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send(405, {"error": "Use POST."})

    def do_POST(self):
        if self.path != "/api/argus_detect":
            self._send(404, {"error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"error": "Invalid request body."})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send(413 if length > MAX_BODY_BYTES else 400, {"error": "Request body must be between 1 and 2,048 bytes."})
            return
        try:
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or set(body) != {"history"}:
                raise ValueError("Send only a history field.")
            result = evaluate_history(body["history"])
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send(400, {"error": "Request body must be valid JSON."})
            return
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
            return
        self._send(200, result)

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)
