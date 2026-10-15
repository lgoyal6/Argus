"""An unobserved server or model cannot acquire fabricated provenance."""
import importlib.util
import json
from pathlib import Path
import sys


def test_no_call_report_keeps_selected_model_and_marks_provenance_unverified(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "experiments/run_local_recovery.py"
    spec = importlib.util.spec_from_file_location("local_recovery_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"protocol_id": "provenance-control", "scenarios": []}))
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [str(path), "--protocol", str(protocol), "--output", str(output),
                                      "--model", "unobserved-model:control"])
    module.main()
    report = json.loads(output.read_text())
    assert report["model_tag"] == "unobserved-model:control"
    assert report["model_manifest_digest"] is None
    assert report["ollama_cloud_disabled_by_runner"] is None
    assert report["model_provenance"].startswith("unverified:")
    assert report["server_configuration"].startswith("unverified:")
