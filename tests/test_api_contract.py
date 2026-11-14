"""The API's real behaviour, checked against a machine-readable contract.

FastAPI generates an OpenAPI document from the code, which makes the document a
description of the code rather than a check on it: any change to a route changes the
document with it and nothing ever disagrees. The check only has teeth once the document
is committed and compared, so `backend/openapi.json` is a reviewed artifact and
`test_the_committed_spec_and_the_live_app_agree` compares the two IN BOTH DIRECTIONS -
an operation or response the spec promises and the app no longer serves, and one the
app serves that the spec never described.

The rest of this file tests what the document cannot: that the responses the app
actually produces match what it says they are.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "backend" / "openapi.json"

from conftest import OTHER_RUN_ID, SEED_RUN_ID  # noqa: E402


# ── the committed contract ─────────────────────────────────────────────────────
def _operations(spec):
    return {
        (path, method.upper())
        for path, item in spec.get("paths", {}).items()
        for method in item
        if method.lower() in {"get", "put", "post", "delete", "patch", "head", "options"}
    }


def _responses(spec):
    return {
        (path, method.upper(), status)
        for path, item in spec.get("paths", {}).items()
        for method, op in item.items()
        if method.lower() in {"get", "put", "post", "delete", "patch"}
        for status in op.get("responses", {})
    }


def test_the_committed_spec_and_the_live_app_agree(app):
    assert SPEC_PATH.exists(), (
        f"{SPEC_PATH} is missing. Regenerate it with "
        "`python backend/export_openapi.py` and review the diff before committing."
    )
    committed = json.loads(SPEC_PATH.read_text())
    live = json.loads(json.dumps(app.openapi()))  # normalise tuples/ordering

    # Direction 1: the app no longer serves something the contract promises.
    dropped_ops = _operations(committed) - _operations(live)
    dropped_responses = _responses(committed) - _responses(live)
    # Direction 2: the app serves something the contract never described.
    undocumented_ops = _operations(live) - _operations(committed)
    undocumented_responses = _responses(live) - _responses(committed)

    assert not dropped_ops, f"operations in the spec that the app no longer serves: {sorted(dropped_ops)}"
    assert not undocumented_ops, f"operations the app serves that the spec does not describe: {sorted(undocumented_ops)}"
    assert not dropped_responses, f"responses promised by the spec but no longer declared: {sorted(dropped_responses)}"
    assert not undocumented_responses, f"responses the app declares that the spec does not: {sorted(undocumented_responses)}"

    # Schema drift: a field added, removed or made optional changes what a client may
    # send and must handle, and none of that shows up in the path/method sets above.
    cs = committed.get("components", {}).get("schemas", {})
    ls = live.get("components", {}).get("schemas", {})
    assert set(cs) == set(ls), f"schema set drift: only in spec {sorted(set(cs) - set(ls))}, only in app {sorted(set(ls) - set(cs))}"
    for name in sorted(cs):
        assert set(cs[name].get("properties", {})) == set(ls[name].get("properties", {})), \
            f"{name}: property drift"
        assert set(cs[name].get("required", [])) == set(ls[name].get("required", [])), \
            f"{name}: required-field drift"

    assert committed == live, "the committed spec and the live app differ; regenerate and review"


# ── behaviour the document claims but did not deliver ──────────────────────────
def _valid_decision(run_id):
    """A body that satisfies the documented Decision schema exactly."""
    return {
        "id": "client-supplied-id",
        "run_id": run_id,
        "timestamp": 1234.5,
        "anomaly_types": [{"type": "loss_spike", "step": 7}],
        "tools_used": ["read_config", "patch_config"],
        "agent_response": "lowered the learning rate",
        "fixed": True,
        "status": "fixed",
    }


def test_posting_a_decision_that_matches_the_schema_is_accepted(client):
    """The documented request body must actually work.

    The route declared `Decision`, whose anomaly field is `anomaly_types`, while
    db.insert_decision read `decision_payload["anomalies"]` - the key the agent's own
    logger payload uses, not the key the API contract declares. Every request that
    satisfied the published schema raised KeyError and came back as a 500.
    """
    r = client.post(f"/runs/{SEED_RUN_ID}/decisions", json=_valid_decision(SEED_RUN_ID))
    assert r.status_code == 200, f"documented body rejected: {r.status_code} {r.text}"
    assert r.json()["anomaly_types"] == [{"type": "loss_spike", "step": 7}]


def test_a_decision_is_filed_under_the_run_in_the_path_not_the_one_in_the_body(client, store):
    """The only ownership relation this API has is decision-belongs-to-run.

    There is no user, tenant or API key anywhere in Argus, so there is no notion of a
    caller owning an object and no authorisation test to write. The containment
    relation is real, though, and it is addressable twice in the same request: once in
    the path and once in the body. They must not be able to disagree.
    """
    body = _valid_decision(OTHER_RUN_ID)  # body points at a DIFFERENT existing run
    r = client.post(f"/runs/{SEED_RUN_ID}/decisions", json=body)
    assert r.status_code in (200, 409, 422), r.text

    if r.status_code == 200:
        assert r.json()["run_id"] == SEED_RUN_ID
    filed = [d for d in store.tables["decisions"] if d["id"] != "d0"]
    assert all(d["run_id"] == SEED_RUN_ID for d in filed), \
        "a decision landed under the run named in the body rather than the path"
    assert not any(d["run_id"] == OTHER_RUN_ID for d in store.tables["decisions"])


def test_a_client_supplied_id_cannot_overwrite_an_existing_decision(client, store):
    body = _valid_decision(SEED_RUN_ID)
    body["id"] = "d0"  # the seeded decision's id
    client.post(f"/runs/{SEED_RUN_ID}/decisions", json=body)
    seeded = [d for d in store.tables["decisions"] if d["id"] == "d0"]
    assert len(seeded) == 1, "a client-supplied id collided with a stored decision"
    assert seeded[0]["agent_response"] == "seeded", "a stored decision was overwritten"


def test_the_recovery_evidence_the_agent_stores_survives_the_response_model(client):
    """agent/logger.py writes an `attempt` object on every decision row.

    That object is the four-rung evidence ladder from agent/attempt.py - the whole
    reason a decision can be re-audited instead of taken on trust. `Decision` had no
    `attempt` field, and a FastAPI response_model drops undeclared keys silently, so
    the API served a decision with its evidence removed and a 200 on the way out.
    """
    r = client.get(f"/runs/{SEED_RUN_ID}/decisions")
    assert r.status_code == 200, r.text
    row = r.json()[0]
    assert "attempt" in row, "the response model stripped the recovery evidence"
    assert row["attempt"]["outcome"] == "requested"


def test_patching_the_status_of_a_run_that_does_not_exist_is_a_404(client, store):
    """Every other run-scoped route checks the run exists first; this one did not.

    It answered 200 {"status": "updated"} for a run id that was never created, so a
    caller with a stale or mistyped id was told its write had landed.
    """
    r = client.patch("/runs/does-not-exist/status", params={"status": "completed"})
    assert r.status_code == 404, f"expected 404, got {r.status_code} {r.text}"
    assert r.json()["detail"] == "run not found"


def test_syncing_metrics_for_a_missing_file_is_not_reported_as_success(client, store):
    """The route returned 200 carrying {"error": ...}.

    A 200 whose body is an error is the worst of both: the status line says the ingest
    succeeded, and nothing in the declared response schema tells a client to go looking
    for an `error` key.
    """
    r = client.post(f"/runs/{SEED_RUN_ID}/metrics/sync")
    assert r.status_code != 200 or "error" not in r.json(), \
        f"success status on a failed sync: {r.status_code} {r.text}"


def test_a_successful_sync_reports_what_it_ingested(client, store, tmp_path):
    run = store.tables["runs"][0]
    Path(run["metrics_file"]).write_text(
        "".join(json.dumps({"step": s, "epoch": 0, "train_loss": 1.0, "val_loss": 1.0,
                            "val_acc": 0.5, "grad_norm": 0.5, "timestamp": 1.0}) + "\n"
                for s in range(5))
    )
    r = client.post(f"/runs/{SEED_RUN_ID}/metrics/sync")
    assert r.status_code == 200, r.text
    assert r.json() == {"inserted": 5}
