"""Regressions for the defects Schemathesis found and the hand-rolled corpus did not.

Each test here names the check that produced it, so the provenance of the assertion is
readable from the file rather than only from the record. The reproduction in each
docstring is the exact request Schemathesis shrank to, translated to the in-process
client; every one of them fails on the parent commit.

Why these were missed by tests/test_api_generated_inputs.py, which sends 293 generated
inputs and is not weak: that corpus varies *values* inside a fixed set of hand-written
request shapes and asserts on status codes and body substrings. It never varies the
byte encoding of a body, never sends a method the route does not implement, never reads
a response header, never validates a response body against the committed schema, and
never composes two requests so that the output of one becomes the input of the next.
Those five blind spots are precisely where these defects live.
"""

import json
from pathlib import Path

import pytest

from conftest import SEED_RUN_ID

SPEC = json.loads((Path(__file__).resolve().parents[1] / "backend" / "openapi.json").read_text())


def _declared(template, method):
    return {int(c) for c in SPEC["paths"][template][method.lower()]["responses"]}


# ── check: not_a_server_error + response_schema_conformance ────────────────────

def test_a_run_created_with_a_blank_metrics_file_does_not_500_on_sync(client):
    """Schemathesis stateful phase, `POST /runs/` then `POST /runs/{id}/metrics/sync`.

    The blank value was accepted at creation and only failed one call later, as
    `PosixPath('.') has an empty name`. The corpus does send `("empty_string", "")` for
    every field of the create body, so it had the right value; what it does not do is
    then call a *second* endpoint with the id the first one returned, so the stored
    poison was never triggered.
    """
    r = client.post("/runs/", json={"name": "n", "config_path": "c",
                                    "metrics_file": "", "training_dir": "t"})
    assert r.status_code == 422, f"a blank metrics_file was accepted: {r.status_code} {r.text}"
    assert "metrics_file" in r.text


@pytest.mark.parametrize("field", ["name", "config_path", "metrics_file", "training_dir"])
def test_a_lone_surrogate_is_refused_rather_than_crashing_serialisation(client, field):
    """Schemathesis fuzzing phase, `POST /runs/`.

    A lone surrogate is a valid `str` and an invalid UTF-8 sequence. It was stored, and
    the crash happened in FastAPI's response serialisation *after* the route returned,
    so the route's own `except Exception` never saw it: the caller received a bare
    text/plain `Internal Server Error` with no error_id and nothing was logged.
    """
    body = {"name": "n", "config_path": "c", "metrics_file": "m", "training_dir": "t"}
    body[field] = "/tmp/\ud800.jsonl"
    # `json=` cannot carry this: httpx encodes the body itself and raises on a lone
    # surrogate before the request is sent, which is correct of the client and would
    # hide whether the server survives the decoded value. `ensure_ascii` renders the
    # surrogate as the six ASCII bytes `\ud800`, so the bytes on the wire are valid
    # JSON and the server is the first thing to decode them - which is exactly what
    # Schemathesis did over a real socket.
    raw = json.dumps(body, ensure_ascii=True).encode("ascii")
    r = client.post("/runs/", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422, f"{field}: surrogate accepted: {r.status_code} {r.text[:200]}"
    assert r.headers["content-type"].startswith("application/json")


# ── check: response_schema_conformance ─────────────────────────────────────────

def test_the_500_body_is_the_shape_the_contract_declares(client, store, monkeypatch):
    """Schemathesis `response_schema_conformance` on `POST /runs/{id}/metrics/sync`.

    `{"detail": {"detail": ..., "error_id": ...}}` is not `ErrorResponse`, which
    declares `detail` to be a string with `error_id` beside it. Every 500 the service
    produced violated its own published schema, and a client reading the document
    looked for `error_id` at the top level and never found it.
    """
    import db
    monkeypatch.setattr(db, "get_client",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    r = client.get("/runs/")
    assert r.status_code == 500
    payload = r.json()

    schema = SPEC["components"]["schemas"]["ErrorResponse"]
    assert set(payload) <= set(schema["properties"]), payload
    assert isinstance(payload["detail"], str), (
        f"the contract declares detail as a string; got {type(payload['detail']).__name__}"
    )
    assert isinstance(payload["error_id"], str) and len(payload["error_id"]) == 12, payload


# ── check: status_code_conformance ─────────────────────────────────────────────

@pytest.mark.parametrize("template,method,path", [
    ("/runs/", "POST", "/runs/"),
    ("/runs/{run_id}/decisions", "POST", f"/runs/{SEED_RUN_ID}/decisions"),
])
def test_an_undecodable_body_answers_a_status_the_contract_declares(client, template, method, path):
    """Schemathesis `status_code_conformance` on the two routes that take a body.

    Bytes that are not valid UTF-8 are rejected by Starlette before pydantic runs, so
    the answer is 400 and not the 422 every hand-written malformed body produces. The
    corpus's nine MALFORMED cases are all valid UTF-8 - even `b"\\x00\\x01\\x02"`, which
    decodes fine - so it only ever observed the 422 branch and the reachable 400 stayed
    undeclared.
    """
    r = client.post(path, content=b"a\xff", headers={"content-type": "application/json"})
    assert r.status_code == 400, r.status_code
    assert r.status_code in _declared(template, method), (
        f"{method} {template} answered {r.status_code}; the committed contract declares "
        f"only {sorted(_declared(template, method))}"
    )


# ── check: negative_data_rejection ─────────────────────────────────────────────

@pytest.mark.parametrize("value", [True, False])
def test_a_boolean_timestamp_is_refused_rather_than_coerced(client, value):
    """Schemathesis `negative_data_rejection` on `POST /runs/{id}/decisions`.

    `bool` subclasses `int`, so pydantic coerced `false` to `0.0` and `true` to `1.0`:
    a decision was stored, timestamped at the Unix epoch, and the caller got a 200. The
    corpus does send `("bool", True)` for every field of this body, but its only
    assertion is that the status is one the contract declares - and 200 is declared -
    so a wrong value accepted under a right status passed.
    """
    body = {"id": "x", "run_id": SEED_RUN_ID, "timestamp": value, "anomaly_types": [],
            "tools_used": [], "agent_response": "r", "fixed": None, "status": None}
    r = client.post(f"/runs/{SEED_RUN_ID}/decisions", json=body)
    assert r.status_code == 422, f"boolean timestamp stored as a number: {r.text[:200]}"


# ── check: allow_header_conformance ────────────────────────────────────────────

@pytest.mark.parametrize("path,expected", [
    ("/runs/", {"GET", "POST"}),
    (f"/runs/{SEED_RUN_ID}/decisions", {"GET", "POST"}),
    (f"/runs/{SEED_RUN_ID}", {"GET"}),
])
def test_a_405_names_the_methods_the_resource_supports(client, path, expected):
    """Schemathesis `allow_header_conformance`.

    RFC 9110 makes `Allow` mandatory on a 405. Starlette sends the status without it,
    so a client that asked the response which method to use instead got nothing. The
    corpus sends no OPTIONS request and reads no response header, so it could not have
    seen this.
    """
    r = client.request("OPTIONS", path)
    assert r.status_code == 405, r.status_code
    allow = r.headers.get("Allow")
    assert allow is not None, "a 405 with no Allow header"
    assert expected <= {m.strip() for m in allow.split(",")}, allow


# ── second round: defects the first round of fixes exposed ─────────────────────
# Fixing a defect changes what the generator can reach, so the run after a fix is not
# a formality. These three were invisible until the first round landed.

@pytest.mark.parametrize("field", ["config_path", "metrics_file", "training_dir"])
def test_the_non_blank_rule_is_in_the_contract_and_not_only_in_the_code(field):
    """Schemathesis `positive_data_acceptance` on `POST /runs/`.

    The first version of the blank-path fix was a pydantic validator, which rejected
    `""` while the published schema still said any string was acceptable. That is the
    same contract lie as the bug it fixed, pointing the other way, and Schemathesis
    reported it immediately as the API rejecting a schema-compliant request. The
    constraint now lives in the schema, so the document and the code agree.
    """
    schema = SPEC["components"]["schemas"]["RunCreate"]["properties"][field]
    assert schema.get("minLength") == 1, schema
    assert schema.get("pattern") == r"\S", schema


@pytest.mark.parametrize("value", [0, 1, 2])
def test_a_numeric_fixed_flag_is_refused_rather_than_coerced(client, value):
    """Schemathesis `negative_data_rejection` on `POST /runs/{id}/decisions`.

    `{"fixed": 0}` was stored as `False` where the contract declares a boolean, so a
    client sending a count by mistake recorded a definite "not fixed". Same class as
    the `timestamp` coercion, on a different field, found on the round after it - which
    is the argument for generation over one hand-written case per field.
    """
    body = {"id": "x", "run_id": SEED_RUN_ID, "timestamp": 1.0, "anomaly_types": [],
            "tools_used": [], "agent_response": "r", "fixed": value, "status": None}
    r = client.post(f"/runs/{SEED_RUN_ID}/decisions", json=body)
    assert r.status_code == 422, f"numeric fixed stored as a boolean: {r.text[:200]}"


def test_the_permitted_statuses_are_published_in_the_contract():
    """Schemathesis `positive_data_acceptance` on `PATCH /runs/{id}/status`.

    The three legal values were enforced in the handler and absent from the document,
    so `status` was declared as any string and a client had no way to discover the set
    short of reading the source or guessing until it stopped getting 400s.
    """
    params = SPEC["paths"]["/runs/{run_id}/status"]["patch"]["parameters"]
    status = next(p for p in params if p["name"] == "status")
    assert status["schema"].get("enum") == ["running", "completed", "failed"], status


# ── third round: found after the second round of fixes landed ──────────────────

@pytest.mark.parametrize("template,path", [
    ("/runs/{run_id}/decisions", f"/runs/{SEED_RUN_ID}/decisions"),
    ("/runs/{run_id}/metrics", f"/runs/{SEED_RUN_ID}/metrics"),
])
def test_the_limit_parameter_does_not_declare_a_null_it_cannot_accept(client, template, path):
    """Schemathesis `positive_data_acceptance` on both paginated reads.

    `limit: int | None` renders as `anyOf: [integer, null]`, so the document said null
    was a permitted value while `?limit=null` answered 422. There is no query string a
    client can write that means null; the parameter is optional, not nullable. The
    corpus never sent `limit` at all - it varies request *bodies* - so it could not
    have found a defect that lives in a query parameter's declared type.
    """
    schema = next(p for p in SPEC["paths"][template]["get"]["parameters"]
                  if p["name"] == "limit")["schema"]
    assert schema.get("type") == "integer", schema
    assert "anyOf" not in schema, schema
    assert schema.get("minimum") == 1 and schema.get("maximum") == 5000, schema

    assert client.get(path).status_code == 200
    assert client.get(path, params={"limit": 2}).status_code == 200
    # Still refused, and now correctly: the contract no longer promises it.
    assert client.get(f"{path}?limit=null").status_code == 422


def test_a_run_whose_metrics_file_is_missing_is_not_reported_as_a_missing_run(client, store):
    """Schemathesis `ensure_resource_availability` on `POST /runs/{id}/metrics/sync`.

    A run created one request earlier answered 404 to its own sync call, because the
    route reused `not_found` for a condition that is not a missing run: the run was
    resolved, and its `metrics_file` was what could not be read. One status carrying
    two meanings on one route is a contract defect on its own, and the two detail
    strings that distinguished them made it an existence oracle as well; see
    tests/test_api_authorization.py for that half.

    The corpus calls sync, but only ever against the seeded run and only asserting the
    status is one the contract declares - and 404 was declared - so a right status for
    the wrong reason passed. Composing create-then-sync is what exposed it.
    """
    r = client.post("/runs/", json={"name": "n", "config_path": "c",
                                    "metrics_file": "/nonexistent/nope.jsonl",
                                    "training_dir": "t"})
    assert r.status_code == 200, r.text
    run_id = r.json()["id"]
    assert client.get(f"/runs/{run_id}").status_code == 200, "the run must exist"

    s = client.post(f"/runs/{run_id}/metrics/sync")
    assert s.status_code == 409, f"the run exists; {s.status_code} says it does not"
    assert s.status_code in _declared("/runs/{run_id}/metrics/sync", "POST"), \
        f"409 is undeclared: {sorted(_declared('/runs/{run_id}/metrics/sync', 'POST'))}"
