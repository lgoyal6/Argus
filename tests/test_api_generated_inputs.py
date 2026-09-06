"""Generated boundary inputs against an isolated instance with seeded data.

Every request in this file goes to the in-memory store built in conftest.py. Nothing
here can reach a real Supabase project: `db.get_client` is replaced and
`supabase.create_client` is poisoned, so a code path that escaped the fixture would
fail the test rather than send a generated destructive body at real rows.

The property under test is not "the app returns the right answer" - for a body of
nonsense there is no right answer. It is that the app answers *within its own
contract*: a status it declared, never a 5xx, and never a body carrying interpreter
internals. A 500 means an unhandled exception reached the client, which is both a
crash and, as the secret-lifecycle work in this record shows, a disclosure channel.

The generator is a seeded deterministic product rather than hypothesis, so the suite
adds no dependency beyond what CI already installs and a failure reproduces exactly
from the case id in the test name.
"""

import json
import random
from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import SEED_RUN_ID  # noqa: E402

# Statuses these routes are allowed to answer with. Anything else, and in particular
# anything 5xx, is a failure of the contract rather than of the input.
ALLOWED = {200, 400, 404, 405, 409, 415, 422}

# The committed contract, so a generated request can be checked against what the API
# publishes rather than against what this file happens to expect. This is the direction
# the OpenAPI document cannot check on its own: the document is generated from the
# route decorators, so it records what the routes CLAIM they answer, and only sending
# real requests shows whether they answer anything else.
SPEC = json.loads((Path(__file__).resolve().parents[1] / "backend" / "openapi.json").read_text())

# Markers that mean interpreter or filesystem internals reached the response body.
LEAK_MARKERS = (
    "Traceback", 'File "/', "/Users/", "/app/", "site-packages",
    "KeyError", "AttributeError", "TypeError(", "psycopg", "postgrest",
    "supabase.co", "SUPABASE_KEY", "sk-ant-",
)

EXTREMES = [
    ("null", None),
    ("empty_string", ""),
    ("long_string", "x" * 20000),
    ("zero", 0),
    ("negative", -1),
    ("int64_max", 2 ** 63 - 1),
    ("int64_overflow", 2 ** 70),
    ("float_max", 1.7976931348623157e308),
    ("float_min", -1.7976931348623157e308),
    ("bool", True),
    ("empty_list", []),
    ("empty_object", {}),
    ("nested", {"a": {"b": {"c": [1, 2, 3]}}}),
    ("nul_byte", "a" + chr(0) + "b"),
    ("path_traversal", "../../../etc/passwd"),
    ("sql_ish", "'; DROP TABLE runs; --"),
    ("html", "<script>alert(1)</script>"),
    ("unicode", chr(0x202E) + chr(0x1F600) + chr(0x200B)),
    ("newlines", "a\r\nb\nc"),
]

VALID_RUN = {"name": "n", "config_path": "c", "metrics_file": "m", "training_dir": "t"}
VALID_DECISION = {
    "id": "i", "run_id": SEED_RUN_ID, "timestamp": 1.0, "anomaly_types": [],
    "tools_used": [], "agent_response": "r", "fixed": None, "status": None,
}


def _declared(template, method):
    op = SPEC["paths"].get(template, {}).get(method.lower())
    assert op is not None, f"the committed spec describes no {method} {template}"
    return {int(code) for code in op["responses"]}


def _check(resp, case, template=None, method=None):
    assert resp.status_code in ALLOWED, \
        f"{case}: undeclared status {resp.status_code} {resp.text[:300]}"
    if template is not None:
        declared = _declared(template, method or resp.request.method)
        assert resp.status_code in declared, (
            f"{case}: answered {resp.status_code}, but the committed contract for "
            f"{method} {template} declares only {sorted(declared)}"
        )
    body = resp.text
    for marker in LEAK_MARKERS:
        assert marker not in body, f"{case}: response leaked {marker!r}: {body[:300]}"


def _cases(valid):
    for field in valid:
        for label, value in EXTREMES:
            body = dict(valid)
            body[field] = value
            yield f"{field}={label}", body


_RUN_CASES = list(_cases(VALID_RUN))
_DECISION_CASES = list(_cases(VALID_DECISION))


@pytest.mark.parametrize("case,body", _RUN_CASES, ids=[c for c, _ in _RUN_CASES])
def test_create_run_survives_every_field_boundary(client, case, body):
    _check(client.post("/runs/", json=body), f"POST /runs/ {case}", "/runs/", "POST")


@pytest.mark.parametrize("case,body", _DECISION_CASES, ids=[c for c, _ in _DECISION_CASES])
def test_post_decision_survives_every_field_boundary(client, case, body):
    _check(client.post(f"/runs/{SEED_RUN_ID}/decisions", json=body),
           f"POST decisions {case}", "/runs/{run_id}/decisions", "POST")


@pytest.mark.parametrize("label,value", EXTREMES, ids=[label for label, _ in EXTREMES])
def test_path_and_query_parameters_survive_every_boundary(client, label, value):
    raw = "" if value is None else str(value)[:2000]
    # Percent-encoded so the bytes actually reach the server. httpx refuses to put a
    # NUL byte or a newline in a request line at all, which is correct of the client
    # and would otherwise hide whether the server handles the decoded value.
    seg = quote(raw, safe="")
    _check(client.get(f"/runs/{seg}"), f"GET /runs/{label}", "/runs/{run_id}", "GET")
    _check(client.get(f"/runs/{seg}/metrics"), f"GET metrics {label}",
           "/runs/{run_id}/metrics", "GET")
    _check(client.get(f"/runs/{seg}/decisions"), f"GET decisions {label}",
           "/runs/{run_id}/decisions", "GET")
    _check(client.patch(f"/runs/{SEED_RUN_ID}/status", params={"status": raw}),
           f"PATCH status={label}", "/runs/{run_id}/status", "PATCH")


# -- malformed bodies, unknown fields and wrong content types -------------------
MALFORMED = [
    ("truncated_json", b'{"name": "x"'),
    ("not_json", b"this is not json at all"),
    ("json_array", b'["name", "x"]'),
    ("json_scalar", b"42"),
    ("json_null", b"null"),
    ("empty_body", b""),
    ("nul_bytes", b"\x00\x01\x02"),
    ("deep_nesting", b'{"name":' + b"[" * 200 + b"]" * 200 + b"}"),
    ("duplicate_keys",
     b'{"name":"a","name":"b","config_path":"c","metrics_file":"m","training_dir":"t"}'),
]


@pytest.mark.parametrize("label,raw", MALFORMED, ids=[label for label, _ in MALFORMED])
def test_malformed_bodies_are_refused_not_crashed(client, label, raw):
    _check(client.post("/runs/", content=raw, headers={"content-type": "application/json"}),
           f"POST /runs/ malformed {label}", "/runs/", "POST")


CONTENT_TYPES = ["text/plain", "application/xml", "application/x-www-form-urlencoded",
                 "multipart/form-data", "application/octet-stream", ""]


@pytest.mark.parametrize("ctype", CONTENT_TYPES, ids=[c or "absent" for c in CONTENT_TYPES])
def test_wrong_content_types_are_refused_not_crashed(client, ctype):
    headers = {"content-type": ctype} if ctype else {}
    _check(client.post("/runs/", content=json.dumps(VALID_RUN).encode(), headers=headers),
           f"POST /runs/ content-type={ctype!r}", "/runs/", "POST")


def test_an_unknown_field_is_refused_rather_than_silently_dropped(client):
    """A field the server ignores is a field the client thinks it set.

    RunCreate used pydantic's default `extra="ignore"`, so a body carrying
    `trainng_dir` with a typo created a run whose real `training_dir` came from
    somewhere else entirely, and answered 200 as though the client had been obeyed.
    """
    body = dict(VALID_RUN)
    body["trainng_dir"] = "/typo"
    r = client.post("/runs/", json=body)
    assert r.status_code == 422, f"unknown field silently accepted: {r.status_code} {r.text}"


def test_no_generated_request_ever_created_a_run_outside_the_isolated_store(client, store):
    before = len(store.tables["runs"])
    client.post("/runs/", json={"name": None})
    client.post("/runs/", content=b"{", headers={"content-type": "application/json"})
    assert len(store.tables["runs"]) == before


# -- pagination bounds ----------------------------------------------------------
# GET /runs/{id}/metrics and /decisions grow with the length of a training run: the
# same growth that made the old whole-file ingest quadratic. They had no bound at all,
# so a client asking about a 20,000-step run got 20,000 rows in one response.
@pytest.mark.parametrize("path", ["metrics", "decisions"])
@pytest.mark.parametrize("params,expected", [
    ({"limit": 0}, 422),
    ({"limit": -1}, 422),
    ({"limit": 1}, 200),
    ({"limit": 5000}, 200),
    ({"limit": 5001}, 422),
    ({"limit": 2 ** 70}, 422),
    ({"limit": "abc"}, 422),
    ({"limit": ""}, 422),
    ({"offset": -1}, 422),
    ({"offset": 0}, 200),
    ({"offset": 10 ** 9}, 200),
    ({"offset": "abc"}, 422),
    ({"limit": 2, "offset": 1}, 200),
])
def test_pagination_bounds_are_enforced(client, path, params, expected):
    r = client.get(f"/runs/{SEED_RUN_ID}/{path}", params=params)
    assert r.status_code == expected, f"{path} {params}: {r.status_code} {r.text[:200]}"
    _check(r, f"GET {path} {params}", "/runs/{run_id}/" + path, "GET")


def test_pagination_actually_pages(client):
    everything = client.get(f"/runs/{SEED_RUN_ID}/metrics").json()
    assert len(everything) == 3, everything
    first = client.get(f"/runs/{SEED_RUN_ID}/metrics", params={"limit": 2}).json()
    second = client.get(f"/runs/{SEED_RUN_ID}/metrics", params={"limit": 2, "offset": 2}).json()
    assert first == everything[:2]
    assert second == everything[2:]
    beyond = client.get(f"/runs/{SEED_RUN_ID}/metrics", params={"offset": 10 ** 6}).json()
    assert beyond == []


def test_omitting_pagination_keeps_the_whole_series(client):
    """The dashboard charts a whole run and passes no limit.

    A default page size would silently truncate every chart, so the default stays
    unbounded and the bound is opt-in. Recorded as a known limit, not a fix.
    """
    assert len(client.get(f"/runs/{SEED_RUN_ID}/metrics").json()) == 3


# -- a seeded random walk over combined mutations -------------------------------
def test_randomised_combined_mutations(client):
    rng = random.Random(20260905)
    for i in range(300):
        body = dict(VALID_RUN)
        for field in list(body):
            if rng.random() < 0.4:
                body[field] = rng.choice(EXTREMES)[1]
        if rng.random() < 0.2:
            body[f"unknown_{i}"] = rng.choice(EXTREMES)[1]
        if rng.random() < 0.2:
            body.pop(rng.choice(list(body)), None)
        _check(client.post("/runs/", json=body),
               f"random case {i} seed=20260905 body={body!r}"[:400], "/runs/", "POST")
