"""Authorization, tested independently of the code path that grants access.

**The finding this file exists to record: Argus has no authorization to test.**

There is no user, no tenant, no owner column, no session, no API key and no token
anywhere in `backend/`. The committed contract declares no `securitySchemes` and no
operation carries a `security` requirement, which is asserted below rather than
asserted in prose. Every caller that can reach the port is the same anonymous
principal and has full read and write access to every run in the database. Nothing in
this file should be read as evidence that access control works, because there is none;
what the file does is pin that absence so it cannot be mistaken for coverage, and test
the one object-boundary relation the service *does* implement.

That relation is **run scoping**: `run_id` is the object identifier, every non-collection
route is addressed by it, and a request naming run B must not read or write run A. It
is a correctness property rather than a security property - with no principal there is
nothing to be authorized - but it is the boundary that exists, and it is the boundary a
future identity model would have to be enforced on top of.

**Independence.** The assertions here do not read the answer back through the API that
served the request, because a scoping bug in the route layer would then be checked by
the same buggy layer and would agree with itself. Every check is made against
`store.tables`, the underlying rows, which the request never passes through. A write
that leaked across runs would show up there even if every response body looked correct.
"""

import copy
import json
from pathlib import Path

import pytest

from conftest import OTHER_RUN_ID, SEED_RUN_ID

SPEC = json.loads((Path(__file__).resolve().parents[1] / "backend" / "openapi.json").read_text())

# A syntactically valid uuid that was never created, and a string that is not an id at
# all. Both are "some other principal's identifier" in the only sense Argus has one.
INVENTED_RUN_ID = "99999999-9999-9999-9999-999999999999"
NOT_AN_ID = "not-a-run-id"


# ── the finding itself ─────────────────────────────────────────────────────────

def test_the_contract_declares_no_authentication_of_any_kind():
    """If this ever fails, someone added auth and every test below needs rewriting.

    It is deliberately an assertion and not a comment. A reader who wants to know
    whether Argus authenticates anything gets the answer from a test result rather
    than from someone's summary of the code.
    """
    assert "securitySchemes" not in SPEC.get("components", {}), \
        "a security scheme appeared; the run-scoping tests below are no longer the " \
        "whole authorization story and real per-principal tests are now required"
    assert "security" not in SPEC, SPEC.get("security")
    declared = {f"{m.upper()} {p}": op.get("security")
                for p, ops in SPEC["paths"].items() for m, op in ops.items()
                if op.get("security")}
    assert declared == {}, declared


def test_no_route_accepts_a_credential_parameter():
    """No route takes an Authorization header, an api key or a user identifier.

    The absence of a `user_id`-shaped parameter is the point: there is nothing a caller
    could send that would make the service treat it as one principal rather than
    another.
    """
    credentialish = {"authorization", "x-api-key", "api_key", "apikey", "token",
                     "user_id", "tenant_id", "owner", "account_id", "session_id"}
    found = []
    for path, ops in SPEC["paths"].items():
        for method, op in ops.items():
            for param in op.get("parameters", []):
                if param["name"].lower() in credentialish:
                    found.append(f"{method.upper()} {path}: {param['name']}")
    assert found == [], found


# ── reads scoped to another run reach nothing of this one ──────────────────────

@pytest.mark.parametrize("suffix", ["", "/metrics", "/decisions"])
def test_a_read_addressed_to_another_run_returns_none_of_this_runs_rows(client, store, suffix):
    """Present run B's identifier at every read entry point; prove A's rows do not come back.

    The oracle is `store.tables`, not a second API call.
    """
    seeded_metric_ids = {m["id"] for m in store.tables["metrics"] if m["run_id"] == SEED_RUN_ID}
    seeded_decision_ids = {d["id"] for d in store.tables["decisions"] if d["run_id"] == SEED_RUN_ID}
    assert seeded_metric_ids and seeded_decision_ids, "the fixture must seed rows to hide"

    r = client.get(f"/runs/{OTHER_RUN_ID}{suffix}")
    assert r.status_code == 200, r.text
    body = r.text

    for row_id in seeded_metric_ids | seeded_decision_ids:
        assert row_id not in body, \
            f"a read scoped to {OTHER_RUN_ID} returned row {row_id}, which belongs to " \
            f"{SEED_RUN_ID}"
    assert SEED_RUN_ID not in body, body[:300]

    payload = r.json()
    if suffix:
        assert payload == [], f"the other run has no rows of its own; got {payload}"


def test_the_collection_route_is_the_one_place_that_is_not_scoped(client, store):
    """`GET /runs/` returns every run to every caller, by design and with no principal.

    Recorded as a test rather than left implicit: this is the route that makes the
    absence of tenancy visible, and any future identity model has to change it.
    """
    r = client.get("/runs/")
    ids = {run["id"] for run in r.json()}
    assert {SEED_RUN_ID, OTHER_RUN_ID} <= ids, ids


# ── writes scoped to another run change nothing of this one ────────────────────

def test_a_decision_written_against_another_run_does_not_touch_this_runs_rows(client, store):
    before = copy.deepcopy(store.tables["decisions"])
    seeded_before = [d for d in before if d["run_id"] == SEED_RUN_ID]

    body = {"id": "written-at-other", "run_id": OTHER_RUN_ID, "timestamp": 1.0,
            "anomaly_types": [], "tools_used": [], "agent_response": "x",
            "fixed": None, "status": None}
    r = client.post(f"/runs/{OTHER_RUN_ID}/decisions", json=body)
    assert r.status_code == 200, r.text

    seeded_after = [d for d in store.tables["decisions"] if d["run_id"] == SEED_RUN_ID]
    assert seeded_after == seeded_before, \
        "a write scoped to another run mutated this run's decisions"


def test_a_status_write_against_another_run_does_not_change_this_runs_status(client, store):
    before = {r["id"]: r["status"] for r in store.tables["runs"]}

    r = client.patch(f"/runs/{OTHER_RUN_ID}/status", params={"status": "completed"})
    assert r.status_code == 200, r.text

    after = {r["id"]: r["status"] for r in store.tables["runs"]}
    assert after[OTHER_RUN_ID] == "completed", after
    assert after[SEED_RUN_ID] == before[SEED_RUN_ID], \
        "a status write scoped to another run changed this run's status"


def test_a_body_naming_a_different_run_than_the_path_is_refused_and_writes_nothing(client, store):
    """The one cross-object guard the service has: path and body must agree.

    Both runs' rows are compared before and after, so a 409 that had already written
    would still fail here.
    """
    before = copy.deepcopy(store.tables["decisions"])
    body = {"id": "smuggled", "run_id": SEED_RUN_ID, "timestamp": 1.0,
            "anomaly_types": [], "tools_used": [], "agent_response": "x",
            "fixed": None, "status": None}
    r = client.post(f"/runs/{OTHER_RUN_ID}/decisions", json=body)
    assert r.status_code == 409, f"{r.status_code} {r.text}"
    assert store.tables["decisions"] == before, "the refused write still landed"


def test_a_write_to_an_invented_run_creates_nothing_anywhere(client, store):
    """A refusal must not be a side door for creating the object it refused."""
    runs_before = copy.deepcopy(store.tables["runs"])
    decisions_before = copy.deepcopy(store.tables["decisions"])

    body = {"id": "ghost", "run_id": INVENTED_RUN_ID, "timestamp": 1.0,
            "anomaly_types": [], "tools_used": [], "agent_response": "x",
            "fixed": None, "status": None}
    assert client.post(f"/runs/{INVENTED_RUN_ID}/decisions", json=body).status_code == 404
    assert client.patch(f"/runs/{INVENTED_RUN_ID}/status",
                        params={"status": "completed"}).status_code == 404
    assert client.post(f"/runs/{INVENTED_RUN_ID}/metrics/sync").status_code == 404

    assert store.tables["runs"] == runs_before
    assert store.tables["decisions"] == decisions_before


# ── the refusal must not be an oracle for how the id was wrong ─────────────────

@pytest.mark.parametrize("suffix", ["", "/metrics", "/decisions"])
def test_two_different_kinds_of_nonexistent_id_are_refused_identically(client, suffix):
    """A well-formed-but-unknown uuid and a string that is not an id at all.

    If these differed - one 404 and one 422, or two different bodies - the error would
    tell a caller which of its guesses had the right *shape*, which is the first half
    of an enumeration oracle. They are required to be byte-identical.
    """
    a = client.get(f"/runs/{INVENTED_RUN_ID}{suffix}")
    b = client.get(f"/runs/{NOT_AN_ID}{suffix}")
    assert a.status_code == b.status_code == 404, (a.status_code, b.status_code)
    assert a.content == b.content, (a.content, b.content)


@pytest.mark.parametrize("suffix", ["", "/metrics", "/decisions"])
def test_a_real_other_run_is_distinguishable_from_an_invented_one(client, suffix):
    """The other half of the oracle, and it is NOT closed. This is the finding.

    C03 asks that refusals be byte-identical for a real identifier belonging to another
    principal and for an invented one, so that the error cannot be used to discover
    which objects exist. Argus cannot satisfy that, and no amount of error shaping
    would fix it: with no principal there is no such thing as "another principal's run",
    every real run is readable in full by every caller, and existence is disclosed by
    the successful response rather than by the error. `GET /runs/` publishes the entire
    list to anyone regardless.

    This test asserts the disclosure exists, so that the gap is a recorded, failing-if-
    changed fact rather than an omission. It is the point at which an identity model
    would have to be introduced; error-message work alone cannot close it.
    """
    real_other = client.get(f"/runs/{OTHER_RUN_ID}{suffix}")
    invented = client.get(f"/runs/{INVENTED_RUN_ID}{suffix}")
    assert real_other.status_code == 200
    assert invented.status_code == 404
    assert real_other.content != invented.content


def test_a_write_entry_point_refuses_a_real_id_and_an_invented_one_identically(client, store):
    """The oracle C03 names, closed on the one route where it was open.

    `POST /runs/{id}/metrics/sync` answered 404 twice with two different bodies:
    `{"detail":"run not found"}` for an id that was never created, and
    `{"detail":"metrics file not found"}` for a real run whose file is absent - which
    every seeded run's is. A caller with no credentials could therefore sweep run ids
    and read existence straight off the refusal body. The two conditions now differ by
    status, so a 404 from this route means exactly one thing and carries exactly one
    body regardless of which id produced it.

    The seeded run is used deliberately: its `metrics_file` points at a path the
    fixture never creates, so it is precisely the case that used to leak.
    """
    assert any(r["id"] == SEED_RUN_ID for r in store.tables["runs"])

    real = client.post(f"/runs/{SEED_RUN_ID}/metrics/sync")
    invented = client.post(f"/runs/{INVENTED_RUN_ID}/metrics/sync")
    not_an_id = client.post(f"/runs/{NOT_AN_ID}/metrics/sync")

    assert real.status_code == 409, f"a real run reported as missing: {real.text}"
    assert invented.status_code == not_an_id.status_code == 404, \
        (invented.status_code, not_an_id.status_code)
    assert invented.content == not_an_id.content, (invented.content, not_an_id.content)


@pytest.mark.parametrize("path_for", [
    lambda rid: (f"/runs/{rid}/decisions", "post"),
    lambda rid: (f"/runs/{rid}/status", "patch"),
    lambda rid: (f"/runs/{rid}/metrics/sync", "post"),
])
def test_no_write_entry_point_distinguishes_two_absent_ids(client, path_for):
    """Every write entry point, both kinds of absent identifier, byte-identical.

    A well-formed uuid that was never created and a string that is not an id at all
    must be refused the same way. If one produced a 404 and the other a 422, the
    refusal would tell a caller which of its guesses had the right shape.
    """
    a_path, method = path_for(INVENTED_RUN_ID)
    b_path, _ = path_for(NOT_AN_ID)
    kwargs = {}
    if method == "post" and a_path.endswith("/decisions"):
        kwargs["json"] = {"id": "x", "run_id": INVENTED_RUN_ID, "timestamp": 1.0,
                          "anomaly_types": [], "tools_used": [], "agent_response": "x",
                          "fixed": None, "status": None}
    elif method == "patch":
        kwargs["params"] = {"status": "completed"}

    a = client.request(method.upper(), a_path, **kwargs)
    if "json" in kwargs:
        kwargs["json"] = {**kwargs["json"], "run_id": NOT_AN_ID}
    b = client.request(method.upper(), b_path, **kwargs)

    assert a.status_code == b.status_code == 404, (a.status_code, b.status_code)
    assert a.content == b.content, (a.content, b.content)
