"""Every bound C08 names, checked, each with a negative control that fails.

A test that only ever passes proves that the test ran, not that the bound holds. For
each of the six dimensions there are two assertions: the limit refuses the abusive
request, and with the same limit disabled (`0`) the same request is NOT refused. The
second is the control. If a bound were deleted from the source the control would keep
passing and the first assertion would fail, which is the whole point of writing it.

The app is imported flat, the way `docker/Dockerfile.backend` runs it (`COPY backend/ .`,
so `main.py` sits at the root and does `from routes import runs`). Importing it as
`backend.main` would exercise an import graph production does not have.

No test may reach a real Supabase project. `db.get_client` is replaced with an in-memory
store and `supabase.create_client` is poisoned, so a code path that bypasses the fixture
fails loudly instead of quietly opening a network client against whatever `SUPABASE_URL`
is in the environment.
"""

import asyncio
import gzip
import importlib
import os
import sys
import time
import zlib
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for _p in (str(ROOT), str(BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── in-memory stand-in for the supabase query builder ─────────────────────────
class _Result:
    def __init__(self, data, count=None):
        self.data = data
        self.count = count


class _Query:
    def __init__(self, rows, op, payload=None):
        self._rows = rows
        self._op = op
        self._payload = payload
        self._filters = []
        self._range = None
        self._count = None
        self._limit = None

    def eq(self, col, val):
        self._filters.append((col, val))
        return self

    def order(self, *a, **k):
        return self

    def limit(self, n):
        self._limit = n
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def execute(self):
        if self._op in ("insert", "upsert"):
            payload = self._payload
            self._rows.extend(payload if isinstance(payload, list) else [payload])
            return _Result(payload)
        data = [r for r in self._rows if all(r.get(c) == v for c, v in self._filters)]
        total = len(data)
        if self._range is not None:
            start, end = self._range
            data = data[start : end + 1]
        elif self._limit is not None:
            data = data[: self._limit]
        return _Result(data, count=total if self._count else None)


class _Table:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *a, **k):
        q = _Query(self._rows, "select")
        q._count = k.get("count") == "exact"
        return q

    def insert(self, payload):
        return _Query(self._rows, "insert", payload)

    def upsert(self, payload, **k):
        return _Query(self._rows, "upsert", payload)

    def update(self, payload):
        return _Query(self._rows, "update", payload)


class FakeClient:
    def __init__(self):
        self.tables = {"runs": [], "metrics": [], "decisions": []}
        self.get_run_delay = 0.0

    def table(self, name):
        return _Table(self.tables.setdefault(name, []))


RUN_ID = "11111111-1111-1111-1111-111111111111"


def _run_row(metrics_file="/nonexistent/metrics.jsonl"):
    return {
        "id": RUN_ID,
        "name": "r",
        "config_path": "c.yaml",
        "metrics_file": metrics_file,
        "training_dir": "/tmp",
        "status": "running",
        "created_at": 0.0,
        "updated_at": 0.0,
    }


def _metric_row(step):
    return {
        "run_id": RUN_ID,
        "step": step,
        "epoch": 0,
        "train_loss": 0.1,
        "val_loss": 0.1,
        "val_acc": 0.9,
        "grad_norm": 1.0,
        "timestamp": 0.0,
    }


def build(env=None, metrics_rows=0, run_delay=0.0, metrics_file=None):
    """Import the app fresh under `env`, wired to an in-memory store.

    `limits` reads its numbers at import so routes and `db` can see them without a
    settings object threaded through every call. That makes re-import, not
    monkeypatching an attribute, the honest way to test a different configuration:
    it is the same code path an operator gets by setting the variable and starting
    the server.
    """
    saved = {k: os.environ.get(k) for k in (env or {})}
    os.environ.setdefault("SUPABASE_URL", "http://unit.test.invalid")
    os.environ.setdefault("SUPABASE_KEY", "unit-test-key")
    os.environ.update({k: str(v) for k, v in (env or {}).items()})
    for name in ("main", "limits", "db", "routes.runs", "routes.metrics", "routes.decisions", "routes"):
        sys.modules.pop(name, None)
    try:
        import supabase

        supabase.create_client = lambda *a, **k: pytest.fail(
            "create_client must never be called from the test suite"
        )
        limits = importlib.import_module("limits")
        db = importlib.import_module("db")
        main = importlib.import_module("main")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    client = FakeClient()
    client.tables["runs"].append(_run_row(metrics_file or "/nonexistent/metrics.jsonl"))
    client.tables["metrics"].extend(_metric_row(i) for i in range(metrics_rows))
    db.get_client = lambda: client

    if run_delay:
        real_get_run = db.get_run

        def slow_get_run(run_id):
            time.sleep(run_delay)
            return real_get_run(run_id)

        db.get_run = slow_get_run
        for mod in ("routes.runs", "routes.metrics", "routes.decisions"):
            sys.modules[mod].db = db

    return main.app, limits, db, client


_RELOADED = ("main", "limits", "db", "routes.runs", "routes.metrics",
             "routes.decisions", "routes")


@pytest.fixture(autouse=True)
def _restore_default_limits():
    """Put the default-configured modules back after every test in this file.

    `build` reconfigures by re-importing under a different environment, which leaves
    the reconfigured modules in sys.modules. Every other test module imports `db` and
    `main` by name, so without this a run of 250 metrics rows in one of them would be
    refused by a 100-row bound that a test here set. The isolation belongs to this
    file, not to conftest, because this file is the only one that reconfigures.
    """
    yield
    for name in _RELOADED:
        sys.modules.pop(name, None)
    for name in ("ARGUS_MAX_BODY_BYTES", "ARGUS_MAX_DECOMPRESSED_BYTES",
                 "ARGUS_MAX_WORK_UNITS", "ARGUS_MAX_INGEST_BYTES",
                 "ARGUS_REQUEST_TIMEOUT_S", "ARGUS_MAX_QUEUE",
                 "ARGUS_MAX_CONCURRENCY", "ARGUS_RESERVED_LIGHT"):
        os.environ.pop(name, None)
    importlib.import_module("limits")
    importlib.import_module("db")
    importlib.import_module("main")


def _slow_metrics(db, delay):
    """Make only the HEAVY read slow. `run_delay` in build() slows db.get_run, which
    the LIGHT single-run route calls too, and would make this measure the handler
    rather than admission."""
    real = db.get_metrics

    def slow(run_id, limit=None, offset=0):
        time.sleep(delay)
        return real(run_id, limit=limit, offset=offset)

    db.get_metrics = slow
    sys.modules["routes.metrics"].db = db


def call(app, method, path, **kw):
    """One request against the app in-process.

    httpx's ASGITransport is async-only, so the synchronous test body drives it
    through `asyncio.run`. That keeps each test a plain function while still
    exercising the real ASGI middleware stack rather than a stubbed one.
    """

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=30.0) as c:
            return await c.request(method, path, **kw)

    return asyncio.run(go())


# ── 1. bytes on the wire ───────────────────────────────────────────────────────
CAP = 64 * 1024
BODY = b'{"name":"' + b"x" * (CAP * 2) + b'","config_path":"c","metrics_file":"m","training_dir":"t"}'


def test_body_over_the_cap_is_refused_with_413():
    app, _, _, _ = build({"ARGUS_MAX_BODY_BYTES": CAP})
    r = call(app, "POST", "/runs/", content=BODY, headers={"content-type": "application/json"})
    assert r.status_code == 413, r.text
    assert str(CAP) in r.json()["detail"]


def test_negative_control_body_cap_disabled_lets_the_same_body_through():
    """Control for the byte bound. The same oversized body, cap off, must NOT be 413.

    It is a 500 rather than a 200 because the in-memory store is not a real Supabase
    and the insert path is not what is under test here; what matters is that the
    request reached the application at all instead of being stopped at the edge.
    """
    app, _, _, _ = build({"ARGUS_MAX_BODY_BYTES": 0})
    r = call(app, "POST", "/runs/", content=BODY, headers={"content-type": "application/json"})
    assert r.status_code != 413, "cap disabled but the request was still refused"


def test_a_lying_content_length_does_not_get_past_the_cap():
    """Chunked, so there is no Content-Length to check; the buffered total catches it."""
    app, _, _, _ = build({"ARGUS_MAX_BODY_BYTES": CAP})

    async def chunks():
        for _ in range(8):
            yield b"x" * (CAP // 2)

    r = call(app, "POST", "/runs/", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413, r.text


# ── 2. bytes after decompression ───────────────────────────────────────────────
def _bomb(n_bytes):
    return gzip.compress(b"0" * n_bytes)


def test_a_gzip_bomb_is_refused_on_its_decompressed_size():
    app, _, _, _ = build({"ARGUS_MAX_BODY_BYTES": 1 << 20, "ARGUS_MAX_DECOMPRESSED_BYTES": 1 << 20})
    bomb = _bomb(64 << 20)  # 64 MiB of zeros
    assert len(bomb) < (1 << 20), "the bomb must be small on the wire or it tests the wrong bound"
    r = call(
        app, "POST", "/runs/",
        content=bomb,
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    )
    assert r.status_code == 413, r.text
    assert "decompressed" in r.json()["detail"]


def test_negative_control_decompression_cap_disabled_expands_the_same_bomb():
    """Control for the decompressed bound.

    A smaller bomb, because the control deliberately lets it expand in memory and
    the point is only that the refusal came from the cap and not from something else.
    """
    app, _, _, _ = build({"ARGUS_MAX_BODY_BYTES": 1 << 20, "ARGUS_MAX_DECOMPRESSED_BYTES": 0})
    r = call(
        app, "POST", "/runs/",
        content=_bomb(4 << 20),
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    )
    assert r.status_code != 413, "decompression cap disabled but the bomb was still refused"


def test_a_truthful_gzip_body_is_accepted_and_read_as_plaintext():
    """Compression is not treated as hostile; only its expansion is bounded."""
    app, _, _, _ = build({"ARGUS_MAX_DECOMPRESSED_BYTES": 1 << 20})
    good = gzip.compress(b'{"name":"n","config_path":"c","metrics_file":"m","training_dir":"t"}')
    r = call(
        app, "POST", "/runs/",
        content=good,
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    )
    assert r.status_code != 413, r.text
    assert r.status_code != 422, "the route did not see valid JSON: " + r.text


def test_a_malformed_gzip_body_is_a_400_not_a_500():
    app, _, _, _ = build({})
    r = call(
        app, "POST", "/runs/",
        content=b"not gzip at all",
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    )
    assert r.status_code == 400, r.text


def test_an_unknown_content_encoding_is_refused_rather_than_guessed():
    app, _, _, _ = build({})
    r = call(
        app, "POST", "/runs/",
        content=b"...",
        headers={"content-type": "application/json", "content-encoding": "br"},
    )
    assert r.status_code == 415, r.text


# ── 3. work per request ────────────────────────────────────────────────────────
def test_a_series_longer_than_the_work_bound_is_refused_not_truncated():
    app, _, _, _ = build({"ARGUS_MAX_WORK_UNITS": 100}, metrics_rows=250)
    r = call(app, "GET", f"/runs/{RUN_ID}/metrics")
    assert r.status_code == 413, r.text
    assert "offset" in r.json()["detail"], "the refusal must say how to page"


def test_paging_under_the_work_bound_still_serves_the_same_run():
    app, _, _, _ = build({"ARGUS_MAX_WORK_UNITS": 100}, metrics_rows=250)
    seen = []
    for offset in (0, 100, 200):
        r = call(app, "GET", f"/runs/{RUN_ID}/metrics?limit=100&offset={offset}")
        assert r.status_code == 200, r.text
        seen.extend(row["step"] for row in r.json())
    assert seen == list(range(250)), "paging lost or duplicated rows"


def test_negative_control_work_bound_disabled_serves_the_whole_series():
    app, _, _, _ = build({"ARGUS_MAX_WORK_UNITS": 0}, metrics_rows=250)
    r = call(app, "GET", f"/runs/{RUN_ID}/metrics")
    assert r.status_code == 200, r.text
    assert len(r.json()) == 250


def test_an_oversized_explicit_limit_is_refused_before_any_query_runs():
    app, _, _, _ = build({"ARGUS_MAX_WORK_UNITS": 100}, metrics_rows=0)
    r = call(app, "GET", f"/runs/{RUN_ID}/metrics?limit=1000")
    assert r.status_code == 413, r.text


def test_a_limit_above_the_published_maximum_is_refused_by_the_contract():
    """The other half of the same bound, and the reason the number above is 1000.

    routes/metrics.py publishes `limit` with `maximum: 5000`, so anything larger is
    refused at validation with a 422 the committed document declares, before the work
    bound is ever consulted. Both refusals are wanted: the contract one is what a
    client reading the document can predict, and the work bound is what holds when an
    operator configures something tighter than the published ceiling.
    """
    app, _, _, _ = build({"ARGUS_MAX_WORK_UNITS": 100}, metrics_rows=0)
    r = call(app, "GET", f"/runs/{RUN_ID}/metrics?limit=100000")
    assert r.status_code == 422, r.text


# ── 4. work per request, ingest side ───────────────────────────────────────────
def _write_metrics_file(tmp_path, n):
    p = tmp_path / "metrics.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            f.write(
                '{"step":%d,"epoch":0,"train_loss":0.1,"val_loss":0.1,'
                '"val_acc":0.9,"grad_norm":1.0,"timestamp":0.0}\n' % i
            )
    return p


# The ingest bound is not a refusal any more, and these three are the rewritten form
# of the three that asserted it was. backend/tailer.py reads only what was appended
# since the last committed cursor, at most MAX_WORK_UNITS rows and MAX_INGEST_BYTES
# per poll, and resumes on the next one. Refusing an oversized file outright would
# now be the wrong answer: a run whose metrics file legitimately grows past any fixed
# cap would become permanently un-ingestable, and the whole-file read the refusal
# needed is exactly the quadratic ingest that was removed. What is still asserted is
# the same property under the new shape: one poll's work is bounded, the bound is the
# operator's number, and nothing is lost.
def test_one_poll_reads_a_bounded_slice_of_an_oversized_file(tmp_path):
    p = _write_metrics_file(tmp_path, 20_000)
    app, _, _, _ = build({"ARGUS_MAX_WORK_UNITS": 100}, metrics_file=str(p))
    r = call(app, "POST", f"/runs/{RUN_ID}/metrics/sync")
    assert r.status_code == 200, r.text
    assert r.json() == {"inserted": 100}, "one poll ingested more than the row bound"


def test_the_byte_bound_tightens_the_same_poll(tmp_path):
    """A row cap of 100 and a byte cap that admits fewer than 100 rows: bytes win."""
    p = _write_metrics_file(tmp_path, 20_000)
    per_row = p.stat().st_size // 20_000
    app, _, _, _ = build({"ARGUS_MAX_WORK_UNITS": 100,
                          "ARGUS_MAX_INGEST_BYTES": per_row * 10},
                         metrics_file=str(p))
    r = call(app, "POST", f"/runs/{RUN_ID}/metrics/sync")
    assert r.status_code == 200, r.text
    assert 0 < r.json()["inserted"] <= 10, r.json()


def test_nothing_is_lost_the_bounded_polls_add_up(tmp_path):
    """The negative control for both bounds: the run is ingested, just not at once.

    A bound that dropped rows would pass the two tests above and fail this one, which
    is the failure the old whole-file version could not have, because it refused
    instead of paging.
    """
    p = _write_metrics_file(tmp_path, 450)
    app, _, _, client = build({"ARGUS_MAX_WORK_UNITS": 100}, metrics_file=str(p))
    inserted = 0
    for _ in range(10):
        r = call(app, "POST", f"/runs/{RUN_ID}/metrics/sync")
        assert r.status_code == 200, r.text
        got = r.json()["inserted"]
        inserted += got
        if got == 0:
            break
    assert inserted == 450, f"polls ingested {inserted} of 450 rows"
    assert sorted(row["step"] for row in client.tables["metrics"]) == list(range(450))


# ── 5. runtime ─────────────────────────────────────────────────────────────────
def test_a_request_over_the_time_bound_gets_504():
    app, _, _, _ = build({"ARGUS_REQUEST_TIMEOUT_S": 0.3}, run_delay=2.0)
    r = call(app, "GET", f"/runs/{RUN_ID}")
    assert r.status_code == 504, r.text


def test_negative_control_timeout_disabled_lets_the_slow_request_finish():
    app, _, _, _ = build({"ARGUS_REQUEST_TIMEOUT_S": 0}, run_delay=2.0)
    t0 = time.monotonic()
    r = call(app, "GET", f"/runs/{RUN_ID}")
    assert r.status_code == 200, r.text
    assert time.monotonic() - t0 > 1.5, "the slow handler did not actually run"


# ── the reserved lane: overload WITHOUT starvation ─────────────────────────────
# The clause the reservation exists for, and the one nothing else here covers. A
# plain semaphore plus a queue gives "refuses under overload": a flood of expensive
# requests fills every slot, fills the queue, and a cheap legitimate request is
# refused along with the flood. It is bounded, it is predictable, and the legitimate
# caller still gets nothing. These two tests are the same flood with the reservation
# on and off, so the reservation is shown to be the thing that carries the clause
# rather than asserted to be.
HEAVY_SECONDS = 1.0


def _flood_then_one_light(app, heavy_n):
    """Fire `heavy_n` slow heavy requests, then one cheap light one, in one loop.

    One event loop for all of them, because the admission Condition binds to the
    loop it is first awaited in. The light request is `GET /`, which touches no
    database at all, so anything it waits for is admission and nothing else.
    Returns (response, seconds the light request took).
    """

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t",
                                     timeout=30.0) as c:
            heavy = [asyncio.ensure_future(c.get(f"/runs/{RUN_ID}/metrics"))
                     for _ in range(heavy_n)]
            # Let every heavy request reach admission before the light one arrives.
            await asyncio.sleep(0.25)
            t0 = time.monotonic()
            light = await c.get("/")
            waited = time.monotonic() - t0
            await asyncio.gather(*heavy, return_exceptions=True)
            return light, waited

    return asyncio.run(go())


# Everything below is held fixed across the pair except RESERVED_LIGHT, so the
# difference in outcome is attributable to the reservation and to nothing else.
_LANE_ENV = {"ARGUS_MAX_CONCURRENCY": 3, "ARGUS_MAX_QUEUE": 8,
             "ARGUS_REQUEST_TIMEOUT_S": 0}


def test_a_heavy_flood_cannot_take_the_slot_a_cheap_request_needs():
    app, _, db, _ = build({**_LANE_ENV, "ARGUS_RESERVED_LIGHT": 1}, metrics_rows=5)
    _slow_metrics(db, HEAVY_SECONDS)
    light, waited = _flood_then_one_light(app, heavy_n=6)
    assert light.status_code == 200, light.text
    assert waited < HEAVY_SECONDS / 2, (
        f"the cheap request waited {waited:.2f}s behind the flood; the reserved "
        "slot did not hold")


def test_negative_control_with_no_reservation_the_same_flood_starves_it():
    """The same flood with RESERVED_LIGHT=0, and nothing else changed.

    The request is still answered, and still bounded, which is exactly the point: a
    plain semaphore plus a queue is not wrong, it is just not the clause. Without a
    reserved slot the cheap request waits for an expensive one to finish, so
    "rejects under overload" holds and "without starving legitimate work" does not.
    """
    app, _, db, _ = build({**_LANE_ENV, "ARGUS_RESERVED_LIGHT": 0}, metrics_rows=5)
    _slow_metrics(db, HEAVY_SECONDS)
    light, waited = _flood_then_one_light(app, heavy_n=6)
    assert light.status_code == 200, light.text
    assert waited > HEAVY_SECONDS * 0.6, (
        f"the cheap request came back in {waited:.2f}s with no reservation in "
        "force; the pair is not discriminating between the two configurations")


def test_the_queue_bound_refuses_instead_of_growing():
    """The other half of admission: waiting is bounded too, and says so.

    An unbounded queue turns a load problem into a memory problem and then into a
    latency problem, and the client has usually given up by the time its turn
    arrives. Retry-After is asserted because a 503 a client cannot pace itself
    against is a retry storm.
    """
    app, _, db, _ = build(
        {"ARGUS_MAX_CONCURRENCY": 2, "ARGUS_RESERVED_LIGHT": 0,
         "ARGUS_MAX_QUEUE": 1, "ARGUS_REQUEST_TIMEOUT_S": 0},
        metrics_rows=5,
    )
    _slow_metrics(db, HEAVY_SECONDS)
    light, _ = _flood_then_one_light(app, heavy_n=6)
    assert light.status_code == 503, light.text
    assert light.headers.get("retry-after") == "1"


def test_negative_control_a_bigger_queue_admits_the_same_request():
    """Same flood, same concurrency, queue raised: the 503 above was the queue bound
    and not simply the server being busy."""
    app, _, db, _ = build(
        {"ARGUS_MAX_CONCURRENCY": 2, "ARGUS_RESERVED_LIGHT": 0,
         "ARGUS_MAX_QUEUE": 16, "ARGUS_REQUEST_TIMEOUT_S": 0},
        metrics_rows=5,
    )
    _slow_metrics(db, HEAVY_SECONDS)
    light, _ = _flood_then_one_light(app, heavy_n=6)
    assert light.status_code == 200, light.text


# ── 6. the limits are readable off a running server ────────────────────────────
def test_the_server_reports_the_limits_in_force():
    app, _, _, _ = build({"ARGUS_MAX_BODY_BYTES": 4321, "ARGUS_RESERVED_LIGHT": 3})
    r = call(app, "GET", "/limits")
    assert r.status_code == 200, r.text
    assert r.json()["max_body_bytes"] == 4321
    assert r.json()["reserved_light"] == 3


# ── layer discrimination ──────────────────────────────────────────────────────
#
# The four tests below come from the harness the previous agent on this task left in
# `.agent-work/test_limits.py`, ported to these fixtures rather than lost. Its own
# mutation matrix had found that three of its tests could not fail: each bound is
# enforced by two layers, so removing one layer left the other to catch the abuse and
# every test stayed green. Redundancy in the enforcement is good. A suite that cannot
# tell the two layers apart is not, because it cannot report which one broke.
#
# Each of these is built around a discriminator that only one layer can produce.


def test_the_request_classifier_matches_the_documented_split():
    _, limits, _, _ = build({})
    assert limits.classify("GET", "/runs/abc/metrics") == limits.HEAVY
    assert limits.classify("GET", "/runs/abc/decisions") == limits.HEAVY
    assert limits.classify("POST", "/runs/") == limits.HEAVY
    assert limits.classify("PATCH", "/runs/abc/status") == limits.HEAVY
    assert limits.classify("GET", "/runs/abc") == limits.LIGHT
    assert limits.classify("GET", "/runs/") == limits.LIGHT
    assert limits.classify("GET", "/") == limits.LIGHT


def test_the_content_length_check_refuses_before_the_body_is_read():
    """Discriminator: a declared 10 MB with only 2 bytes actually sent.

    If the header check is gone the byte counter sees 2 bytes, lets it through, and the
    request gets some other status. Only the header path can produce a 413 here, so this
    test fails if that specific layer is removed even though the streaming check remains.
    """
    _, limits, _, _ = build({"ARGUS_MAX_BODY_BYTES": 4096})
    seen = {}

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            seen["status"] = message["status"]

    async def never_called(scope, receive, send):
        seen["reached_app"] = True

    scope = {
        "type": "http", "method": "POST", "path": "/runs/",
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", b"10000000")],
    }
    asyncio.run(limits.BodyLimitMiddleware(never_called)(scope, receive, send))
    assert seen.get("status") == 413
    assert "reached_app" not in seen, "oversized request reached the application"


def test_a_bounded_zip_bomb_does_not_grow_the_process():
    """The decompression cap must bound memory, not merely produce a 413.

    200 MiB of expansion against a 1 MiB cap. A status-code assertion cannot see the
    difference between refusing incrementally and materializing the whole expansion
    before refusing it; resident set can.
    """
    import resource

    app, _, _, _ = build({"ARGUS_MAX_BODY_BYTES": 4 * 1024 * 1024,
                          "ARGUS_MAX_DECOMPRESSED_BYTES": 1024 * 1024})

    # The bomb is built a megabyte at a time. `gzip.compress(b"0" * 200MB)` would make the
    # TEST allocate the 200 MiB it is trying to prove the SERVER does not allocate, and
    # since ru_maxrss is a high-water mark that never falls, the test would then measure
    # its own construction and pass or fail for the wrong reason. That is exactly what the
    # first version of this port did: it reported 199.7 MB of growth against a bound that
    # was working correctly.
    def streamed_bomb(n_bytes, chunk=1 << 20):
        co = zlib.compressobj(9, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        block = b"0" * chunk
        parts = [co.compress(block) for _ in range(n_bytes // chunk)]
        parts.append(co.flush())
        return b"".join(parts)

    body = streamed_bomb(200 * 1024 * 1024)

    def rss_mb():
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1048576.0 if sys.platform == "darwin" else r / 1024.0

    before = rss_mb()
    r = call(app, "POST", "/runs/", content=body,
             headers={"content-type": "application/json", "content-encoding": "gzip"})
    growth = rss_mb() - before
    assert r.status_code == 413, r.text
    assert growth < 64, f"process grew {growth:.1f} MB refusing a bounded bomb"


def test_an_environment_variable_can_only_tighten_the_ingest_bound(tmp_path):
    """Discriminator, replacing the one that read the refusal's byte count.

    The tailer carries its own per-poll caps. Wiring the environment in has one way
    to go wrong that nothing else here would catch: taking the variable's value
    outright, so `ARGUS_MAX_WORK_UNITS=50000` raises a 1,000-row cap to 50,000 and
    the bound is whatever a caller with environment access says it is. The default
    for that variable IS 50,000, so this is the configuration a plain deployment
    already has.
    """
    p = _write_metrics_file(tmp_path, 3_000)
    app, limits_mod, _, _ = build({"ARGUS_MAX_WORK_UNITS": 50_000}, metrics_file=str(p))
    assert limits_mod.MAX_WORK_UNITS == 50_000
    r = call(app, "POST", f"/runs/{RUN_ID}/metrics/sync")
    assert r.status_code == 200, r.text
    assert r.json()["inserted"] == 1_000, \
        "the environment loosened the tailer's own cap instead of tightening it"
