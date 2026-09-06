"""Observability that has to earn its cost.

Three claims are tested here, and the third is the only one that matters:

1. A run id and an attempt id survive from ingest, through detection, into recovery, so
   one filter reconstructs what happened to one training run across three processes.
2. Neither a metric value nor a configuration ever becomes a log or span attribute, and
   neither does a credential.
3. Metric label cardinality is bounded by construction. `run_id` as a label is one time
   series per training run forever, so `declare_counter` refuses it outright rather
   than trusting whoever writes the next call site.

And then the part a dashboard cannot do: a controlled run with one injected dependency
delay and one injected dependency failure, where the emitted records are required to
identify the cause. Counters that only ever go up prove nothing; the test asserts that
the record naming the failing dependency and the reason exists, that it carries the run
id, and that the same records under a healthy run do NOT contain it.
"""

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import obs  # noqa: E402
from agent.attempt import Attempt  # noqa: E402
from agent.detector import detect_anomalies  # noqa: E402

RUN_ID = "run-11111111"


@pytest.fixture(autouse=True)
def clean_metrics():
    obs.reset()
    yield
    obs.reset()


def write_run(path, steps=40, spike_at=None):
    """A metrics stream with an optional loss spike, in the trainer's own row shape."""
    rows = []
    for step in range(steps):
        loss = 1.0 / (step + 1) + 0.05
        if spike_at is not None and step >= spike_at:
            loss *= 12.0
        rows.append({"step": step, "epoch": 0, "train_loss": round(loss, 6),
                     "val_loss": round(loss * 1.1, 6), "val_acc": 0.5,
                     "grad_norm": 0.5, "timestamp": 1000.0 + step})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


# -- 1. correlation -------------------------------------------------------------
def test_the_run_id_reaches_every_stage_of_one_run(tmp_path, monkeypatch):
    """One filter over the records has to reconstruct one run across three processes."""
    metrics = tmp_path / "metrics.jsonl"
    write_run(metrics, steps=40, spike_at=38)

    sys.path.insert(0, str(ROOT / "backend"))
    import db

    sent = {}

    class _Exec:
        def execute(self):
            sent["called"] = True
            return type("R", (), {"data": []})()

    class _Table:
        def upsert(self, rows, on_conflict=None):
            return _Exec()

    monkeypatch.setattr(db, "get_client", lambda: type("C", (), {"table": lambda s, n: _Table()})())
    monkeypatch.setattr(db, "_tailers", {})

    with obs.capture() as records:
        # ingest
        db.insert_metrics(RUN_ID, str(metrics))
        # detection
        with obs.bind(run_id=RUN_ID):
            anomalies = detect_anomalies(str(metrics))
        # recovery
        a = Attempt(run_id=RUN_ID, trigger=anomalies[0], ledger_path=tmp_path / "led.jsonl")
        a.capture_baseline(str(metrics))
        with open(metrics, "a") as f:
            for step in range(100, 105):
                f.write(json.dumps({"step": step, "epoch": 1, "train_loss": 0.09,
                                    "val_loss": 0.1, "val_acc": 0.9, "grad_norm": 0.4,
                                    "timestamp": 2000.0 + step}) + "\n")
        a.observe(str(metrics))

    assert sent.get("called"), "the ingest never reached the sink"
    mine = [r for r in records if r.get("run_id") == RUN_ID]
    events = {r["event"] for r in mine}
    assert any(e.startswith("ingest.") for e in events), events
    assert any(e.startswith("detection.") for e in events), events
    assert any(e.startswith("recovery.") for e in events), events

    # The attempt id appears only on the recovery records, and it is the same one the
    # ledger holds, so a record and a ledger row can be joined after the fact.
    recovery = [r for r in mine if r["event"].startswith("recovery.")]
    assert recovery, mine
    assert {r["attempt_id"] for r in recovery} == {a.attempt_id}

    # Every record inside a span carries that span's id, so a slow stage is attributable
    # to the work it was doing rather than to the process as a whole.
    spans = {r["span_id"] for r in records if "span_id" in r}
    assert spans, "no span ids were emitted"


def test_records_outside_a_bind_carry_no_stale_identifiers():
    with obs.capture() as records:
        with obs.bind(run_id="a", attempt_id="b"):
            obs.log("inside")
        obs.log("outside")
    assert records[0]["run_id"] == "a" and records[0]["attempt_id"] == "b"
    assert "run_id" not in records[1] and "attempt_id" not in records[1]


def test_bind_refuses_anything_that_is_not_a_correlation_identifier():
    with pytest.raises(ValueError):
        with obs.bind(train_loss=0.5):
            pass


# -- 2. measurements and configuration never become attributes -----------------
@pytest.mark.parametrize("field", ["train_loss", "val_loss", "val_acc", "grad_norm",
                                   "learning_rate", "zscore", "batch_size"])
def test_a_measurement_never_becomes_an_attribute(field):
    with obs.capture() as records:
        obs.log("probe", **{field: 0.31337})
    assert records[0][field] == "<float>", records[0]
    assert "0.31337" not in json.dumps(records[0])


@pytest.mark.parametrize("field", ["config", "resulting_config", "patches", "headers"])
def test_configuration_contents_never_become_an_attribute(field):
    payload = {"training": {"learning_rate": 0.01}, "SUPABASE_KEY": "eyJabc.def.ghi"}
    with obs.capture() as records:
        obs.log("probe", **{field: payload})
    rendered = json.dumps(records[0])
    assert "learning_rate" not in rendered and "eyJabc" not in rendered, rendered
    assert records[0][field].startswith("<dict")


def test_an_undeclared_container_is_summarised_rather_than_inlined():
    with obs.capture() as records:
        obs.log("probe", something_new=[{"train_loss": 9.9}])
    assert records[0]["something_new"] == "<list len=1>"
    assert "9.9" not in json.dumps(records[0])


def test_the_detection_cycle_logs_the_type_and_never_the_values(tmp_path):
    metrics = tmp_path / "m.jsonl"
    write_run(metrics, steps=40, spike_at=38)
    with obs.capture() as records:
        anomalies = detect_anomalies(str(metrics))
    assert anomalies, "the fixture failed to produce an anomaly to log"
    lines = [r for r in records if r["event"] == "detection.anomaly"]
    assert lines and lines[0]["type"] == "loss_spike"
    rendered = json.dumps(records)
    # The detector's own result dict carries train_loss, a z-score and a rendered
    # description containing both. None of it may reach a record.
    for banned in ("zscore", "prev_loss", "curr_loss", "description"):
        assert banned not in rendered, f"{banned} reached the log: {rendered[:400]}"


# -- credentials ---------------------------------------------------------------
@pytest.mark.parametrize("secret", [
    "eyJhbGciOiJIUzI1NiJ9.eyJyZWYiOiJhYmMifQ.c2lnbmF0dXJl",  # credential-shape-fixture
    "sk-ant-api03-AAAABBBBCCCCDDDDEEEE",  # credential-shape-fixture
    "https://abcdefghijklmnop.supabase.co/rest/v1/metrics",  # credential-shape-fixture
])
def test_a_credential_shaped_value_is_redacted_from_a_record(secret):
    with obs.capture() as records:
        obs.log("probe", detail=f"upsert failed: {secret}")
    assert secret not in json.dumps(records[0]), records[0]
    assert obs.REDACTED in records[0]["detail"]


def test_a_live_environment_credential_is_redacted_by_exact_match(monkeypatch):
    monkeypatch.setenv("SUPABASE_KEY", "a-perfectly-ordinary-looking-value-42")
    with obs.capture() as records:
        obs.log("probe", detail="auth error for a-perfectly-ordinary-looking-value-42")
    assert "a-perfectly-ordinary-looking-value-42" not in json.dumps(records[0])


# -- 3. cardinality -------------------------------------------------------------
@pytest.mark.parametrize("label", ["run_id", "attempt_id", "step", "error_id",
                                   "metrics_file", "config_path", "url", "id"])
def test_an_identifier_is_refused_as_a_metric_label(label):
    """One series per run, per attempt or per request is one series per anything.

    This is the whole reason the ids ride on the log record instead. The refusal is at
    declaration time, so the bomb cannot be planted by a later call site.
    """
    with pytest.raises(obs.CardinalityError) as e:
        obs.declare_counter("probe_total", {label: ("a", "b")})
    assert "identifier" in str(e.value)


def test_a_label_must_declare_the_values_it_may_take():
    with pytest.raises(obs.CardinalityError):
        obs.declare_counter("probe2_total", {"outcome": "not a tuple"})


def test_every_declared_metric_has_a_series_count_known_in_advance():
    declared = obs.declared()
    assert declared, "no metrics are declared"
    total = 0
    for name, spec in declared.items():
        assert spec["max_series"] <= obs.MAX_SERIES_PER_METRIC, name
        total += spec["max_series"]
    # A single number for the whole registry, computable without running anything.
    assert total <= 128, f"registry ceiling is {total} series"


def test_a_declaration_that_would_exceed_the_cap_is_refused():
    with pytest.raises(obs.CardinalityError):
        obs.declare_counter("huge_total", {
            "a": tuple(str(i) for i in range(8)),
            "b": tuple(str(i) for i in range(8)),
        })


def test_an_undeclared_label_value_folds_into_one_bucket_instead_of_a_new_series():
    """A closed value set is only a bound if unrecognised values do not create series."""
    before = len(obs.snapshot()["argus_dependency_calls_total"])
    for i in range(50):
        obs.incr("argus_dependency_calls_total",
                 dependency="training_server", outcome=f"weird_{i}")
    after = obs.snapshot()["argus_dependency_calls_total"]
    assert len(after) - before == 1, after
    assert obs.read_counter("argus_dependency_calls_total",
                            dependency="training_server", outcome="anything_unknown") == 50


def test_incrementing_with_the_wrong_labels_is_an_error_not_a_new_series():
    with pytest.raises(obs.CardinalityError):
        obs.incr("argus_dependency_calls_total", dependency="training_server")
    with pytest.raises(obs.CardinalityError):
        obs.incr("never_declared_total", outcome="ok")


# -- 4. injected delay and injected failure ------------------------------------
INJECTED_DELAY_S = 0.25


def test_an_injected_dependency_delay_is_identified_by_the_records(tmp_path, monkeypatch):
    """The training server accepts the request and then does not answer in time.

    This is the real failure shape: agent/tools.py gives the rerun endpoint 10 seconds
    while a training run takes minutes, so a slow server is indistinguishable from a
    dead one unless the call site says which it was.
    """
    import requests

    from agent import tools

    def slow_post(*_a, **_k):
        time.sleep(INJECTED_DELAY_S)
        raise requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(requests, "post", slow_post)
    monkeypatch.setattr(tools, "resolve_within", lambda p: Path(p))

    with obs.capture() as records:
        with obs.bind(run_id=RUN_ID):
            result = tools.rerun_training(str(tmp_path), max_steps=50, attempt_id="att-1")

    assert result["status"] == "error"

    # The metric names the dependency and the reason, and nothing else moved.
    assert obs.read_counter("argus_dependency_calls_total",
                            dependency="training_server", outcome="timeout") == 1
    assert obs.read_counter("argus_dependency_calls_total",
                            dependency="training_server", outcome="ok") == 0

    failed = [r for r in records if r["event"] == "recovery.rerun_failed"]
    assert len(failed) == 1, records
    line = failed[0]
    assert line["dependency"] == "training_server"
    assert line["reason"] == "timeout"
    assert line["timeout_s"] == tools.REQUEST_TIMEOUT_S
    assert line["run_id"] == RUN_ID and line["attempt_id"] == "att-1"

    # The span puts a number on how long the dependency held the caller, which is what
    # separates "slow" from "refused instantly".
    ends = [r for r in records if r["event"] == "recovery.rerun_request.end"]
    assert len(ends) == 1
    assert ends[0]["duration_ms"] >= INJECTED_DELAY_S * 1000 * 0.9, ends[0]
    # And the span must not claim the call succeeded. rerun_training catches the
    # timeout and returns a result dict, so a span that inferred its outcome from
    # "did the block raise" reported outcome="ok" for a call that timed out. That is
    # a dashboard saying the system is healthy while the dependency is down.
    assert ends[0]["outcome"] == "timeout", ends[0]
    assert ends[0]["level"] == "error", ends[0]


def test_a_healthy_dependency_call_produces_no_failure_record(tmp_path, monkeypatch):
    """The negative half. Without this, the test above only proves a counter exists."""
    import requests

    from agent import tools

    class _Resp:
        def json(self):
            return {"status": "accepted"}

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    monkeypatch.setattr(tools, "resolve_within", lambda p: Path(p))

    with obs.capture() as records:
        with obs.bind(run_id=RUN_ID):
            result = tools.rerun_training(str(tmp_path), max_steps=50, attempt_id="att-2")

    assert result["status"] == "requested"
    assert not [r for r in records if r["event"] == "recovery.rerun_failed"]
    assert obs.read_counter("argus_dependency_calls_total",
                            dependency="training_server", outcome="ok") == 1
    assert obs.read_counter("argus_dependency_calls_total",
                            dependency="training_server", outcome="timeout") == 0
    ends = [r for r in records if r["event"] == "recovery.rerun_request.end"]
    assert len(ends) == 1 and ends[0]["outcome"] == "ok" and ends[0]["level"] == "info"


def test_an_injected_sink_failure_is_identified_and_the_cursor_is_not_advanced(tmp_path, monkeypatch):
    """Supabase rejects the batch. The records have to say which dependency and why.

    The cursor assertion is here because the classification is only useful if the
    system also did the right thing: a failed sink must replay the batch, not skip it.
    """
    sys.path.insert(0, str(ROOT / "backend"))
    import db

    metrics = tmp_path / "metrics.jsonl"
    write_run(metrics, steps=6)

    class _Boom:
        def execute(self):
            raise RuntimeError("401 Unauthorized: invalid apikey for project")

    class _Table:
        def upsert(self, rows, on_conflict=None):
            return _Boom()

    monkeypatch.setattr(db, "get_client",
                        lambda: type("C", (), {"table": lambda s, n: _Table()})())
    monkeypatch.setattr(db, "_tailers", {})

    with obs.capture() as records:
        with pytest.raises(RuntimeError):
            db.insert_metrics(RUN_ID, str(metrics))

    assert obs.read_counter("argus_ingest_failures_total",
                            dependency="supabase", reason="auth") == 1
    failed = [r for r in records if r["event"] == "ingest.sink_failed"]
    assert len(failed) == 1, records
    line = failed[0]
    assert line["dependency"] == "supabase"
    assert line["reason"] == "auth"
    assert line["run_id"] == RUN_ID
    assert line["cursor_committed"] is False
    assert line["batch_rows"] == 6

    # The batch really does replay: a healthy retry ingests the same six rows rather
    # than skipping them.
    accepted = []

    class _Ok:
        def __init__(self, rows):
            self.rows = rows

        def execute(self):
            accepted.extend(self.rows)
            return type("R", (), {"data": []})()

    monkeypatch.setattr(db, "get_client", lambda: type(
        "C", (), {"table": lambda s, n: type("T", (), {"upsert": lambda s2, rows, on_conflict=None: _Ok(rows)})()})())
    assert db.insert_metrics(RUN_ID, str(metrics)) == {"inserted": 6}
    assert [r["step"] for r in accepted] == list(range(6))


def test_a_missing_metrics_file_is_a_different_dependency_than_a_failing_sink(tmp_path, monkeypatch):
    """"Ingest is broken" is not an answer. Which of the two dependencies is."""
    sys.path.insert(0, str(ROOT / "backend"))
    import db

    monkeypatch.setattr(db, "get_client", lambda: type("C", (), {"table": lambda s, n: None})())
    monkeypatch.setattr(db, "_tailers", {})

    with obs.capture() as records:
        out = db.insert_metrics(RUN_ID, str(tmp_path / "absent.jsonl"))

    assert out == {"error": "metrics file not found"}
    assert obs.read_counter("argus_ingest_failures_total",
                            dependency="metrics_file", reason="connection") == 1
    assert obs.read_counter("argus_ingest_failures_total",
                            dependency="supabase", reason="auth") == 0
    assert [r["event"] for r in records if r["level"] == "error"] == ["ingest.metrics_file_missing"]
