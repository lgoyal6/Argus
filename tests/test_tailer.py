"""Incremental metrics ingestion.

The property that matters is the one the old whole-file read got wrong: work per
poll must be proportional to what is NEW, not to the run's total length.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.tailer import MetricsTailer


def append(path, *entries):
    with open(path, "a") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def row(step):
    return {"step": step, "train_loss": 1.0 / (step + 1), "val_loss": 1.0, "grad_norm": 0.5}


def test_only_new_lines_are_returned(tmp_path):
    m = tmp_path / "metrics.jsonl"
    t = MetricsTailer(m, state_path=tmp_path / "cursor.json")

    append(m, row(0), row(1), row(2))
    entries, off = t.read_batch()
    assert [e["step"] for e in entries] == [0, 1, 2]
    t.commit(off)

    # No new data: a poll must return nothing and read nothing.
    before = t.stats["bytes_read"]
    entries, off = t.read_batch()
    assert entries == []
    assert t.stats["bytes_read"] == before, "re-read the file despite no new data"
    t.commit(off)

    append(m, row(3))
    entries, off = t.read_batch()
    assert [e["step"] for e in entries] == [3]
    t.commit(off)


def test_work_per_poll_does_not_grow_with_history(tmp_path):
    """The quadratic-ingest regression test.

    Append one row at a time for 200 steps. Under the old whole-file read the rows
    written per poll would climb 1, 2, 3, ... 200 (20,100 writes total). Here every
    poll must emit exactly the one new row.
    """
    m = tmp_path / "metrics.jsonl"
    t = MetricsTailer(m, state_path=tmp_path / "cursor.json")
    emitted_per_poll = []
    for step in range(200):
        append(m, row(step))
        entries, off = t.read_batch()
        t.commit(off)
        emitted_per_poll.append(len(entries))

    assert emitted_per_poll == [1] * 200
    assert t.stats["lines_emitted"] == 200, (
        f"wrote {t.stats['lines_emitted']} rows to ingest 200 - "
        "cost is still proportional to history"
    )


def test_a_partial_line_is_not_parsed_until_complete(tmp_path):
    m = tmp_path / "metrics.jsonl"
    t = MetricsTailer(m, state_path=tmp_path / "cursor.json")

    # Poll lands mid-write, halfway through a JSON object.
    with open(m, "w") as f:
        f.write('{"step": 0, "train_lo')
    entries, off = t.read_batch()
    assert entries == [], "emitted a half-written record"
    t.commit(off)

    with open(m, "a") as f:
        f.write('ss": 1.0}\n')
    entries, off = t.read_batch()
    assert [e["step"] for e in entries] == [0], "lost the record once it completed"
    t.commit(off)


def test_rotation_restarts_from_the_new_file(tmp_path):
    m = tmp_path / "metrics.jsonl"
    t = MetricsTailer(m, state_path=tmp_path / "cursor.json")
    append(m, row(0), row(1), row(2), row(3), row(4))
    _, off = t.read_batch()
    t.commit(off)

    # Trainer restarts and recreates the file. Carrying the old offset would skip
    # the whole new run.
    os.remove(m)
    append(m, row(0), row(1))
    entries, off = t.read_batch()
    assert [e["step"] for e in entries] == [0, 1], "missed the recreated run"
    assert t.stats["restarts"] == 1


def test_truncation_in_place_restarts(tmp_path):
    m = tmp_path / "metrics.jsonl"
    t = MetricsTailer(m, state_path=tmp_path / "cursor.json")
    append(m, *[row(i) for i in range(10)])
    _, off = t.read_batch()
    t.commit(off)

    with open(m, "w") as f:  # truncate, same inode
        f.write(json.dumps(row(0)) + "\n")
    entries, _ = t.read_batch()
    assert [e["step"] for e in entries] == [0]
    assert t.stats["restarts"] == 1


def test_cursor_survives_a_restart_of_the_backend(tmp_path):
    m = tmp_path / "metrics.jsonl"
    state = tmp_path / "cursor.json"
    t = MetricsTailer(m, state_path=state)
    append(m, row(0), row(1))
    _, off = t.read_batch()
    t.commit(off)

    # A brand-new tailer, as after a process restart.
    t2 = MetricsTailer(m, state_path=state)
    entries, off = t2.read_batch()
    assert entries == [], "re-ingested the whole run after a restart"
    append(m, row(2))
    entries, off = t2.read_batch()
    assert [e["step"] for e in entries] == [2]


def test_uncommitted_batch_is_replayed_after_a_crash(tmp_path):
    """A crash between reading and the sink accepting must not lose rows.

    read_batch does not advance the durable cursor, so an uncommitted batch comes
    back. The existing (run_id, step) upsert makes the replay harmless.
    """
    m = tmp_path / "metrics.jsonl"
    state = tmp_path / "cursor.json"
    t = MetricsTailer(m, state_path=state)
    append(m, row(0), row(1))
    entries, _ = t.read_batch()          # sink never acknowledged
    assert len(entries) == 2

    t2 = MetricsTailer(m, state_path=state)
    entries, off = t2.read_batch()
    assert [e["step"] for e in entries] == [0, 1], "rows lost across the crash"
    t2.commit(off)


def test_batches_are_bounded(tmp_path):
    m = tmp_path / "metrics.jsonl"
    t = MetricsTailer(m, state_path=tmp_path / "cursor.json", max_lines=10)
    append(m, *[row(i) for i in range(100)])

    entries, off = t.read_batch()
    assert len(entries) == 10, "an unbounded catch-up batch"
    t.commit(off)
    # and the rest arrives on subsequent polls, in order, with nothing skipped.
    seen = [e["step"] for e in entries]
    for _ in range(20):
        entries, off = t.read_batch()
        t.commit(off)
        seen.extend(e["step"] for e in entries)
        if len(seen) >= 100:
            break
    assert seen == list(range(100)), "bounded batching dropped or reordered rows"


def test_a_corrupt_line_does_not_wedge_ingestion(tmp_path):
    m = tmp_path / "metrics.jsonl"
    t = MetricsTailer(m, state_path=tmp_path / "cursor.json")
    with open(m, "w") as f:
        f.write(json.dumps(row(0)) + "\n")
        f.write("{not valid json at all\n")
        f.write(json.dumps(row(2)) + "\n")
    entries, off = t.read_batch()
    assert [e["step"] for e in entries] == [0, 2]
    t.commit(off)
    append(m, row(3))
    entries, _ = t.read_batch()
    assert [e["step"] for e in entries] == [3]


def test_missing_file_is_not_an_error(tmp_path):
    t = MetricsTailer(tmp_path / "nope.jsonl", state_path=tmp_path / "c.json")
    assert t.read_batch() == ([], 0)
