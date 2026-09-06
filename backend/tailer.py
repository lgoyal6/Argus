"""Incremental reader for the trainer's append-only JSONL metrics file.

The original ingest read the whole file and upserted every row on every poll. That
is O(total history) work per poll, so a run's ingest cost grows quadratically in its
own length: by step 10,000 each poll rewrites 10,000 rows to learn about a handful of
new ones. This reads only what is new.

Four things make that safe rather than merely faster:

  * File identity, not just a byte offset. A trainer that restarts and recreates the
    file gets a new inode; carrying the old offset over would silently skip the whole
    new run. Identity is (st_dev, st_ino).
  * Truncation detection. If the file is now shorter than the cursor, the offset is
    meaningless and reading resumes from zero.
  * Partial lines. A poll can land mid-write, in the middle of a JSON object. Only
    whole newline-terminated lines are emitted, and the cursor never advances past
    the last complete one - so the incomplete tail is simply re-read next poll
    rather than buffered in memory. Re-reading costs at most one line and keeps the
    cursor, not process memory, as the single source of truth about progress.
  * Commit after acknowledgement. read_batch() does NOT move the durable cursor;
    the caller calls commit() once the sink has accepted the rows. A crash between
    the two replays the batch, which the existing (run_id, step) upsert makes
    harmless.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

# A poll returns at most this many lines, so one enormous catch-up cannot produce an
# unbounded insert or hold the loop indefinitely.
DEFAULT_MAX_LINES = 1000
# and at most this many bytes, so a single pathological line cannot exhaust memory.
DEFAULT_MAX_BYTES = 8 * 1024 * 1024


@dataclass
class Cursor:
    dev: int = 0
    ino: int = 0
    offset: int = 0

    def matches(self, st: os.stat_result) -> bool:
        return self.dev == st.st_dev and self.ino == st.st_ino


class MetricsTailer:
    """Reads new whole lines from an append-only file across polls and restarts.

    state_path, when given, persists the cursor so a restarted backend resumes where
    it left off instead of re-ingesting the run from the beginning.
    """

    def __init__(self, metrics_file, state_path=None,
                 max_lines=DEFAULT_MAX_LINES, max_bytes=DEFAULT_MAX_BYTES):
        self.path = Path(metrics_file)
        self.state_path = Path(state_path) if state_path else None
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.cursor = self._load_cursor()
        self.stats = {"polls": 0, "lines_emitted": 0, "bytes_read": 0, "restarts": 0}

    # ── cursor persistence ────────────────────────────────────────────────────
    def _load_cursor(self) -> Cursor:
        if self.state_path and self.state_path.exists():
            try:
                return Cursor(**json.loads(self.state_path.read_text()))
            except (ValueError, TypeError):
                pass  # corrupt state: start over rather than crash the ingest loop
        return Cursor()

    def _save_cursor(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a crash mid-write must not leave a truncated cursor
        # that resumes from a wrong offset.
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self.cursor)))
        os.replace(tmp, self.state_path)

    # ── reading ───────────────────────────────────────────────────────────────
    def read_batch(self):
        """Return (entries, next_offset) for whole lines appended since the cursor.

        Does not advance the durable cursor - call commit(next_offset) after the
        sink has accepted the rows.
        """
        self.stats["polls"] += 1
        if not self.path.exists():
            return [], self.cursor.offset

        st = self.path.stat()
        start = self.cursor.offset
        if self.cursor.ino == 0:
            self.cursor = Cursor(st.st_dev, st.st_ino, 0)
            start = 0
        elif not self.cursor.matches(st):
            # Recreated file: a new run, not a continuation of the old one.
            self.stats["restarts"] += 1
            self.cursor = Cursor(st.st_dev, st.st_ino, 0)
            start = 0
        elif st.st_size < start:
            # Truncated in place; the offset no longer means anything.
            self.stats["restarts"] += 1
            start = 0

        if st.st_size == start:
            return [], start

        with open(self.path, "rb") as f:
            f.seek(start)
            chunk = f.read(self.max_bytes)
        self.stats["bytes_read"] += len(chunk)

        # Everything after the final newline is an incomplete record; leave the
        # cursor before it so the next poll re-reads it whole.
        cut = chunk.rfind(b"\n")
        if cut == -1:
            return [], start
        complete = chunk[:cut + 1]

        entries = []
        consumed = 0
        for raw in complete.splitlines(keepends=True):
            if len(entries) >= self.max_lines:
                break
            consumed += len(raw)
            line = raw.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except ValueError:
                # A corrupt line is skipped, not fatal: one bad record must not
                # wedge ingestion for the rest of the run.
                continue

        next_offset = start + consumed
        self.stats["lines_emitted"] += len(entries)
        return entries, next_offset

    def commit(self, offset: int) -> None:
        """Durably record that everything before `offset` has been accepted."""
        self.cursor.offset = offset
        self._save_cursor()
