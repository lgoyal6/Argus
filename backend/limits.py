"""Abuse limits and backpressure for the Argus API (C08).

The API had no bound of any kind. Every dimension the capability names was open:
a request body could be any size, a gzipped body was not accepted at all so its
decompressed size was undefined, `GET /runs/{id}/metrics` serialized however many
rows the run had emitted, `POST /runs/{id}/metrics/sync` read and upserted an
entire file off disk, nothing timed out, nothing queued and any number of requests
could be in flight at once.

Six bounds, one per dimension, each an environment variable where 0 disables it so
an operator can turn one off without editing code:

    ARGUS_MAX_BODY_BYTES          bytes on the wire
    ARGUS_MAX_DECOMPRESSED_BYTES  bytes after gunzip
    ARGUS_MAX_WORK_UNITS          rows a single request may read or ingest
    ARGUS_MAX_INGEST_BYTES        bytes one ingest poll may read off disk
    ARGUS_REQUEST_TIMEOUT_S       wall-clock runtime of one request
    ARGUS_MAX_QUEUE               requests waiting for a slot
    ARGUS_MAX_CONCURRENCY         requests executing at once
    ARGUS_RESERVED_LIGHT          slots heavy requests may never occupy

The last one is the difference between "rejects under overload" and "rejects under
overload without starving legitimate work". A plain semaphore plus a queue gives
you the first: a flood of expensive requests fills every slot, fills the queue, and
then a cheap legitimate request is refused along with the flood. It is bounded, it
is predictable, and the legitimate caller still gets nothing. Reserving slots that
the expensive class cannot take is what makes the second clause true, and the
overload experiment measures both configurations so the reservation is shown to be
the thing that carries it.

Requests are classified, not guessed at. A request is HEAVY if it writes, or if it
is one of the two per-run list reads whose cost grows with the length of the run.
Everything else is LIGHT: the health check, the run index, and a single run row.

One note on the ingest dimension, because it is the one place these bounds do not
work the way the list above suggests. The whole-file `readlines()` this was written
against no longer exists: backend/tailer.py reads only what was appended since the
last committed cursor, at most MAX_INGEST_BYTES and MAX_WORK_UNITS per poll, and
resumes on the next one. So an oversized metrics file is ingested in bounded pieces
rather than refused outright, which is a better answer for a run whose file
legitimately grows past any fixed cap, and the two variables above configure the
tailer instead of a refusal. `check_work` below still refuses on the READ side,
where truncation would silently drop half a chart.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import zlib

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


# Read at import so a test can set the environment and re-import, and so the values
# are visible to routes and to db.py without threading a settings object through
# every call site.
MAX_BODY_BYTES = _int_env("ARGUS_MAX_BODY_BYTES", 1 * 1024 * 1024)
MAX_DECOMPRESSED_BYTES = _int_env("ARGUS_MAX_DECOMPRESSED_BYTES", 8 * 1024 * 1024)
MAX_WORK_UNITS = _int_env("ARGUS_MAX_WORK_UNITS", 50_000)
MAX_INGEST_BYTES = _int_env("ARGUS_MAX_INGEST_BYTES", 32 * 1024 * 1024)
REQUEST_TIMEOUT_S = _float_env("ARGUS_REQUEST_TIMEOUT_S", 15.0)
MAX_QUEUE = _int_env("ARGUS_MAX_QUEUE", 16)
MAX_CONCURRENCY = _int_env("ARGUS_MAX_CONCURRENCY", 8)
RESERVED_LIGHT = _int_env("ARGUS_RESERVED_LIGHT", 2)


class WorkLimitExceeded(Exception):
    """A request asked for more units of work than one request is allowed."""

    def __init__(self, requested: int, limit: int, unit: str):
        self.requested = requested
        self.limit = limit
        self.unit = unit
        super().__init__(
            f"request needs {requested} {unit} but one request is limited to {limit}; "
            f"use limit and offset to page"
        )


def check_work(requested: int, unit: str = "rows") -> None:
    """Refuse rather than silently truncate.

    Truncating would be the quieter option and the wrong one. A chart that silently
    drops the second half of a run is a worse failure than a 413 that names the
    limit, because nobody notices it.
    """
    if MAX_WORK_UNITS and requested > MAX_WORK_UNITS:
        raise WorkLimitExceeded(requested, MAX_WORK_UNITS, unit)


# ── which requests are expensive ───────────────────────────────────────────────

_HEAVY_READ = re.compile(r"^/runs/[^/]+/(metrics|decisions)/?$")
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

HEAVY = "heavy"
LIGHT = "light"


def classify(method: str, path: str) -> str:
    if method.upper() not in _SAFE_METHODS:
        return HEAVY
    if _HEAVY_READ.match(path):
        return HEAVY
    return LIGHT


# ── plain ASGI responses, so a refusal costs nothing above the socket ──────────

async def _send_json(send: Send, status: int, body: bytes, extra_headers=()) -> None:
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def _err(detail: str) -> bytes:
    escaped = detail.replace("\\", "\\\\").replace('"', '\\"')
    return b'{"detail":"' + escaped.encode() + b'"}'


# ── bytes in, and bytes after gunzip ──────────────────────────────────────────


class BodyLimitMiddleware:
    """Bounds bytes on the wire and bytes after decompression.

    It sits outside admission control on purpose. An oversized body must be
    refused without ever occupying a concurrency slot, otherwise a flood of large
    bodies is an easier denial of service than a flood of real requests.

    The body is buffered here and replayed to the app rather than checked as it
    streams past. The first version refused mid-stream from inside the wrapped
    `receive`, which does not work: Starlette turns the resulting `http.disconnect`
    into `ClientDisconnect`, that propagates out of the route, and the server then
    tries to start a second response on a request whose 413 has already been sent.
    That was reproduced directly before this was rewritten. Buffering is safe
    precisely because the cap is the buffer: nothing over MAX_BODY_BYTES compressed
    or MAX_DECOMPRESSED_BYTES expanded is ever held.

    Content-Length is checked first because it is free, but it is not trusted: the
    buffered total is checked too, so a chunked body or a lying header hits the same
    limit. Decompression is incremental with a per-call output cap, so a zip bomb is
    refused as soon as the running total crosses the line rather than after it has
    been fully expanded in memory.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        encoding = headers.get("content-encoding", "").strip().lower()
        declared = headers.get("content-length")
        chunked = "chunked" in headers.get("transfer-encoding", "").lower()

        if MAX_BODY_BYTES and declared is not None:
            try:
                if int(declared) > MAX_BODY_BYTES:
                    await _send_json(
                        send, 413, _err(f"request body exceeds {MAX_BODY_BYTES} bytes")
                    )
                    return
            except ValueError:
                await _send_json(send, 400, _err("malformed content-length"))
                return

        if encoding and encoding not in ("identity", "gzip", "x-gzip", "deflate"):
            await _send_json(send, 415, _err(f"unsupported content-encoding: {encoding}"))
            return

        has_body = chunked or (declared is not None and declared != "0")
        if not has_body:
            await self.app(scope, receive, send)
            return

        try:
            body = await self._collect(receive, encoding)
        except _Refused as refusal:
            await _send_json(send, refusal.status, _err(refusal.detail))
            return

        if encoding in ("gzip", "x-gzip", "deflate"):
            # The body handed downstream is now plaintext, so the headers that
            # described the compressed form would be wrong. Correct them rather
            # than leaving a route to trust a stale length.
            mutable = MutableHeaders(scope=scope)
            del mutable["content-encoding"]
            mutable["content-length"] = str(len(body))

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if replayed:
                return {"type": "http.disconnect"}
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    async def _collect(self, receive: Receive, encoding: str) -> bytes:
        if encoding in ("gzip", "x-gzip"):
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif encoding == "deflate":
            decompressor = zlib.decompressobj(zlib.MAX_WBITS)
        else:
            decompressor = None

        compressed_seen = 0
        out = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunk = message.get("body", b"")
            compressed_seen += len(chunk)
            if MAX_BODY_BYTES and compressed_seen > MAX_BODY_BYTES:
                raise _Refused(413, f"request body exceeds {MAX_BODY_BYTES} bytes")

            if decompressor is None:
                out += chunk
            elif chunk:
                if MAX_DECOMPRESSED_BYTES:
                    remaining = MAX_DECOMPRESSED_BYTES - len(out)
                    # decompress(data, 0) means "no limit" in zlib, so a remaining
                    # budget of zero has to ask for one byte to force the overflow
                    # rather than silently accepting an unbounded expansion.
                    cap = remaining if remaining > 0 else 1
                else:
                    cap = 0
                try:
                    out += decompressor.decompress(chunk, cap)
                except zlib.error as exc:
                    raise _Refused(400, f"malformed {encoding} body: {exc}") from None
                if MAX_DECOMPRESSED_BYTES and (
                    len(out) > MAX_DECOMPRESSED_BYTES or decompressor.unconsumed_tail
                ):
                    raise _Refused(
                        413,
                        f"decompressed body exceeds {MAX_DECOMPRESSED_BYTES} bytes",
                    )

            if not message.get("more_body", False):
                break

        if decompressor is not None:
            try:
                tail = decompressor.flush()
            except zlib.error as exc:
                raise _Refused(400, f"malformed {encoding} body: {exc}") from None
            if MAX_DECOMPRESSED_BYTES and len(out) + len(tail) > MAX_DECOMPRESSED_BYTES:
                raise _Refused(
                    413, f"decompressed body exceeds {MAX_DECOMPRESSED_BYTES} bytes"
                )
            out += tail

        return bytes(out)


class _Refused(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(detail)


# ── runtime, queue growth, concurrency, and the reserved lane ─────────────────


class AdmissionMiddleware:
    """Bounds how many requests run at once, how many may wait, and how long one runs.

    Three numbers and one invariant. At most MAX_CONCURRENCY requests execute. At
    most MAX_QUEUE wait for a slot; the next one is refused immediately with 503 and
    a Retry-After rather than being allowed to grow the queue, because an unbounded
    queue converts a load problem into a memory problem and then into a latency
    problem, and the client has usually given up by the time its turn arrives. And
    heavy requests may occupy at most MAX_CONCURRENCY - RESERVED_LIGHT slots, so a
    heavy flood cannot take the last slot a cheap request needs.

    The timeout returns 504 to the caller but keeps the slot held until the handler
    actually finishes. Releasing early would make the concurrency number a lie:
    routes here are synchronous and run in a threadpool, and nothing in this process
    can preempt a running thread. Holding the slot is the honest choice; it means a
    slow handler still costs a slot, which is what is really true.
    """

    def __init__(self, app: ASGIApp):
        self.app = app
        self._lock: asyncio.Lock | None = None
        self._slot_free: asyncio.Condition | None = None
        self.in_flight = 0
        self.heavy_in_flight = 0
        self.waiting = 0
        # Counters, kept so the overload experiment reads offered and completed work
        # off the server rather than inferring it from the client's side.
        self.stats = {
            "offered": 0,
            "completed": 0,
            "refused_queue": 0,
            "refused_timeout": 0,
            "peak_in_flight": 0,
            "peak_waiting": 0,
        }

    def _ensure(self) -> asyncio.Condition:
        if self._slot_free is None:
            self._slot_free = asyncio.Condition()
        return self._slot_free

    def _cap_for(self, cls: str) -> int:
        if not MAX_CONCURRENCY:
            return 1 << 30
        if cls == HEAVY:
            return max(1, MAX_CONCURRENCY - RESERVED_LIGHT)
        return MAX_CONCURRENCY

    def _can_admit(self, cls: str) -> bool:
        if not MAX_CONCURRENCY:
            return True
        if self.in_flight >= MAX_CONCURRENCY:
            return False
        if cls == HEAVY and self.heavy_in_flight >= self._cap_for(HEAVY):
            return False
        return True

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        cls = classify(scope.get("method", "GET"), scope.get("path", "/"))
        cond = self._ensure()
        self.stats["offered"] += 1

        async with cond:
            if not self._can_admit(cls):
                if MAX_QUEUE and self.waiting >= MAX_QUEUE:
                    self.stats["refused_queue"] += 1
                    await _send_json(
                        send, 503,
                        _err("server is at capacity; retry shortly"),
                        extra_headers=[(b"retry-after", b"1")],
                    )
                    return
                self.waiting += 1
                self.stats["peak_waiting"] = max(self.stats["peak_waiting"], self.waiting)
                try:
                    await cond.wait_for(lambda: self._can_admit(cls))
                finally:
                    self.waiting -= 1
            self.in_flight += 1
            if cls == HEAVY:
                self.heavy_in_flight += 1
            self.stats["peak_in_flight"] = max(self.stats["peak_in_flight"], self.in_flight)

        released = False

        async def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            async with cond:
                self.in_flight -= 1
                if cls == HEAVY:
                    self.heavy_in_flight -= 1
                cond.notify_all()

        task = asyncio.ensure_future(self.app(scope, receive, send))
        try:
            if REQUEST_TIMEOUT_S:
                await asyncio.wait_for(asyncio.shield(task), REQUEST_TIMEOUT_S)
            else:
                await task
        except asyncio.TimeoutError:
            self.stats["refused_timeout"] += 1
            # The handler is still running. Hold the slot until it is not.
            task.add_done_callback(
                lambda _: asyncio.ensure_future(release())
            )
            try:
                await _send_json(send, 504, _err(f"request exceeded {REQUEST_TIMEOUT_S}s"))
            except RuntimeError:
                # The handler already started the response; nothing to add.
                pass
            return
        except BaseException:
            await release()
            raise
        self.stats["completed"] += 1
        await release()
