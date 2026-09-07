"""One place that decides what an API failure tells the caller.

Every route used to end in:

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

which relays whatever text the exception happened to carry straight into the response
body. That is an unconditional disclosure channel: nobody chooses what goes through it
and nobody reviews what comes out.

Measured, not assumed. Two things were observed reaching a client through it: the
string `'anomalies'`, an internal field name, from the KeyError that made
POST /runs/{id}/decisions return 500 on every valid request; and absolute host
filesystem paths from the ingest path. Supabase 2.31.0 was checked directly and its
own exception messages carry neither the project URL nor the key - `create_client` with
a bad URL says only "Invalid URL" - so the client library is not itself the leak. It
does hold the key in `postgrest.session.headers` under both `apikey` and
`authorization`, which is why obs.scrub_text still runs over everything logged: the
disclosure risk there is any future code that logs a request or session object, not the
exception text as it stands today.

The replacement keeps the operator's half and drops the caller's half. The caller gets
a stable shape and a correlation id; the full exception goes to the process log next to
the same id, where it is useful and not public.
"""

import uuid

from fastapi import HTTPException
from pydantic import BaseModel

# obs.py sits at the repository root and is copied to /app alongside the flat backend
# and agent modules by both Dockerfiles, so this one plain import resolves identically
# in the image and from the repository root. No try/except is needed here precisely
# because the module is not inside either package.
import obs


class ErrorResponse(BaseModel):
    """The single shape every declared error response uses."""

    detail: str
    error_id: str | None = None


class InternalError(HTTPException):
    """A 500 whose wire body is the shape `ErrorResponse` declares.

    `HTTPException(500, detail={...})` looks right in the source and is wrong on the
    wire: FastAPI's default handler renders any HTTPException as `{"detail": <detail>}`,
    so a dict detail arrives nested as `{"detail": {"detail": ..., "error_id": ...}}`.
    The committed contract declares `detail` to be a string and `error_id` to sit
    beside it, so every 500 the service returned violated its own published schema and
    a client following the document looked for `error_id` at the top level and never
    found it. Schemathesis found this by validating error bodies against the spec, which
    is the one thing the hand-rolled corpus never did.

    `main.py` registers a handler for this class that writes the flat shape.
    """

    def __init__(self, error_id: str):
        super().__init__(status_code=500, detail="internal error")
        self.error_id = error_id


def not_found(what: str = "run not found") -> HTTPException:
    return HTTPException(status_code=404, detail=what)


def conflict(what: str) -> HTTPException:
    return HTTPException(status_code=409, detail=what)


def bad_request(what: str) -> HTTPException:
    return HTTPException(status_code=400, detail=what)


def internal_error(exc: BaseException, operation: str) -> HTTPException:
    """A 500 that says an operation failed without saying how.

    The correlation id is the only thing shared between the response and the log, so
    an operator can find the real cause from a report that contains nothing sensitive.
    """
    error_id = uuid.uuid4().hex[:12]
    # error_type only. The exception's message is the part that carries URLs, paths
    # and, from some clients, credentials, so it goes to the log body and never into
    # a span or metric attribute.
    obs.log(
        "api.request_failed",
        level="error",
        operation=operation,
        error_id=error_id,
        error_type=type(exc).__name__,
        detail=str(exc),
    )
    return InternalError(error_id)


# Declared on the routes so the committed OpenAPI document describes the failure shapes
# as well as the success ones. tests/test_api_generated_inputs.py asserts the app never
# answers with a status its own contract does not declare, so these have to be accurate
# rather than generous: an over-declared 404 is a promise no caller can rely on.
_SERVER = {500: {"model": ErrorResponse,
                 "description": "the request failed inside the service"}}
_NOT_FOUND = {404: {"model": ErrorResponse,
                    "description": "the addressed run does not exist"}}
_BAD_REQUEST = {400: {"model": ErrorResponse,
                      "description": "the request was well formed but not permitted"}}
_CONFLICT = {409: {"model": ErrorResponse,
                   "description": "the path and the body address different runs"}}
# The run exists; the file its row points at does not. This was a 404 with a different
# detail string from the 404 for an unknown run, which made the refusal an existence
# oracle: POST the sync route at an id and the body tells you whether that run is real.
# Splitting the two conditions by status is what lets 404 mean exactly one thing and
# carry exactly one body.
_MISSING_FILE = {409: {"model": ErrorResponse,
                       "description": "the run exists but its metrics file is not readable"}}
# Starlette answers 400, not 422, when the request body is not decodable at all - bytes
# that are not valid UTF-8 never reach pydantic, so there is no validation error to
# report and the body is rejected before the route is entered. The routes that take a
# body therefore have a reachable 400 that the document did not declare. Every
# hand-written malformed body in tests/test_api_generated_inputs.py is valid UTF-8 and
# lands on 422, which is why the corpus never saw this and Schemathesis did.
_UNPARSEABLE = {400: {"model": ErrorResponse,
                      "description": "the request body could not be decoded"}}

# Every read whose cost grows with the data can refuse instead of truncating. The
# status is declared on those routes rather than handled globally and left undeclared,
# because tests/test_api_generated_inputs.py holds the app to answering only statuses
# its own contract names.
_WORK_LIMIT = {413: {"model": ErrorResponse,
                     "description": "the request asked for more rows than one request may read"}}

# The run index: does not resolve a run, and pages, so it can refuse.
COLLECTION_READ_RESPONSES = {**_WORK_LIMIT, **_SERVER}
# Routes that do not resolve a run but do take a body.
COLLECTION_WRITE_RESPONSES = {**_UNPARSEABLE, **_SERVER}
# Routes scoped to one run.
ERROR_RESPONSES = {**_NOT_FOUND, **_SERVER}
# Run-scoped reads that page.
PAGED_READ_RESPONSES = {**_NOT_FOUND, **_WORK_LIMIT, **_SERVER}
# PATCH /runs/{id}/status also rejects a status outside the permitted set.
STATUS_RESPONSES = {**_BAD_REQUEST, **_NOT_FOUND, **_SERVER}
# POST /runs/{id}/decisions also rejects a body that names a different run.
DECISION_WRITE_RESPONSES = {**_UNPARSEABLE, **_NOT_FOUND, **_CONFLICT, **_SERVER}
# POST /runs/{id}/metrics/sync also refuses a run whose metrics file is unreadable.
SYNC_RESPONSES = {**_NOT_FOUND, **_MISSING_FILE, **_SERVER}
