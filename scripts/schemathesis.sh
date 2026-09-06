#!/bin/bash
# Run Schemathesis against the committed contract and a real, isolated Argus.
#
# The tests in tests/ drive the app in-process; this drives it over a socket, which is
# the only way to observe what Schemathesis observes - byte-level bodies, response
# headers, methods the router does not implement, and multi-request sequences. The
# server is tests/isolated_server.py, so the store is the in-memory fake from
# tests/conftest.py and supabase.create_client is poisoned: the generated corpus
# contains destructive bodies and must not be able to reach a real project.
#
#     scripts/schemathesis.sh            # the standard run
#     PORT=8899 EXAMPLES=1000 scripts/schemathesis.sh
#
# Three reports are expected and are not defects. Schemathesis treats an undeclared
# query parameter as a schema violation the API must reject; OpenAPI has no way to
# forbid one, and refusing every unknown query parameter would break any client that
# appends a cache-buster. They are left standing rather than silenced so that the next
# person sees the same output this one did.
set -euo pipefail

PORT="${PORT:-8731}"
EXAMPLES="${EXAMPLES:-300}"
SEED="${SEED:-20260905}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"

python "$ROOT/tests/isolated_server.py" --port "$PORT" --tmp "$TMP" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; rm -rf "$TMP"' EXIT

for _ in $(seq 1 40); do
    curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1 && break
    sleep 0.25
done

# uvx keeps the tool out of the app's own dependency set: Schemathesis is a client, it
# has no business in the image, and pinning the version here is what makes two runs
# comparable.
uvx schemathesis@4.25.2 run "$ROOT/backend/openapi.json" \
    --url "http://127.0.0.1:$PORT" \
    --checks all \
    --phases examples,coverage,fuzzing,stateful \
    -n "$EXAMPLES" \
    --seed "$SEED" \
    --continue-on-failure \
    "$@"
