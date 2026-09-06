"""Regenerate the committed API contract at backend/openapi.json.

FastAPI derives the OpenAPI document from the code, which makes it a description of the
code rather than a check on it: change a route and the document changes with it, so it
can never disagree and never catches anything. It only becomes a contract once a
generated copy is committed and reviewed, because then a change to a route shows up as
a diff on a reviewed file instead of as nothing at all.

    python backend/export_openapi.py          # rewrite the file
    python backend/export_openapi.py --check   # exit 1 if it is out of date

`tests/test_api_contract.py` compares the committed copy with the live app in both
directions, so an unreviewed route change fails the suite.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = HERE / "openapi.json"

# Load the app the way the image runs it: `COPY backend/ .` leaves no package, so
# main.py imports `from routes import runs`, not `from backend.routes import runs`.
# The repository root joins it only so that `import obs` resolves here the way it does
# in the image, where the Dockerfile copies obs.py in beside main.py.
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))


def build() -> dict:
    import main
    return json.loads(json.dumps(main.app.openapi()))


def main_cli() -> int:
    spec = build()
    rendered = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    if "--check" in sys.argv:
        if not SPEC.exists():
            print(f"{SPEC} does not exist; run this script without --check", file=sys.stderr)
            return 1
        if SPEC.read_text() != rendered:
            print(f"{SPEC} is out of date with the app; regenerate and review the diff",
                  file=sys.stderr)
            return 1
        print(f"{SPEC} matches the app")
        return 0
    SPEC.write_text(rendered)
    print(f"wrote {SPEC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
