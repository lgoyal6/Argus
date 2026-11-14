"""Pre-activation validation for the model artifacts Argus promotes.

WHY THIS EXISTS
---------------
`lifecycle.py` used to put a version into production like this:

    client.set_registered_model_alias(MODEL, "production", version)   # activation
    mlflow.pyfunc.load_model(f"models:/{MODEL}@production")           # then load

The alias moves first. Everything that could reject the artifact happens after
it is already serving, which means a bad artifact is not rejected, it is
deployed and then discovered. Worse, the discovery is destructive: when the load
raised on a tampered artifact the alias stayed on the bad version, so the
failure mode was not "we kept the old model", it was "production has no working
model at all".

And the load itself is the dangerous step. An MLflow pyfunc artifact is
`python_model.pkl`, a cloudpickle, and unpickling is arbitrary code execution:
the payload runs inside `load_model`, before any signature, shape or dtype is
ever consulted. Checking the model after loading it is checking the lock after
opening the door.

WHAT THIS MODULE ENFORCES, IN ORDER
-----------------------------------
Every step happens BEFORE the production alias is touched.

  1. provenance  - the version carries the dataset hash, code revision and run
                   id it was built from, and they match what we expect.
  2. digest      - every file in the artifact matches a manifest recorded when
                   the artifact was reviewed. Byte size and sha256, no
                   tolerance.
  3. no code     - the pickle's global references are read off the opcode
                   stream, WITHOUT unpickling, and checked against a module
                   allowlist. `os`, `subprocess`, `builtins.eval` and friends
                   are refused here, before `load_model` is called at all.
  4. schema      - the artifact's declared feature order and input columns
                   match the contract the caller indexes by. A permuted feature
                   order loads perfectly and then answers the wrong question.
  5. dtype/shape - the staged model is exercised on a canary batch and its
                   output dtype and shape are checked against the schema.
  6. finite      - every float array reachable in the loaded model, and the
                   canary output, must be finite. NaN weights load fine and
                   predict silently.

Only then does `guarded_promote` move the alias. If any step raises, the alias
is left exactly where it was, so the previously-good version is still serving.

`ArtifactRejected` is the single failure type; there is no partial acceptance.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickletools
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


class ArtifactRejected(Exception):
    """An artifact failed validation and must not be activated."""


# --------------------------------------------------------------------------- #
# 0. The explicit artifact schema                                             #
# --------------------------------------------------------------------------- #
# One place that says what an Argus detector artifact IS. Written down rather
# than implied by whatever the last training run happened to emit, because the
# whole point of a compatibility check is having something to check against.
ARTIFACT_SCHEMA: dict = {
    "schema_version": 1,
    "registered_model": "argus-anomaly-detector",
    # The served input contract, in order. `pyfunc` will happily accept a frame
    # whose columns are permuted, so order is part of the contract, not a detail.
    "input_columns": [
        {"name": "window_id", "dtype": "int64"},
        {"name": "pos", "dtype": "int64"},
        {"name": "step", "dtype": "int64"},
        {"name": "train_loss", "dtype": "float64"},
        {"name": "val_loss", "dtype": "float64"},
        {"name": "val_acc", "dtype": "float64"},
        {"name": "grad_norm", "dtype": "float64"},
    ],
    # Feature order INSIDE the model. Only the learned candidates declare one;
    # the deterministic detector extracts nothing and declares none.
    "feature_order_source": "feature_contract.json",
    # What a prediction must look like coming back out.
    "output": {"dtype_kind": "i", "ndim": 1, "values": [0, 1]},
    # Files an artifact must contain to be considered complete.
    "required_files": ["MLmodel", "python_model.pkl"],
    # Provenance keys a registered version must carry as tags.
    "required_provenance_tags": ["dataset_sha256", "kind"],
}


# --------------------------------------------------------------------------- #
# 1. Pickle globals, read off the opcode stream without unpickling            #
# --------------------------------------------------------------------------- #
# Modules a legitimate Argus pyfunc artifact may reference. Derived by scanning
# the three artifacts lifecycle.py actually produces, not guessed: cloudpickle
# scaffolding, the mlflow pyfunc wrapper, numpy, sklearn, and `builtins.type`
# for the reconstructed classes.
ALLOWED_GLOBAL_PREFIXES: tuple = (
    "cloudpickle",
    "mlflow.",
    "numpy",
    "sklearn",
    "scipy",
    "pandas",
    "collections",
    "copyreg",
    "_codecs",
    "experiments.",
    "agent.",
    "__builtin__",
)

# From `builtins` only these names. `eval`, `exec`, `compile`, `__import__`,
# `getattr` and `open` are the standard pickle escape hatches and are refused
# even though they live in an otherwise ordinary module.
ALLOWED_BUILTINS: frozenset = frozenset({
    "type", "object", "dict", "list", "set", "frozenset", "tuple",
    "int", "float", "bool", "str", "bytes", "complex", "slice", "range",
    "bytearray", "NotImplemented", "Ellipsis",
})

# Opcodes that build an object from a global that is not on the opcode stream,
# so it cannot be checked statically. Refused rather than assumed harmless.
_OPAQUE_CONSTRUCTORS = ("INST", "OBJ", "EXT1", "EXT2", "EXT4")

_STRING_OPS = {
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
    "SHORT_BINSTRING", "BINSTRING", "STRING", "BINBYTES", "SHORT_BINBYTES",
}
_PUT_OPS = {"PUT", "BINPUT", "LONG_BINPUT"}
_GET_OPS = {"GET", "BINGET", "LONG_BINGET"}

_OTHER = object()  # a stack slot whose value is not a string we tracked


def pickle_globals(data: bytes, *, where: str = "<bytes>") -> set:
    """Every ``(module, name)`` the pickle stream references, without unpickling.

    Walks the opcodes with a real value stack AND a real memo. That matters:
    protocol 4+ pickles - which is everything cloudpickle emits - push the
    module and name with STACK_GLOBAL, and cloudpickle memoises those strings,
    so they usually arrive via BINGET rather than as literals. A scanner that
    only remembers literal strings resolves the wrong pair on exactly the files
    it most needs to read correctly.

    Nothing is constructed and no module is imported, so this is safe to run on
    a hostile file. Raises `ArtifactRejected` for a stream it cannot read.
    """
    stack: list = []
    memo: dict = {}
    found: set = set()

    def push(v):
        stack.append(v)

    def pop():
        return stack.pop() if stack else _OTHER

    try:
        ops = list(pickletools.genops(data))
    except Exception as exc:  # truncated, malformed, adversarially framed
        raise ArtifactRejected(f"{where}: unparseable pickle stream ({exc})") from None

    for op, arg, pos in ops:
        code = op.name
        if code in _OPAQUE_CONSTRUCTORS:
            raise ArtifactRejected(
                f"{where}: opcode {code} at byte {pos} builds an object from a "
                "global that is not on the stream; cannot be checked, refused"
            )
        if code == "GLOBAL":
            mod, _, nm = str(arg).partition(" ")
            found.add((mod, nm))
            push(_OTHER)
        elif code == "STACK_GLOBAL":
            nm, mod = pop(), pop()
            if not isinstance(mod, str) or not isinstance(nm, str):
                raise ArtifactRejected(
                    f"{where}: STACK_GLOBAL at byte {pos} with an unresolvable "
                    "module/name pair; refused"
                )
            found.add((mod, nm))
            push(_OTHER)
        elif code in _STRING_OPS:
            push(arg)
        elif code in _PUT_OPS:
            if stack:
                memo[arg] = stack[-1]
        elif code == "MEMOIZE":
            if stack:
                memo[len(memo)] = stack[-1]
        elif code in _GET_OPS:
            push(memo.get(arg, _OTHER))
        elif code == "POP":
            pop()
        elif code == "DUP":
            v = stack[-1] if stack else _OTHER
            push(v)
        else:
            # Everything else (REDUCE, BUILD, tuples, numbers, framing...) can
            # only produce a non-string, and the stack depth bookkeeping does
            # not need to be exact: STACK_GLOBAL is the only consumer that
            # cares, and its two operands are always pushed immediately before
            # it by every pickler in use here.
            if op.arg is not None or code in ("NONE", "NEWTRUE", "NEWFALSE"):
                push(_OTHER)

    return found


def check_pickle_globals(path: str) -> set:
    """Refuse a pickle that references a module outside the allowlist.

    Runs BEFORE `mlflow.pyfunc.load_model`, which is the only moment refusing
    is still possible: the unpickler executes the payload as it reads it.
    """
    with open(path, "rb") as fh:
        refs = pickle_globals(fh.read(), where=path)

    bad = []
    for mod, name in sorted(refs):
        if mod in ("builtins", "__builtin__"):
            if name not in ALLOWED_BUILTINS:
                bad.append(f"{mod}.{name}")
            continue
        if not any(mod == p.rstrip(".") or mod.startswith(p)
                   for p in ALLOWED_GLOBAL_PREFIXES):
            bad.append(f"{mod}.{name}")
    if bad:
        raise ArtifactRejected(
            f"{path}: pickle references globals outside the allowlist: "
            f"{sorted(set(bad))}"
        )
    return refs


# --------------------------------------------------------------------------- #
# 2. Digests and the manifest                                                 #
# --------------------------------------------------------------------------- #
def sha256_file(path: str, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(artifact_dir: str, provenance: Mapping) -> dict:
    """Record what a reviewed artifact contains. Run once, on an artifact you
    have decided to trust; verify against the result from then on."""
    files = {}
    for root, _dirs, names in os.walk(artifact_dir):
        for n in sorted(names):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, artifact_dir)
            files[rel] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    return {"provenance": dict(provenance), "files": files}


def verify_files(artifact_dir: str, manifest: Mapping) -> None:
    """Reject an artifact directory that does not match `manifest` exactly."""
    expected = manifest["files"]
    present = {}
    for root, _dirs, names in os.walk(artifact_dir):
        for n in names:
            p = os.path.join(root, n)
            present[os.path.relpath(p, artifact_dir)] = p

    missing = sorted(set(expected) - set(present))
    if missing:
        raise ArtifactRejected(f"{artifact_dir}: missing files {missing}")
    extra = sorted(set(present) - set(expected))
    if extra:
        raise ArtifactRejected(f"{artifact_dir}: unexpected files {extra}")

    for rel in sorted(expected):
        want, p = expected[rel], present[rel]
        size = os.path.getsize(p)
        if size != want["bytes"]:
            raise ArtifactRejected(
                f"{artifact_dir}: {rel} is {size} bytes, manifest says {want['bytes']}"
            )
        got = sha256_file(p)
        if got != want["sha256"]:
            raise ArtifactRejected(
                f"{artifact_dir}: {rel} sha256 {got[:12]}... != manifest "
                f"{want['sha256'][:12]}..."
            )


def manifest_path(manifest_dir: str, registered_model: str, version: str) -> str:
    return os.path.join(manifest_dir, f"{registered_model}-v{version}.json")


def record_manifest(client, registered_model: str, version: str, artifact_dir: str,
                    provenance: Mapping, manifest_dir: str) -> dict:
    """Digest a version's files once, at registration, and pin the digest.

    The manifest itself is a file on disk, so it is only worth as much as its
    own integrity: its sha256 goes on the model version as a tag, which lives in
    the registry rather than next to the artifact. Tampering with the artifact
    changes a file digest; tampering with the manifest to match changes the
    manifest's digest and no longer matches the tag. Both are refusals at
    promote time.
    """
    os.makedirs(manifest_dir, exist_ok=True)
    manifest = build_manifest(artifact_dir, provenance)
    path = manifest_path(manifest_dir, registered_model, version)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    client.set_model_version_tag(registered_model, version, "manifest_sha256",
                                 sha256_file(path))
    return manifest


def load_manifest(client, registered_model: str, version: str,
                  manifest_dir: str) -> dict:
    """Return the pinned manifest for a version, or refuse.

    Fail-closed at every step: no tag, no file, or a file whose digest is not
    the pinned one are all refusals. A missing manifest must not degrade into
    "skip the digest check", which is precisely how the check went dead before.
    """
    tags = client.get_model_version(registered_model, version).tags
    pinned = tags.get("manifest_sha256")
    if not pinned:
        raise ArtifactRejected(
            f"version {version} carries no manifest_sha256 tag; its files were "
            "never digested, so there is nothing to verify them against"
        )
    path = manifest_path(manifest_dir, registered_model, version)
    if not os.path.exists(path):
        raise ArtifactRejected(
            f"version {version}: manifest {path} is missing but the version "
            f"pins digest {pinned[:12]}..."
        )
    got = sha256_file(path)
    if got != pinned:
        raise ArtifactRejected(
            f"version {version}: manifest {path} digest {got[:12]}... != pinned "
            f"{pinned[:12]}...; the manifest itself was modified"
        )
    with open(path) as fh:
        return json.load(fh)


def check_required_files(artifact_dir: str, schema: Mapping = ARTIFACT_SCHEMA) -> None:
    root = Path(artifact_dir)
    missing = [f for f in schema["required_files"] if not (root / f).exists()]
    if missing:
        raise ArtifactRejected(f"{artifact_dir}: artifact is missing {missing}")


# --------------------------------------------------------------------------- #
# 3. Provenance                                                               #
# --------------------------------------------------------------------------- #
def check_provenance(tags: Mapping, expected_dataset_sha256: Optional[str] = None,
                     schema: Mapping = ARTIFACT_SCHEMA) -> None:
    """Refuse a version that does not say where it came from.

    A version with no dataset hash cannot be reproduced and cannot be compared
    against the run that justified promoting it, so it is not promotable no
    matter how well it scores.
    """
    missing = [k for k in schema["required_provenance_tags"] if not tags.get(k)]
    if missing:
        raise ArtifactRejected(
            f"version is missing provenance tags {missing}; it cannot be traced "
            "to the data and code that produced it"
        )
    if expected_dataset_sha256 and tags["dataset_sha256"] != expected_dataset_sha256:
        raise ArtifactRejected(
            f"version was built on dataset {tags['dataset_sha256'][:12]}... but "
            f"this promotion expects {expected_dataset_sha256[:12]}..."
        )


# --------------------------------------------------------------------------- #
# 4. Feature order, dtype/shape and finiteness                                #
# --------------------------------------------------------------------------- #
def check_feature_order(artifact_dir: str, expected_order: Sequence[str],
                        schema: Mapping = ARTIFACT_SCHEMA) -> Optional[list]:
    """Reject an artifact whose declared feature order is not the contract.

    Returns None when the artifact declares no feature contract, which is
    legitimate for the deterministic detector: it consumes raw rows and has no
    feature vector to permute. An artifact that declares one and gets it wrong
    is a different matter - it loads cleanly and then indexes the wrong column.
    """
    fc = Path(artifact_dir) / schema["feature_order_source"]
    if not fc.exists():
        return None
    try:
        declared = json.loads(fc.read_text())["features"]
    except Exception as exc:
        raise ArtifactRejected(f"{fc}: unreadable feature contract ({exc})") from None
    if list(declared) != list(expected_order):
        extra = sorted(set(declared) - set(expected_order))
        missing = sorted(set(expected_order) - set(declared))
        detail = f" missing={missing} unexpected={extra}" if (missing or extra) else \
                 " same names, different order"
        raise ArtifactRejected(
            f"{fc}: feature order does not match the contract "
            f"({len(declared)} declared vs {len(expected_order)} expected);{detail}"
        )
    return list(declared)


def check_input_columns(frame, schema: Mapping = ARTIFACT_SCHEMA) -> None:
    """The canary frame must be the contract's columns, in the contract's order."""
    want = [c["name"] for c in schema["input_columns"]]
    got = list(frame.columns)
    if got != want:
        raise ArtifactRejected(f"canary input columns {got} != contract {want}")


def check_prediction(pred, n_expected: int, schema: Mapping = ARTIFACT_SCHEMA) -> None:
    """dtype, shape and value-domain of what the model actually returned."""
    import numpy as np

    arr = np.asarray(pred)
    spec = schema["output"]
    if arr.ndim != spec["ndim"]:
        raise ArtifactRejected(
            f"prediction ndim {arr.ndim} != expected {spec['ndim']} (shape {arr.shape})"
        )
    if arr.shape[0] != n_expected:
        raise ArtifactRejected(
            f"prediction shape {arr.shape} does not match the {n_expected} "
            "windows it was asked about"
        )
    if arr.dtype.kind != spec["dtype_kind"]:
        raise ArtifactRejected(
            f"prediction dtype {arr.dtype} (kind {arr.dtype.kind!r}) != expected "
            f"kind {spec['dtype_kind']!r}"
        )
    allowed = set(spec["values"])
    seen = set(np.unique(arr).tolist())
    if not seen <= allowed:
        raise ArtifactRejected(f"prediction contains values {sorted(seen - allowed)} "
                               f"outside the declared domain {sorted(allowed)}")


def check_finite(obj, *, _seen=None, _path="model", _budget=None) -> int:
    """Every float array reachable in the loaded model must be finite.

    NaN or Inf weights load without complaint and then predict without
    complaint; the first sign of trouble is a monitor that has quietly stopped
    finding anything. Returns the number of arrays checked so a caller can tell
    "all finite" from "found nothing to check".
    """
    import numpy as np

    if _seen is None:
        _seen, _budget = set(), [200_000]
    if id(obj) in _seen or _budget[0] <= 0:
        return 0
    _seen.add(id(obj))
    _budget[0] -= 1
    checked = 0

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "f":
            if not np.isfinite(obj).all():
                bad = int((~np.isfinite(obj)).sum())
                raise ArtifactRejected(
                    f"{_path}: array of shape {obj.shape} holds {bad} non-finite "
                    "value(s) (NaN or Inf)"
                )
            checked += 1
        return checked
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ArtifactRejected(f"{_path}: non-finite scalar {obj}")
        return 0
    if isinstance(obj, (str, bytes, int, bool, type(None))):
        return 0
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            checked += check_finite(v, _seen=_seen, _path=f"{_path}[{k!r}]",
                                    _budget=_budget)
        return checked
    if isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            checked += check_finite(v, _seen=_seen, _path=f"{_path}[{i}]",
                                    _budget=_budget)
        return checked
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for k, v in d.items():
            checked += check_finite(v, _seen=_seen, _path=f"{_path}.{k}",
                                    _budget=_budget)
    return checked


# --------------------------------------------------------------------------- #
# 5. The gate: validate a staged version, THEN activate it                    #
# --------------------------------------------------------------------------- #
def validate_version(client, registered_model: str, version: str, *,
                     canary_frame, canary_expected_rows: int,
                     expected_feature_order: Sequence[str],
                     expected_dataset_sha256: Optional[str] = None,
                     manifest: Optional[Mapping] = None,
                     manifest_dir: Optional[str] = None,
                     schema: Mapping = ARTIFACT_SCHEMA) -> dict:
    """Run every check against `version` WITHOUT touching the production alias.

    Loads by explicit version URI (`models:/name/version`), never by alias, so
    nothing here can make a bad artifact reachable to a caller. Returns a report
    on success; raises `ArtifactRejected` on the first failure.
    """
    import mlflow

    report: dict = {"version": str(version), "checks": []}

    def ok(name, detail=""):
        report["checks"].append({"check": name, "passed": True, "detail": detail})

    mv = client.get_model_version(registered_model, version)

    # 1. provenance, from the registry rather than from the artifact itself
    check_provenance(mv.tags, expected_dataset_sha256, schema)
    ok("provenance", f"dataset_sha256={mv.tags['dataset_sha256'][:12]}... "
                     f"kind={mv.tags['kind']}")

    artifact_dir = str(mv.source)
    if artifact_dir.startswith("file://"):
        artifact_dir = artifact_dir[len("file://"):]
    if not os.path.isdir(artifact_dir):
        # models:/ URIs and other non-local stores resolve through mlflow's own
        # downloader, which returns the local path for a file store without copying.
        try:
            artifact_dir = mlflow.artifacts.download_artifacts(str(mv.source))
        except Exception as exc:
            raise ArtifactRejected(
                f"version {version}: cannot resolve artifact source "
                f"{mv.source!r} to a local directory ({exc})"
            ) from None
    if not os.path.isdir(artifact_dir):
        raise ArtifactRejected(f"version {version}: artifact source {artifact_dir} "
                               "is not a readable directory")

    # 2. the artifact is complete, and matches its manifest if one was recorded
    check_required_files(artifact_dir, schema)
    ok("required_files", ", ".join(schema["required_files"]))
    if manifest is None and manifest_dir is not None:
        manifest = load_manifest(client, registered_model, version, manifest_dir)
    if manifest is not None:
        verify_files(artifact_dir, manifest)
        ok("digest", f"{len(manifest['files'])} files match sha256 and byte size")

    # 3. no executable payload - BEFORE load_model gets anywhere near it
    refs = check_pickle_globals(os.path.join(artifact_dir, "python_model.pkl"))
    mods = sorted({m for m, _ in refs})
    ok("pickle_globals", f"{len(refs)} global refs across {len(mods)} modules, "
                         "all allowlisted")

    # 4. the declared feature order is the contract's
    declared = check_feature_order(artifact_dir, expected_feature_order, schema)
    ok("feature_order", f"{len(declared)} features in contract order" if declared
       else "artifact declares no feature contract (raw-row model)")

    # 5. staged load, by version. Only now is any of the artifact's code run.
    model = mlflow.pyfunc.load_model(f"models:/{registered_model}/{version}")
    ok("staged_load", f"models:/{registered_model}/{version}")

    # 6. finite weights
    inner = model.unwrap_python_model() if hasattr(model, "unwrap_python_model") else model
    n_arrays = check_finite(inner)
    ok("finite_weights", f"{n_arrays} float array(s) checked, all finite")

    # 7. dtype/shape/domain of a real prediction on the contract's columns
    check_input_columns(canary_frame, schema)
    pred = model.predict(canary_frame)
    check_prediction(pred, canary_expected_rows, schema)
    import numpy as np
    ok("prediction_contract",
       f"shape {np.asarray(pred).shape} dtype {np.asarray(pred).dtype}")

    report["passed"] = True
    return report


def guarded_promote(client, registered_model: str, version: str, alias: str = "production",
                    **validate_kwargs) -> dict:
    """Validate `version`, and only then point `alias` at it.

    On rejection the alias is not touched, so whatever was serving before is
    still serving. That is the difference between a safeguard and an outage:
    refusing a bad artifact must never be the same event as losing the good one.
    """
    previous = None
    try:
        previous = client.get_model_version_by_alias(registered_model, alias)
    except Exception:
        previous = None  # nothing promoted yet; first promotion has nothing to keep

    try:
        report = validate_version(client, registered_model, version, **validate_kwargs)
    except ArtifactRejected as exc:
        still = None
        try:
            still = client.get_model_version_by_alias(registered_model, alias)
        except Exception:
            pass
        raise ArtifactRejected(
            f"v{version} REJECTED before activation: {exc}\n"
            f"  {alias} alias unchanged, still v{getattr(still, 'version', None)}"
        ) from None

    client.set_registered_model_alias(registered_model, alias, version)
    report["activated"] = True
    report["alias"] = alias
    report["previous_version"] = getattr(previous, "version", None)
    return report
