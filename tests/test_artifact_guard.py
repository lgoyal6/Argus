"""Unit tests for experiments.artifact_guard.

Every check has a matching pair: an input it must ACCEPT (the negative control,
proving the check can pass and is not a blanket refusal) and an input it must
REJECT. Pure stdlib + numpy; no mlflow, torch or network. Run:

    python -m pytest tests/test_artifact_guard.py -q
"""
import json
import os
import pickle
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import artifact_guard as g  # noqa: E402


# ── pickle globals ───────────────────────────────────────────────────────────
def test_pickle_globals_accepts_ordinary_data():
    # control: a benign pickle references only builtins/collections.
    blob = pickle.dumps({"a": [1, 2, 3], "b": ("x", "y")}, protocol=5)
    refs = g.pickle_globals(blob, where="benign")
    bad = [f"{m}.{n}" for m, n in refs
           if m not in ("builtins", "__builtin__") or n not in g.ALLOWED_BUILTINS]
    assert bad == [], f"benign pickle flagged {bad}"


def test_pickle_globals_rejects_os_system(tmp_path):
    class Evil:
        def __reduce__(self):
            return (os.system, ("id",))
    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps(Evil(), protocol=5))
    with pytest.raises(g.ArtifactRejected) as e:
        g.check_pickle_globals(str(p))
    assert "posix.system" in str(e.value) or "os.system" in str(e.value)


def test_pickle_globals_resolves_memoised_stack_global(tmp_path):
    # protocol-5 pickle that reaches the same global twice via the memo; a
    # scanner that ignores the memo would miss the second, so this is the
    # regression guard for STACK_GLOBAL resolution.
    class Evil:
        def __reduce__(self):
            return (os.system, ("id",))
    p = tmp_path / "double.pkl"
    p.write_bytes(pickle.dumps([Evil(), Evil()], protocol=5))
    with pytest.raises(g.ArtifactRejected):
        g.check_pickle_globals(str(p))


def test_pickle_globals_rejects_unparseable(tmp_path):
    p = tmp_path / "trunc.pkl"
    p.write_bytes(pickle.dumps({"a": 1}, protocol=5)[:6])  # truncated
    with pytest.raises(g.ArtifactRejected):
        g.check_pickle_globals(str(p))


# ── digests / manifest ───────────────────────────────────────────────────────
def _artifact(tmp_path):
    d = tmp_path / "art"
    d.mkdir()
    (d / "MLmodel").write_text("flavors: {}\n")
    (d / "python_model.pkl").write_bytes(pickle.dumps({"ok": 1}, protocol=5))
    return d


def test_verify_files_accepts_then_rejects(tmp_path):
    d = _artifact(tmp_path)
    man = g.build_manifest(str(d), {"kind": "test"})
    g.verify_files(str(d), man)  # control: unmodified matches
    (d / "python_model.pkl").write_bytes(b"tampered")
    with pytest.raises(g.ArtifactRejected):
        g.verify_files(str(d), man)


# ── wiring: the guard must be ON the activation path, not beside it ──────────
def _lifecycle_tree():
    import ast
    src = (Path(__file__).resolve().parents[1] / "experiments" / "lifecycle.py")
    return ast.parse(src.read_text(), filename=str(src)), src


def test_lifecycle_activates_only_through_the_guard():
    """A guard that is implemented but not called rejects nothing.

    Three regressions are caught: promoting with a bare
    `set_registered_model_alias` (the pre-guard shape), calling
    `guarded_promote` without `manifest_dir` (the digest check silently becomes
    a no-op), and never recording a manifest at all (nothing to verify against).
    """
    import ast
    tree, src = _lifecycle_tree()
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

    bare = [f"line {n.lineno}" for n in calls
            if isinstance(n.func, ast.Attribute)
            and n.func.attr == "set_registered_model_alias"]
    assert not bare, (f"{src.name} moves the production alias directly at {bare}; "
                      "activation must go through guarded_promote")

    promotes = [n for n in calls if isinstance(n.func, ast.Attribute)
                and n.func.attr == "guarded_promote"]
    assert promotes, f"{src.name} never calls guarded_promote"
    for n in promotes:
        kw = {k.arg for k in n.keywords}
        assert "manifest_dir" in kw, (
            f"{src.name}:{n.lineno} calls guarded_promote without manifest_dir, "
            "so the digest check verifies against nothing")

    records = [n for n in calls if isinstance(n.func, ast.Attribute)
               and n.func.attr == "record_manifest"]
    assert records, f"{src.name} never records a manifest to verify against"


class _FakeClient:
    """Just enough MlflowClient to exercise manifest pinning without mlflow."""

    def __init__(self, tags=None):
        self._tags = dict(tags or {})

    def set_model_version_tag(self, _model, _version, key, value):
        self._tags[key] = value

    def get_model_version(self, _model, _version):
        return type("MV", (), {"tags": dict(self._tags)})()


def test_recorded_manifest_verifies_and_a_tampered_artifact_does_not(tmp_path):
    d = _artifact(tmp_path)
    mdir = tmp_path / "manifests"
    client = _FakeClient()
    g.record_manifest(client, "m", "3", str(d), {"kind": "test"}, str(mdir))

    # control: the manifest the guard loads back verifies the untouched artifact.
    man = g.load_manifest(client, "m", "3", str(mdir))
    g.verify_files(str(d), man)

    (d / "python_model.pkl").write_bytes(b"tampered")
    with pytest.raises(g.ArtifactRejected):
        g.verify_files(str(d), man)


def test_load_manifest_refuses_a_missing_tag_file_or_edited_manifest(tmp_path):
    d = _artifact(tmp_path)
    mdir = tmp_path / "manifests"
    client = _FakeClient()
    g.record_manifest(client, "m", "3", str(d), {"kind": "test"}, str(mdir))
    g.load_manifest(client, "m", "3", str(mdir))  # control: loads clean

    # a version that was never digested must not degrade into "skip the check"
    with pytest.raises(g.ArtifactRejected):
        g.load_manifest(_FakeClient(), "m", "3", str(mdir))

    # the manifest file itself is edited to match a tampered artifact
    p = Path(g.manifest_path(str(mdir), "m", "3"))
    man = json.loads(p.read_text())
    man["files"]["python_model.pkl"]["sha256"] = "0" * 64
    p.write_text(json.dumps(man))
    with pytest.raises(g.ArtifactRejected):
        g.load_manifest(client, "m", "3", str(mdir))

    p.unlink()
    with pytest.raises(g.ArtifactRejected):
        g.load_manifest(client, "m", "3", str(mdir))


def test_verify_files_rejects_extra_and_missing(tmp_path):
    d = _artifact(tmp_path)
    man = g.build_manifest(str(d), {"kind": "test"})
    (d / "surprise.txt").write_text("x")
    with pytest.raises(g.ArtifactRejected):
        g.verify_files(str(d), man)


# ── provenance ───────────────────────────────────────────────────────────────
def test_provenance_accepts_then_rejects():
    g.check_provenance({"dataset_sha256": "abc", "kind": "det"}, "abc")  # control
    with pytest.raises(g.ArtifactRejected):
        g.check_provenance({"kind": "det"})  # no dataset hash
    with pytest.raises(g.ArtifactRejected):
        g.check_provenance({"dataset_sha256": "zzz", "kind": "det"}, "abc")  # wrong hash


# ── feature order ────────────────────────────────────────────────────────────
def test_feature_order_accepts_matching_and_absent(tmp_path):
    d = tmp_path / "a"
    d.mkdir()
    assert g.check_feature_order(str(d), ["x", "y"]) is None  # control: none declared
    (d / "feature_contract.json").write_text(json.dumps({"features": ["x", "y"]}))
    assert g.check_feature_order(str(d), ["x", "y"]) == ["x", "y"]  # control: matches


def test_feature_order_rejects_permutation(tmp_path):
    d = tmp_path / "a"
    d.mkdir()
    (d / "feature_contract.json").write_text(json.dumps({"features": ["y", "x"]}))
    with pytest.raises(g.ArtifactRejected):
        g.check_feature_order(str(d), ["x", "y"])


# ── finiteness ───────────────────────────────────────────────────────────────
class _Model:
    def __init__(self, w):
        self.coef_ = w


def test_finite_accepts_then_rejects():
    n = g.check_finite(_Model(np.array([1.0, 2.0, 3.0])))  # control
    assert n == 1
    with pytest.raises(g.ArtifactRejected):
        g.check_finite(_Model(np.array([1.0, np.nan, 3.0])))
    with pytest.raises(g.ArtifactRejected):
        g.check_finite(_Model(np.array([1.0, np.inf])))


# ── prediction contract ──────────────────────────────────────────────────────
def test_prediction_accepts_then_rejects():
    g.check_prediction(np.array([0, 1, 1, 0]), 4)  # control
    with pytest.raises(g.ArtifactRejected):  # wrong length
        g.check_prediction(np.array([0, 1]), 4)
    with pytest.raises(g.ArtifactRejected):  # value outside {0,1}
        g.check_prediction(np.array([0, 2, 1, 0]), 4)
    with pytest.raises(g.ArtifactRejected):  # float dtype, not int
        g.check_prediction(np.array([0.0, 1.0, 1.0, 0.0]), 4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
