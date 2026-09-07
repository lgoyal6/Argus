"""The images run a FLAT module layout, and the test suite does not.

`docker/Dockerfile.agent` does `COPY agent/ .` and `docker/Dockerfile.backend` does
`COPY backend/ .`, so inside either image there is no `agent` or `backend` package:
every module sits directly in `/app`. The test suite runs from the repository root,
where both packages exist. That gap is silent in the worst possible way - a
package-qualified `from agent.sandbox import ...` passes every test and crashes the
container on the first import - so it gets a test of its own rather than a comment.

Three rules, checked statically so this needs no runtime dependency and runs in CI:

1. A first-party package-qualified import must sit under a `try/except ImportError`
   with a flat fallback. That is the pattern already used by agent/attempt.py,
   agent/tools.py and backend/db.py.
2. A module imported plainly by name, that lives at the repository root rather than in
   the package directory, must be copied into the image by that package's Dockerfile.
   `obs.py` is the case: both trees do `import obs`, and without the matching `COPY`
   the image starts and immediately dies.
3. The same rule for a module that lives in the *other* first-party package.
   `argus_secrets.py` is the case: it belongs to `backend/` and `agent/` imports it by
   plain name, which resolves in the image only because `docker/Dockerfile.agent`
   copies that one file in. Rule 2 does not cover it, because the module is not at the
   repository root, and nothing else would notice it going missing.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {"agent": ROOT / "docker" / "Dockerfile.agent",
            "backend": ROOT / "docker" / "Dockerfile.backend"}
FIRST_PARTY = {"agent", "backend"}


def _sources(package):
    return sorted(p for p in (ROOT / package).rglob("*.py")
                  if "__pycache__" not in p.parts)


def _guarded_import_nodes(tree):
    """Every import node that sits inside a `try:` with an ImportError handler."""
    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        handles = any(
            h.type is None
            or (isinstance(h.type, ast.Name) and h.type.id == "ImportError")
            or (isinstance(h.type, ast.Tuple)
                and any(isinstance(e, ast.Name) and e.id == "ImportError"
                        for e in h.type.elts))
            for h in node.handlers
        )
        if not handles:
            continue
        for stmt in node.body:
            for inner in ast.walk(stmt):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    guarded.add(id(inner))
    return guarded


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_no_unguarded_package_qualified_import(package):
    offenders = []
    for path in _sources(package):
        tree = ast.parse(path.read_text())
        guarded = _guarded_import_nodes(tree)
        for node in ast.walk(tree):
            if id(node) in guarded:
                continue
            if isinstance(node, ast.ImportFrom) and node.module:
                head = node.module.split(".")[0]
                if head in FIRST_PARTY:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} "
                                     f"from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in FIRST_PARTY:
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} "
                                         f"import {alias.name}")
    assert not offenders, (
        "package-qualified first-party imports that will not resolve in the flat image "
        "layout, and that no test can catch because the tests run from the repository "
        "root:\n  " + "\n  ".join(offenders)
    )


def _copied_sources(dockerfile):
    """The COPY sources in a Dockerfile, as written."""
    copied = set()
    for line in dockerfile.read_text().splitlines():
        m = re.match(r"\s*COPY\s+(.+)$", line)
        if m:
            parts = m.group(1).split()
            copied.update(parts[:-1])  # everything but the destination
    return copied


ROOT_MODULES = {p.stem for p in ROOT.glob("*.py")}


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_repository_root_modules_are_copied_into_the_image(package):
    imported = set()
    for path in _sources(package):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])

    needed = sorted(imported & ROOT_MODULES)
    copied = _copied_sources(PACKAGES[package])
    missing = [f"{m}.py" for m in needed if f"{m}.py" not in copied]
    assert not missing, (
        f"{PACKAGES[package].relative_to(ROOT)} does not COPY {missing}, which "
        f"{package}/ imports by plain name. The image would start and die on import "
        f"while the suite stayed green."
    )
    assert needed, "expected at least one shared root module; has obs.py moved?"


def _plain_imports(package):
    """Plain, unqualified, absolute imports made anywhere in a package."""
    names = set()
    for path in _sources(package):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names if "." not in a.name)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if "." not in node.module:
                    names.add(node.module)
    return names


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_modules_borrowed_from_the_other_package_are_copied_into_the_image(package):
    """A module owned by the other package, imported here by plain name.

    Shared rather than duplicated is the right call, but it means the image only
    works because one COPY line names one file. Deleting that line leaves every test
    green - the suite runs from the repository root, where both packages are
    importable - and breaks the container on its first import.
    """
    others = {p: {s.stem for s in (ROOT / p).glob("*.py")}
              for p in FIRST_PARTY if p != package}
    mine = {s.stem for s in (ROOT / package).glob("*.py")}
    copied = _copied_sources(PACKAGES[package])

    missing = []
    for name in sorted(_plain_imports(package)):
        if name in mine or name in ROOT_MODULES:
            continue
        for other, modules in others.items():
            if name in modules and f"{other}/{name}.py" not in copied:
                missing.append(f"{other}/{name}.py")

    assert not missing, (
        f"{PACKAGES[package].relative_to(ROOT)} does not COPY {missing}, which "
        f"{package}/ imports by plain name from the other package. The image would "
        f"start and die on import while the suite stayed green."
    )
