"""Guards against a class of bug that unit tests alone can't catch: a new
top-level module that the local `pythonpath=["."]` test config imports
fine, but that isn't declared in pyproject's `py-modules` or COPY'd into
the Dockerfile -- so `pytest` is green while `pip install .` and the
Docker image are broken. (This actually happened: conversation.py was
imported by api/app.py but missing from both.)
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).parent.parent

# Top-level single-file modules that are imported at runtime and therefore
# must be shipped, both as pyproject py-modules and in the Docker image.
RUNTIME_MODULES = ["features", "model", "explain", "conversation", "baselines", "cli"]


def test_runtime_modules_are_declared_in_pyproject():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    declared = set(data["tool"]["setuptools"]["py-modules"])
    missing = [m for m in RUNTIME_MODULES if m not in declared]
    assert not missing, f"modules imported at runtime but not in pyproject py-modules: {missing}"


def test_runtime_modules_are_copied_into_the_docker_image():
    dockerfile = (ROOT / "Dockerfile").read_text()
    copied = set(re.findall(r"[\w./]+\.py", dockerfile))
    missing = [m for m in RUNTIME_MODULES if f"{m}.py" not in copied]
    assert not missing, f"modules imported at runtime but not COPY'd in Dockerfile: {missing}"


def test_api_app_only_imports_shipped_top_level_modules():
    """If api/app.py grows a new top-level import, this fails until that
    module is added to RUNTIME_MODULES (and thus to the two checks above),
    so the packaging guards can't silently fall out of date."""
    tree = ast.parse((ROOT / "api" / "app.py").read_text())
    local_module_files = {p.stem for p in ROOT.glob("*.py")}

    imported_local: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top = node.module.split(".")[0]
            if top in local_module_files:
                imported_local.add(top)

    undeclared = imported_local - set(RUNTIME_MODULES)
    assert not undeclared, (
        f"api/app.py imports top-level modules not tracked in RUNTIME_MODULES: {undeclared}"
    )
