from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
ROADMAP_NAME = re.compile(
    r"(?:phase[-_ ]?\d+|implementation[-_ ]?plan|execution[-_ ]?roadmap)",
    re.IGNORECASE,
)


def _python_files() -> tuple[Path, ...]:
    roots = (ROOT / "src", ROOT / "scripts", ROOT / "tests")
    return tuple(
        path
        for root in roots
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_committed_python_names_do_not_expose_implementation_roadmap_labels() -> None:
    violations = [
        path.relative_to(ROOT).as_posix()
        for path in _python_files()
        if ROADMAP_NAME.search(path.name)
    ]
    assert violations == []


def test_code_docstrings_do_not_expose_implementation_roadmap_labels() -> None:
    violations: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            docstring = ast.get_docstring(node, clean=False)
            if docstring and ROADMAP_NAME.search(docstring):
                violations.append(
                    f"{path.relative_to(ROOT).as_posix()}:{getattr(node, 'lineno', 1)}"
                )
    assert violations == []
