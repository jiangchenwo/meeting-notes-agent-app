import ast
from pathlib import Path

BACKEND_V2_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = BACKEND_V2_ROOT.parent
LEGACY_BACKEND_ROOT = (REPOSITORY_ROOT / "backend").resolve()
SOURCE_ROOTS = (BACKEND_V2_ROOT / "src", BACKEND_V2_ROOT / "scripts")
BANNED_TOP_LEVEL_MODULES = {"backend", "agents", "eval", "lm_config"}


def _python_sources() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for source_root in SOURCE_ROOTS
            for path in source_root.rglob("*.py")
        )
    )


def _top_level(module_name: str) -> str:
    return module_name.split(".", maxsplit=1)[0]


def _is_sys_path(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        and node.attr == "path"
    )


def _contains_sys_path_target(node: ast.AST) -> bool:
    if _is_sys_path(node):
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_contains_sys_path_target(item) for item in node.elts)
    if isinstance(node, ast.Subscript):
        return _is_sys_path(node.value)
    return False


def _dynamic_import_name(call: ast.Call) -> str | None:
    is_builtin_import = isinstance(call.func, ast.Name) and call.func.id == "__import__"
    is_importlib_import = (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "importlib"
        and call.func.attr == "import_module"
    )
    if not (is_builtin_import or is_importlib_import) or not call.args:
        return None
    first_argument = call.args[0]
    if isinstance(first_argument, ast.Constant) and isinstance(first_argument.value, str):
        return first_argument.value
    return None


def test_source_has_no_legacy_imports_or_path_bypasses() -> None:
    violations: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _top_level(alias.name) in BANNED_TOP_LEVEL_MODULES:
                        violations.append(f"{path}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if _top_level(node.module) in BANNED_TOP_LEVEL_MODULES:
                    violations.append(f"{path}:{node.lineno}: from {node.module}")
            elif isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and _is_sys_path(node.func.value)
                ):
                    violations.append(f"{path}:{node.lineno}: sys.path mutation")
                dynamic_name = _dynamic_import_name(node)
                if (
                    dynamic_name is not None
                    and _top_level(dynamic_name) in BANNED_TOP_LEVEL_MODULES
                ):
                    violations.append(
                        f"{path}:{node.lineno}: dynamic import {dynamic_name}"
                    )
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(_contains_sys_path_target(target) for target in targets):
                    violations.append(f"{path}:{node.lineno}: sys.path assignment")

    assert violations == []


def test_sources_and_symlinks_stay_inside_clean_room() -> None:
    violations: list[str] = []
    for path in _python_sources():
        resolved = path.resolve()
        if not resolved.is_relative_to(BACKEND_V2_ROOT):
            violations.append(f"Python source resolves outside backend-v2: {path}")

    for path in BACKEND_V2_ROOT.rglob("*"):
        if path.is_symlink() and path.resolve().is_relative_to(LEGACY_BACKEND_ROOT):
            violations.append(f"symlink resolves into legacy backend: {path}")

    assert violations == []
