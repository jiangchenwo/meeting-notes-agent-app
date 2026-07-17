from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid


class ActivationError(RuntimeError):
    pass


def activate_gold(staging: Path, active: Path, *, expected_case_ids: set[str], audit_decisions: dict[str, tuple[bool, bool]]) -> None:
    if not staging.is_dir() or not (staging / "manifest.json").is_file():
        raise ActivationError("staging set is incomplete")
    if set(audit_decisions) != expected_case_ids or any(decision != (True, True) for decision in audit_decisions.values()):
        raise ActivationError("activation requires two unanimous audits per case")
    incoming = active.parent / f".{active.name}.incoming-{uuid.uuid4().hex}"
    archive_root = active.parent / "archive"
    archive = archive_root / f"{active.name}-{uuid.uuid4().hex}"
    shutil.copytree(staging, incoming)
    try:
        if active.exists():
            archive_root.mkdir(parents=True, exist_ok=True)
            os.replace(active, archive)
        os.replace(incoming, active)
    except Exception:
        if archive.exists() and not active.exists():
            os.replace(archive, active)
        shutil.rmtree(incoming, ignore_errors=True)
        raise
