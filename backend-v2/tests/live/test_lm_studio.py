from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.lm_studio
def test_approved_runtime_passes_all_live_probes(tmp_path: Path) -> None:
    runtime_report = os.getenv("NOTES_RUNTIME_REPORT")
    if not runtime_report:
        pytest.skip("set NOTES_RUNTIME_REPORT to explicitly authorize live probes")

    root = Path(__file__).parents[2]
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/probe_lm_studio.py",
            "--runtime-report",
            runtime_report,
            "--json-out",
            str(tmp_path / "public.json"),
            "--trace-out",
            str(tmp_path / "private.trace.json"),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
