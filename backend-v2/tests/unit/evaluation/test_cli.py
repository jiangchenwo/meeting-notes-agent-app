from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from notes_agent_v2.evaluation.artifacts import verify_bundle
from notes_agent_v2.evaluation.tracing import validate_trace


def test_evaluate_feature_cli_writes_sealed_reproducible_bundle(tmp_path: Path) -> None:
    cells = tmp_path / "cells.json"
    rows = []
    for variant, score in (("baseline", 0.0), ("treatment", 1.0)):
        rows.append({
            "cell": {"feature_id": "eval.test", "case_id": "c1", "variant": variant, "seed": 41, "fingerprint": "a" * 64},
            "result": {"valid": True, "score": score, "requests": 0},
        })
    cells.write_text(json.dumps({"cells": rows}))
    output = tmp_path / "bundle"
    script = Path(__file__).resolve().parents[3] / "scripts/evaluate_feature.py"
    completed = subprocess.run([sys.executable, str(script), "--feature", "eval.test", "--cells", str(cells), "--private-out", str(output)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    manifest = verify_bundle(output)
    assert "events.jsonl" in manifest.files
    assert "report.json" in manifest.files
    assert "report.md" in manifest.files
    assert validate_trace(output / "events.jsonl").span_count >= 6
