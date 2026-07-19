from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from notes_agent_v2.evaluation.artifacts import verify_bundle
from notes_agent_v2.evaluation.evidence_memory import (
    PHASE4_CASE_COUNTS,
    evaluate_phase4_feature,
    render_phase4_report,
    write_phase4_evaluation_bundle,
)
from notes_agent_v2.evaluation.specs import load_feature_specs
from notes_agent_v2.evaluation.tracing import validate_trace


ROOT = Path(__file__).resolve().parents[3]
FEATURES = (
    "evidence.token_aware_chunking",
    "evidence.cited_atomic_extraction",
    "evidence.source_verification",
    "evidence.loss_aware_consolidation",
    "evidence.scoped_tools",
)


def test_phase4_registry_defines_all_feature_hypotheses_and_gates() -> None:
    specs = load_feature_specs(ROOT / "config/evaluation/features.json")
    assert set(FEATURES).issubset(specs)
    assert all(specs[item].report_owner == "phase-04" for item in FEATURES)
    assert specs["evidence.token_aware_chunking"].max_requests == 0
    assert specs["evidence.scoped_tools"].max_requests == 0
    assert specs["evidence.cited_atomic_extraction"].max_requests > 0


@pytest.mark.parametrize("feature_id", FEATURES)
def test_phase4_offline_evaluations_are_complete_traced_and_sealable(
    tmp_path: Path, feature_id: str
) -> None:
    report, results, trace_path = evaluate_phase4_feature(
        feature_id, tmp_path / "work"
    )

    assert len(results) == PHASE4_CASE_COUNTS[feature_id]
    assert report.case_count == PHASE4_CASE_COUNTS[feature_id]
    assert report.treatment_correct == report.case_count
    assert report.provider_requests == 0
    assert report.live_evaluation_status in {"not_required", "blocked"}
    expected_verdict = (
        "passed"
        if feature_id in {"evidence.token_aware_chunking", "evidence.scoped_tools"}
        else "blocked_live_evaluation"
    )
    assert report.verdict == expected_verdict
    trace = validate_trace(trace_path)
    assert trace.failure_count == 0
    assert trace.request_count == 0
    assert trace.span_count == report.case_count * 2 + 1

    first_report = render_phase4_report(report)
    assert first_report == render_phase4_report(report)
    output = tmp_path / "bundle"
    manifest = write_phase4_evaluation_bundle(
        output,
        report=report,
        results=results,
        trace_path=trace_path,
    )
    verified = verify_bundle(output)
    assert verified.bundle_digest == manifest.bundle_digest
    assert verified.fingerprint == report.evaluation_fingerprint
    assert json.loads((output / "report.json").read_text())["feature_id"] == feature_id


def test_phase4_cli_writes_verified_offline_bundle(tmp_path: Path) -> None:
    output = tmp_path / "cli-bundle"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_evidence_memory.py"),
            "--feature",
            "evidence.token_aware_chunking",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert verify_bundle(output).run_id.startswith("evidence.token_aware_chunking")
