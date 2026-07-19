from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from notes_agent_v2.evaluation.artifacts import verify_bundle
from notes_agent_v2.evaluation.contract_persistence import (
    PHASE3_CASE_COUNTS,
    evaluate_phase3_feature,
    render_phase3_report,
    write_phase3_evaluation_bundle,
)
from notes_agent_v2.evaluation.specs import load_feature_specs
from notes_agent_v2.evaluation.tracing import validate_trace


def test_phase3_conformance_runs_exact_case_schedules_with_passing_gates(tmp_path: Path) -> None:
    for feature_id, expected_count in PHASE3_CASE_COUNTS.items():
        work = tmp_path / feature_id
        report, results, trace_path = evaluate_phase3_feature(feature_id, work)
        assert report.verdict == "passed"
        assert report.case_count == expected_count
        assert len(results) == expected_count
        assert report.treatment_correct == expected_count
        assert report.baseline_correct < report.treatment_correct
        assert len(report.code_fingerprint) == 64
        assert len(report.evaluation_fingerprint) == 64
        assert all(report.hard_gates.values())
        trace = validate_trace(trace_path)
        assert trace.span_count == expected_count * 2 + 1
        assert trace.failure_count == 0
        assert trace.request_count == 0


def test_phase3_conformance_bundle_is_sealed_and_report_is_reproducible(tmp_path: Path) -> None:
    work = tmp_path / "work"
    feature_id = "api.instruction_presets"
    report, results, trace_path = evaluate_phase3_feature(feature_id, work)
    output = tmp_path / "bundle"
    manifest = write_phase3_evaluation_bundle(
        output, report=report, results=results, trace_path=trace_path,
    )
    verified = verify_bundle(output)
    assert verified.bundle_digest == manifest.bundle_digest
    assert (output / "report.json").read_text() == render_phase3_report(report)
    assert {"events.jsonl", "report.json", "report.md", "results.json"} <= set(manifest.files)


def test_phase3_features_are_registered_with_offline_budgets() -> None:
    registry = Path(__file__).resolve().parents[3] / "config/evaluation/features.json"
    specs = load_feature_specs(registry)
    assert set(PHASE3_CASE_COUNTS) <= set(specs)
    for feature_id in PHASE3_CASE_COUNTS:
        assert specs[feature_id].max_requests == 0
        assert specs[feature_id].max_cost_usd == 0
        assert specs[feature_id].report_owner == "phase-03"


def test_phase3_evaluation_cli_writes_private_bundle(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[3] / "scripts/evaluate_contract_persistence.py"
    output = tmp_path / "strict-contracts"
    completed = subprocess.run(
        [sys.executable, str(script), "--feature", "domain.strict_contracts", "--private-out", str(output)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "verdict=passed cases=60" in completed.stdout
    assert verify_bundle(output).run_id == "domain.strict_contracts-phase3-conformance"
