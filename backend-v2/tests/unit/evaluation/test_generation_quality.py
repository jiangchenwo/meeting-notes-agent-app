from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from notes_agent_v2.evaluation.artifacts import verify_bundle
from notes_agent_v2.evaluation.generation_quality import (
    OFFLINE_CASE_COUNTS,
    evaluate_generation_feature,
    render_generation_report,
    write_generation_evaluation_bundle,
)
from notes_agent_v2.evaluation.tracing import validate_trace


def test_offline_schedules_are_complete_and_execute_real_boundaries(tmp_path) -> None:
    expected_verdicts = {
        "generation.fact_covered_outline": "passed",
        "generation.evidence_linked_writing": "blocked_live_evaluation",
        "quality.specialist_critics": "blocked_live_evaluation",
        "quality.deterministic_acceptance": "passed",
        "quality.targeted_revision": "blocked_live_evaluation",
    }
    for feature_id, expected_count in OFFLINE_CASE_COUNTS.items():
        report, results, trace_path = evaluate_generation_feature(
            feature_id, tmp_path / feature_id
        )
        assert len(results) == expected_count
        assert report.treatment_correct == expected_count
        assert report.provider_requests == 0
        assert report.verdict == expected_verdicts[feature_id]
        assert all(report.hard_gates.values())
        trace = validate_trace(trace_path)
        assert trace.request_count == 0
        assert trace.span_count == expected_count * 2 + 1


def test_generation_bundle_is_sealed_and_report_is_reproducible(tmp_path) -> None:
    work = tmp_path / "work"
    bundle = tmp_path / "bundle"
    report, results, trace_path = evaluate_generation_feature(
        "quality.deterministic_acceptance", work
    )
    manifest = write_generation_evaluation_bundle(
        bundle,
        report=report,
        results=results,
        trace_path=trace_path,
    )
    assert verify_bundle(bundle) == manifest
    rendered = render_generation_report(report)
    assert (bundle / "report.json").read_text() == rendered
    assert render_generation_report(
        type(report).model_validate_json((bundle / "report.json").read_text())
    ) == rendered


def test_generation_features_have_exact_registered_gates() -> None:
    registry = json.loads(
        (
            __import__("pathlib").Path(__file__).parents[3]
            / "config"
            / "evaluation"
            / "features.json"
        ).read_text()
    )
    features = {item["feature_id"]: item for item in registry["features"]}
    expected_metrics = {
        "generation.fact_covered_outline": {
            "mandatory_fact_coverage",
            "unsupported_reference_count",
            "valid_assignment_rate",
            "correction_ordering",
            "trace_completeness",
        },
        "generation.evidence_linked_writing": {
            "completion_and_validity",
            "accepted_evidence_link_rate",
            "unsupported_accepted_claims",
            "reference_recall_delta",
            "reference_precision_delta",
            "qag_alignment_delta",
        },
        "quality.specialist_critics": {
            "injected_critical_recall",
            "clean_critical_false_positives",
            "issue_target_validity",
            "critic_failure_conversion",
            "semantic_recall_delta",
        },
        "quality.deterministic_acceptance": {
            "disposition_ranking_agreement",
            "unsafe_acceptance_count",
            "critic_failure_review_routing",
            "deterministic_repeatability",
        },
        "quality.targeted_revision": {
            "targeted_issue_resolution",
            "new_critical_issues",
            "unchanged_block_integrity",
            "mandatory_fact_retention",
            "evidence_link_rate",
            "qag_parent_regression",
        },
    }
    for feature_id, metrics in expected_metrics.items():
        assert {item["metric"] for item in features[feature_id]["metrics"]} == metrics
        assert features[feature_id]["seeds"] == [41]
        assert {"stage", "evaluator", "report", "artifact"}.issubset(
            features[feature_id]["trace_requirements"]
        )


def test_generation_quality_cli_writes_a_sealed_private_bundle(tmp_path) -> None:
    output = tmp_path / "bundle"
    script = Path(__file__).parents[3] / "scripts" / "evaluate_generation_quality.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--feature",
            "generation.fact_covered_outline",
            "--private-out",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "verdict=passed" in completed.stdout
    assert "cases=32" in completed.stdout
    assert verify_bundle(output).files["report.json"]


def test_generation_quality_cli_does_not_overwrite_evidence(tmp_path) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    script = Path(__file__).parents[3] / "scripts" / "evaluate_generation_quality.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--feature",
            "quality.deterministic_acceptance",
            "--private-out",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "private output already exists" in completed.stderr
