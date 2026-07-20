import json

import pytest

from notes_agent_v2.evaluation.artifacts import verify_bundle
from notes_agent_v2.evaluation.instruction_runtime import (
    OFFLINE_CASE_COUNTS,
    evaluate_planning_feature,
    render_planning_report,
    write_planning_evaluation_bundle,
)
from notes_agent_v2.evaluation.tracing import validate_trace


@pytest.mark.parametrize("feature_id", sorted(OFFLINE_CASE_COUNTS))
def test_offline_evaluation_is_complete_traced_and_sealed(tmp_path, feature_id) -> None:
    work = tmp_path / "work"
    report, results, trace_path = evaluate_planning_feature(feature_id, work)
    assert len(results) == OFFLINE_CASE_COUNTS[feature_id]
    assert report.treatment_correct == len(results)
    assert all(report.hard_gates.values())
    assert report.provider_requests == 0
    trace = validate_trace(trace_path)
    assert trace.failure_count == 0
    assert trace.request_count == 0
    output = tmp_path / "bundle"
    manifest = write_planning_evaluation_bundle(
        output, report=report, results=results, trace_path=trace_path
    )
    assert verify_bundle(output) == manifest
    assert (output / "report.json").read_text() == render_planning_report(report)
    serialized = json.dumps(manifest.model_dump(mode="json"))
    assert "prompt" not in serialized
    assert "reasoning" not in serialized


def test_semantic_features_remain_blocked_until_live_paired_evaluation(tmp_path) -> None:
    semantic = {
        "planning.generation_brief",
        "planning.salience_selection",
        "planning.closed_capability_plan",
    }
    for feature_id in semantic:
        report, _, _ = evaluate_planning_feature(feature_id, tmp_path / feature_id)
        assert report.verdict == "blocked_live_evaluation"
        assert report.live_evaluation_status == "blocked"
        assert report.blocker


def test_dispatcher_conformance_can_pass_without_provider_calls(tmp_path) -> None:
    report, _, _ = evaluate_planning_feature(
        "planning.bounded_dispatcher", tmp_path / "dispatcher"
    )
    assert report.verdict == "passed"
    assert report.live_evaluation_status == "not_required"


def test_unknown_feature_fails_closed(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown planning feature"):
        evaluate_planning_feature("planning.unknown", tmp_path)
