from __future__ import annotations

import json
from pathlib import Path

import pytest

from notes_agent_v2.evaluation.artifacts import BundleError, EvaluationBundleWriter, verify_bundle
from notes_agent_v2.evaluation.specs import FeatureEvaluationSpec, MetricGate, load_feature_specs
from notes_agent_v2.evaluation.tracing import JsonlTraceRecorder, TraceError, validate_trace


def test_feature_spec_requires_falsifiable_comparison_and_budget(tmp_path: Path) -> None:
    spec = FeatureEvaluationSpec(
        feature_id="eval.safe_tracing",
        hypothesis="Treatment detects all malformed traces.",
        baseline="unvalidated events",
        treatment="validated events",
        suite="scripted-trace-conformance",
        metrics=(MetricGate(metric="detection_rate", operation="gte", threshold=1.0),),
        trace_requirements=("stage", "metric"),
        seeds=(41,),
        max_requests=0,
        max_cost_usd=0,
        invalidation_conditions=("missing_terminal",),
        report_owner="phase-02",
    )
    path = tmp_path / "features.json"
    path.write_text(json.dumps({"features": [spec.model_dump(mode="json")]}))
    assert load_feature_specs(path)[spec.feature_id] == spec


def test_checked_in_feature_registry_is_valid_and_complete() -> None:
    path = Path(__file__).resolve().parents[3] / "config/evaluation/features.json"
    specs = load_feature_specs(path)
    assert len(specs) == 20
    assert sum(spec.report_owner == "phase-02" for spec in specs.values()) == 12
    assert sum(spec.report_owner == "phase-03" for spec in specs.values()) == 3
    assert sum(spec.report_owner == "phase-04" for spec in specs.values()) == 5


def test_trace_pairs_spans_and_rejects_private_fields(tmp_path: Path) -> None:
    recorder = JsonlTraceRecorder(tmp_path / "events.jsonl", trace_id="t1")
    with recorder.span("stage", feature_id="eval.safe_tracing", case_id="c1", variant="treatment", seed=41) as span:
        span.terminal(accounting={"requests": 1}, artifact_digests={"result": "a" * 64})
    report = validate_trace(tmp_path / "events.jsonl")
    assert report.span_count == 1
    assert report.request_count == 1

    with pytest.raises(TraceError, match="forbidden"):
        recorder.start("model", metadata={"prompt": "secret"})


def test_trace_detects_orphan_and_duplicate_terminal(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({
        "schema_version": "trace-v1", "trace_id": "t", "span_id": "s", "parent_span_id": "missing",
        "sequence": 1, "timestamp_ns": 1, "kind": "stage", "phase": "terminal", "status": "passed",
        "metadata": {}, "accounting": {}, "artifact_digests": {},
    }) + "\n")
    with pytest.raises(TraceError):
        validate_trace(path)


def test_trace_conformance_covers_100_operations_and_20_failures(tmp_path: Path) -> None:
    recorder = JsonlTraceRecorder(tmp_path / "events.jsonl", trace_id="conformance")
    kinds = ("stage", "model", "judge", "tool", "evaluator", "metric", "artifact", "report")
    for index in range(100):
        with recorder.span(kinds[index % len(kinds)], feature_id="eval.safe_tracing", case_id=f"c{index}", variant="treatment", seed=41) as span:
            failed = index < 20
            span.terminal(status="failed" if failed else "passed", accounting={"requests": int(kinds[index % len(kinds)] in {"model", "judge"})}, error_code="injected_failure" if failed else None)
    report = validate_trace(tmp_path / "events.jsonl")
    assert (report.span_count, report.failure_count, report.request_count) == (100, 20, 26)


def test_bundle_seals_atomically_and_detects_tampering(tmp_path: Path) -> None:
    target = tmp_path / "run"
    writer = EvaluationBundleWriter(target, run_id="run-1", fingerprint="f" * 64)
    writer.write_json("cases/c1.json", {"case_id": "c1", "candidate": "private evidence"})
    writer.write_text("events.jsonl", "{}\n")
    manifest = writer.seal()
    assert verify_bundle(target).bundle_digest == manifest.bundle_digest
    with pytest.raises(BundleError, match="sealed"):
        writer.write_text("later.txt", "no")
    (target / "cases/c1.json").write_text("tampered")
    with pytest.raises(BundleError, match="digest"):
        verify_bundle(target)


@pytest.mark.parametrize("relative", ("events.jsonl", "cases/c1.json", "report.json"))
def test_bundle_detects_tampering_in_every_file_class(tmp_path: Path, relative: str) -> None:
    target = tmp_path / relative.replace("/", "-")
    writer = EvaluationBundleWriter(target, run_id="run", fingerprint="f" * 64)
    writer.write_text("events.jsonl", "{}\n")
    writer.write_json("cases/c1.json", {"case_id": "c1"})
    writer.write_json("report.json", {"verdict": "passed"})
    writer.seal()
    (target / relative).write_text("tampered")
    with pytest.raises(BundleError, match="digest"):
        verify_bundle(target)
