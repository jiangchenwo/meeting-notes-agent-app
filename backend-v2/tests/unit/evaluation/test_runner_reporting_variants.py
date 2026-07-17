from __future__ import annotations

from pathlib import Path

import pytest

from notes_agent_v2.evaluation.baselines import HistoricalResult, OneShotBaseline
from notes_agent_v2.evaluation.reporting import ReportError, build_report, render_report_json, render_report_markdown
from notes_agent_v2.evaluation.runner import EvaluationCell, EvaluationRunner, RunError
from notes_agent_v2.evaluation.tracing import validate_trace
from notes_agent_v2.evaluation.variants import EvaluationVariant, workflow_policy_for


def test_variant_registry_is_closed_and_application_owned() -> None:
    assert workflow_policy_for(EvaluationVariant.without_verification).verification is False
    with pytest.raises(ValueError):
        EvaluationVariant("invented")


def test_historical_results_can_never_be_certified() -> None:
    result = HistoricalResult(case_id="c1", output="old", reason="missing fingerprints")
    assert result.certified is False
    with pytest.raises(ValueError):
        HistoricalResult(case_id="c1", output="old", reason="x", certified=True)


def test_paired_runner_uses_identical_cases_seeds_and_rejects_missing_pair(tmp_path: Path) -> None:
    def execute(cell: EvaluationCell) -> dict[str, object]:
        return {"valid": True, "score": 1.0 if cell.variant == "treatment" else 0.5, "requests": 0}

    cells = [EvaluationCell(feature_id="eval.x", case_id="c1", variant=variant, seed=41, fingerprint="a" * 64) for variant in ("baseline", "treatment")]
    results = EvaluationRunner(tmp_path, execute).run(cells)
    assert all(result.trace_id and result.span_id for result in results)
    assert validate_trace(tmp_path / "events.jsonl").span_count == 2
    report = build_report("eval.x", results)
    assert report.verdict == "passed"
    assert report.paired_delta == 0.5
    assert report.bootstrap_low == report.bootstrap_high == 0.5
    assert render_report_json(report) == render_report_json(report)
    assert "eval.x" in render_report_markdown(report)
    with pytest.raises(ReportError, match="pair"):
        build_report("eval.x", results[:1])


def test_runner_reuses_only_exact_fingerprint(tmp_path: Path) -> None:
    cell = EvaluationCell(feature_id="eval.x", case_id="c1", variant="baseline", seed=41, fingerprint="a" * 64)
    runner = EvaluationRunner(tmp_path, lambda _: {"valid": True, "score": 1, "requests": 0})
    runner.run([cell])
    with pytest.raises(RunError, match="fingerprint"):
        runner.run([cell.model_copy(update={"fingerprint": "b" * 64})])


def test_report_rejects_mixed_pair_fingerprints(tmp_path: Path) -> None:
    cells = [EvaluationCell(feature_id="eval.x", case_id="c1", variant=variant, seed=41, fingerprint=fingerprint) for variant, fingerprint in (("baseline", "a" * 64), ("treatment", "b" * 64))]
    results = EvaluationRunner(tmp_path, lambda _: {"valid": True, "score": 1, "requests": 0}).run(cells)
    with pytest.raises(ReportError, match="fingerprint"):
        build_report("eval.x", results)


def test_report_rejects_duplicate_result_cells(tmp_path: Path) -> None:
    cells = [EvaluationCell(feature_id="eval.x", case_id="c1", variant=variant, seed=41, fingerprint="a" * 64) for variant in ("baseline", "treatment")]
    results = EvaluationRunner(tmp_path, lambda _: {"valid": True, "score": 1, "requests": 0}).run(cells)
    with pytest.raises(ReportError, match="duplicate"):
        build_report("eval.x", results + (results[0],))
