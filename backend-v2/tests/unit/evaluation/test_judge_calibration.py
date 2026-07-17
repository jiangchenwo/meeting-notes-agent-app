from pathlib import Path

from notes_agent_v2.evaluation.judge_calibration import (
    IssueCalibrationCase,
    JudgeCalibrationSuite,
    PairCalibrationCase,
    estimate_calibration_budget,
    load_calibration_suite,
    run_judge_calibration,
    validate_fixed_suite,
    write_calibration_bundle,
)
from notes_agent_v2.evaluation.artifacts import verify_bundle
from notes_agent_v2.evaluation.judge_settings import JudgeSettings
from notes_agent_v2.evaluation.judges import (
    EvaluationJudgeGateway,
    JudgeQualificationStatus,
    ScriptedJudgeProvider,
)
from notes_agent_v2.evaluation.tracing import JsonlTraceRecorder, validate_trace


def settings() -> JudgeSettings:
    return JudgeSettings(
        provider="openai_compatible", model="scripted", base_url="https://example.test/v1",
        max_cost_usd=1, input_cost_per_million=0.25,
        output_cost_per_million=1.5, api_token="super-secret-value",
    )


def small_suite() -> JudgeCalibrationSuite:
    return JudgeCalibrationSuite(
        issues=(
            IssueCalibrationCase(
                case_id="critical-01", reference="[u1] The board approved $10.",
                candidate="The board rejected $10.", expected_critical=True,
            ),
            IssueCalibrationCase(
                case_id="clean-01", reference="[u1] The board approved $10.",
                candidate="The board approved $10.", expected_critical=False,
            ),
        ),
        pairs=(PairCalibrationCase(
            case_id="pair-01", reference="[u1] The board approved $10.",
            better="The board approved $10.", worse="The board rejected $10.",
        ),),
    )


def test_calibration_aggregates_issue_and_swapped_pair_results(tmp_path: Path) -> None:
    provider = ScriptedJudgeProvider([
        {"issues": [{"code": "contradiction", "severity": "critical", "evidence_refs": ["u1"], "justification": "The disposition is reversed."}]},
        {"issues": []},
        {"winner": "A", "evidence_refs": ["u1"], "justification": "A matches the reference."},
        {"winner": "B", "evidence_refs": ["u1"], "justification": "B matches the reference."},
    ])
    trace = JsonlTraceRecorder(tmp_path / "events.jsonl", trace_id="qualification")
    gateway = EvaluationJudgeGateway(settings(), provider, allow_remote_judge=True, trace_recorder=trace)

    report = run_judge_calibration(small_suite(), gateway)

    assert report.calibration.schema_valid == report.calibration.schema_total == 4
    assert report.calibration.critical_true_positive == report.calibration.critical_total == 1
    assert report.calibration.clean_critical_false_positive == 0
    assert report.calibration.clean_total == 1
    assert report.calibration.weighted_kappa == 1
    assert report.calibration.pair_order_agree == report.calibration.pair_order_total == 2
    assert report.qualification.status is JudgeQualificationStatus.qualified
    assert report.accounting.requests == 4
    assert len(report.results) == 4
    assert report.results[0].issues[0].code == "contradiction"
    assert report.results[2].pairwise is not None
    assert report.results[2].pairwise.winner == "A"
    assert len(report.configuration_fingerprint) == 64
    assert len(report.run_fingerprint) == 64
    assert validate_trace(tmp_path / "events.jsonl").request_count == 4


def test_calibration_records_malformed_response_and_fails_closed(tmp_path: Path) -> None:
    provider = ScriptedJudgeProvider([
        {"not_issues": []},
        {"issues": []},
        {"winner": "A", "evidence_refs": ["u1"], "justification": "A matches."},
        {"winner": "B", "evidence_refs": ["u1"], "justification": "B matches."},
    ])
    gateway = EvaluationJudgeGateway(
        settings(), provider, allow_remote_judge=True,
        trace_recorder=JsonlTraceRecorder(tmp_path / "events.jsonl"),
    )

    report = run_judge_calibration(small_suite(), gateway)

    assert report.calibration.schema_valid == 3
    assert report.calibration.schema_total == 4
    assert report.results[0].error_code == "judge_schema_invalid"
    assert report.qualification.status is JudgeQualificationStatus.diagnostic_unqualified
    assert validate_trace(tmp_path / "events.jsonl").failure_count == 1


def test_fixed_suite_requires_20_critical_20_clean_and_10_pairs() -> None:
    suite = small_suite()
    try:
        validate_fixed_suite(suite)
    except ValueError as exc:
        assert str(exc) == "fixed judge calibration requires 20 critical, 20 clean, and 10 pair cases"
    else:
        raise AssertionError("small suite unexpectedly passed the fixed-suite gate")


def test_fixed_suite_rejects_duplicate_case_ids() -> None:
    path = Path(__file__).resolve().parents[2] / "fixtures/evaluation/judge-calibration.json"
    suite = load_calibration_suite(path)
    duplicated = suite.model_copy(update={
        "pairs": (*suite.pairs[:-1], suite.pairs[0].model_copy(update={"case_id": suite.issues[0].case_id})),
    })
    try:
        validate_fixed_suite(duplicated)
    except ValueError as exc:
        assert str(exc) == "fixed judge calibration case IDs must be unique"
    else:
        raise AssertionError("duplicate calibration case unexpectedly passed")


def test_checked_in_fixed_suite_has_exact_shape() -> None:
    path = Path(__file__).resolve().parents[2] / "fixtures/evaluation/judge-calibration.json"
    suite = load_calibration_suite(path)
    validate_fixed_suite(suite)
    assert len({item.case_id for item in (*suite.issues, *suite.pairs)}) == 50


def test_calibration_bundle_contains_report_results_suite_and_trace(tmp_path: Path) -> None:
    provider = ScriptedJudgeProvider([
        {"issues": [{"code": "contradiction", "severity": "critical", "evidence_refs": ["u1"], "justification": "The disposition is reversed."}]},
        {"issues": []},
        {"winner": "A", "evidence_refs": ["u1"], "justification": "A matches."},
        {"winner": "B", "evidence_refs": ["u1"], "justification": "B matches."},
    ])
    trace_path = tmp_path / "events.jsonl"
    gateway = EvaluationJudgeGateway(
        settings(), provider, allow_remote_judge=True,
        trace_recorder=JsonlTraceRecorder(trace_path),
    )
    suite = small_suite()
    report = run_judge_calibration(suite, gateway)
    output = tmp_path / "bundle"

    manifest = write_calibration_bundle(output, suite=suite, report=report, trace_path=trace_path)

    assert set(manifest.files) == {
        "calibration-results.json", "events.jsonl", "report.json", "report.md", "suite.json",
    }
    assert manifest.fingerprint == report.run_fingerprint
    assert verify_bundle(output).bundle_digest == manifest.bundle_digest
    assert "super-secret-value" not in "".join(path.read_text() for path in output.rglob("*") if path.is_file())


def test_run_fingerprint_tracks_configuration_but_not_api_token(tmp_path: Path) -> None:
    responses = [
        {"issues": [{"code": "contradiction", "severity": "critical", "evidence_refs": ["u1"], "justification": "Reversed."}]},
        {"issues": []},
        {"winner": "A", "evidence_refs": ["u1"], "justification": "A matches."},
        {"winner": "B", "evidence_refs": ["u1"], "justification": "B matches."},
    ]

    def run(name: str, configured: JudgeSettings):
        gateway = EvaluationJudgeGateway(
            configured,
            ScriptedJudgeProvider(responses.copy()),
            allow_remote_judge=True,
            trace_recorder=JsonlTraceRecorder(tmp_path / f"{name}.jsonl"),
        )
        return run_judge_calibration(small_suite(), gateway)

    original = run("original", settings())
    new_token = run("new-token", settings().model_copy(update={"api_token": "different-secret"}))
    new_rubric = run("new-rubric", settings().model_copy(update={"rubric": "issues-v2"}))

    assert new_token.configuration_fingerprint == original.configuration_fingerprint
    assert new_token.run_fingerprint == original.run_fingerprint
    assert new_rubric.configuration_fingerprint != original.configuration_fingerprint
    assert new_rubric.run_fingerprint != original.run_fingerprint


def test_calibration_budget_is_predeclared_and_rejects_unpriced_calls() -> None:
    budget = estimate_calibration_budget(small_suite(), settings())
    assert budget.request_cap == 4
    assert budget.input_token_reservation > 0
    assert budget.output_token_cap == 768
    assert budget.estimated_cost_usd < 1
    assert budget.time_cap_seconds == 120
    paced = estimate_calibration_budget(
        small_suite(), settings().model_copy(update={"min_interval_seconds": 4.1}),
    )
    assert paced.time_cap_seconds == 132.3

    unpriced = settings().model_copy(update={"input_cost_per_million": 0, "output_cost_per_million": 0})
    try:
        estimate_calibration_budget(small_suite(), unpriced)
    except ValueError as exc:
        assert str(exc) == "judge qualification requires positive input and output token prices"
    else:
        raise AssertionError("unpriced qualification unexpectedly passed")
