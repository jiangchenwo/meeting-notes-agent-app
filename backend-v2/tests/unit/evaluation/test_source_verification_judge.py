from pathlib import Path

import pytest

from notes_agent_v2.evaluation.artifacts import EvaluationBundleWriter, verify_bundle
from notes_agent_v2.evaluation.judge_settings import JudgeSettings
from notes_agent_v2.evaluation.judges import (
    EvaluationJudgeGateway,
    ScriptedJudgeProvider,
)
from notes_agent_v2.evaluation.source_verification_judge import (
    SourceVerificationJudgeCase,
    estimate_source_verification_judge_budget,
    judge_configuration_fingerprint,
    run_source_verification_judge,
    validate_judge_qualification_bundle,
    write_source_verification_judge_bundle,
)
from notes_agent_v2.evaluation.tracing import JsonlTraceRecorder


def _settings() -> JudgeSettings:
    return JudgeSettings(
        provider="openai_compatible",
        model="qualified-model",
        base_url="https://example.test/v1",
        max_cost_usd=1,
        input_cost_per_million=0.25,
        output_cost_per_million=1.5,
        api_token="secret-one",
    )


def _qualification_bundle(path: Path, settings: JudgeSettings) -> Path:
    configuration = judge_configuration_fingerprint(settings)
    run_fingerprint = "a" * 64
    writer = EvaluationBundleWriter(
        path, run_id="remote-judge-qualification", fingerprint=run_fingerprint
    )
    writer.write_json(
        "report.json",
        {
            "schema_version": "judge-calibration-report-v1",
            "adapter_version": "openai-compatible-judge-v1",
            "suite_fingerprint": "b" * 64,
            "model_fingerprint": "c" * 64,
            "configuration_fingerprint": configuration,
            "run_fingerprint": run_fingerprint,
            "calibration": {
                "schema_valid": 60,
                "schema_total": 60,
                "critical_true_positive": 20,
                "critical_total": 20,
                "clean_critical_false_positive": 0,
                "clean_total": 20,
                "weighted_kappa": 1.0,
                "pair_order_agree": 20,
                "pair_order_total": 20,
                "privacy_failures": 0,
            },
            "qualification": {
                "status": "qualified",
                "schema_validity": 1.0,
                "critical_recall": 1.0,
                "clean_critical_false_positive_rate": 0.0,
                "weighted_kappa": 1.0,
                "pair_order_agreement": 1.0,
                "privacy_failures": 0,
            },
            "accounting": {
                "requests": 60,
                "reserved_input_tokens": 100,
                "reserved_output_tokens": 100,
                "estimated_cost_usd": 0.01,
            },
            "results": [],
        },
    )
    writer.seal()
    return path


def _cases() -> tuple[SourceVerificationJudgeCase, ...]:
    return (
        SourceVerificationJudgeCase(
            case_id="sample-clean",
            candidate="The team will use the API.",
            reference="[u000001] The team will use the API.",
            evidence_ids=("u000001",),
            injected_error=False,
        ),
        SourceVerificationJudgeCase(
            case_id="sample-injected",
            candidate="The team will not use the API.",
            reference="[u000001] The team will use the API.",
            evidence_ids=("u000001",),
            injected_error=True,
        ),
    )


def test_qualification_preflight_requires_exact_current_configuration(
    tmp_path: Path,
) -> None:
    settings = _settings()
    bundle = _qualification_bundle(tmp_path / "qualification", settings)

    authorization = validate_judge_qualification_bundle(bundle, settings)

    assert authorization.run_fingerprint == "a" * 64
    assert authorization.configuration_fingerprint == judge_configuration_fingerprint(
        settings
    )
    assert authorization.bundle_digest == verify_bundle(bundle).bundle_digest
    assert judge_configuration_fingerprint(
        settings.model_copy(update={"api_token": "secret-two"})
    ) == judge_configuration_fingerprint(settings)
    with pytest.raises(ValueError, match="configuration drift"):
        validate_judge_qualification_bundle(
            bundle, settings.model_copy(update={"rubric": "different-rubric"})
        )


def test_source_verification_budget_is_predeclared_and_bounded() -> None:
    budget = estimate_source_verification_judge_budget(_cases(), _settings())

    assert budget.request_cap == 2
    assert budget.output_token_cap == 512
    assert 0 < budget.estimated_cost_usd < _settings().max_cost_usd
    with pytest.raises(ValueError, match="positive input and output token prices"):
        estimate_source_verification_judge_budget(
            _cases(), _settings().model_copy(update={"input_cost_per_million": 0})
        )
    with pytest.raises(ValueError, match="configured cost cap"):
        estimate_source_verification_judge_budget(
            _cases(), _settings().model_copy(update={"max_cost_usd": 0.000001})
        )


def test_source_verification_judge_enforces_exact_clean_and_injected_gates(
    tmp_path: Path,
) -> None:
    provider = ScriptedJudgeProvider(
        [
            {"issues": []},
            {
                "issues": [
                    {
                        "code": "reversed_polarity",
                        "severity": "critical",
                        "evidence_refs": ["u000001"],
                        "justification": "The candidate reverses the source commitment.",
                    }
                ]
            },
        ]
    )
    trace_path = tmp_path / "events.jsonl"
    gateway = EvaluationJudgeGateway(
        _settings(),
        provider,
        allow_remote_judge=True,
        trace_recorder=JsonlTraceRecorder(trace_path),
    )

    report = run_source_verification_judge(
        _cases(),
        gateway,
        judge_authorization_fingerprint="a" * 64,
        judge_qualification_bundle_digest="d" * 64,
        development_fingerprint="e" * 64,
    )

    assert report.verdict == "passed"
    assert report.case_count == 2
    assert report.metrics == {
        "schema_validity": 1.0,
        "injected_error_recall": 1.0,
        "clean_critical_false_positive_rate": 0.0,
        "evidence_citation_precision": 1.0,
    }
    assert all(report.hard_gates.values())
    assert report.accounting.requests == 2
    assert report.results[1].critical_issue_detected is True


def test_source_verification_judge_fails_for_invalid_citation(tmp_path: Path) -> None:
    provider = ScriptedJudgeProvider(
        [
            {"issues": []},
            {
                "issues": [
                    {
                        "code": "reversed_polarity",
                        "severity": "critical",
                        "evidence_refs": ["u999999"],
                        "justification": "The candidate reverses the source commitment.",
                    }
                ]
            },
        ]
    )
    gateway = EvaluationJudgeGateway(
        _settings(), provider, allow_remote_judge=True
    )

    report = run_source_verification_judge(
        _cases(),
        gateway,
        judge_authorization_fingerprint="a" * 64,
        judge_qualification_bundle_digest="d" * 64,
        development_fingerprint="e" * 64,
    )

    assert report.verdict == "failed"
    assert report.hard_gates["evidence_citation_precision"] is False


def test_source_verification_judge_bundle_is_sealed(tmp_path: Path) -> None:
    trace_path = tmp_path / "events.jsonl"
    gateway = EvaluationJudgeGateway(
        _settings(),
        ScriptedJudgeProvider([{"issues": []}, {"issues": []}]),
        allow_remote_judge=True,
        trace_recorder=JsonlTraceRecorder(trace_path),
    )
    report = run_source_verification_judge(
        _cases(),
        gateway,
        judge_authorization_fingerprint="a" * 64,
        judge_qualification_bundle_digest="d" * 64,
        development_fingerprint="e" * 64,
    )
    output = tmp_path / "bundle"

    manifest = write_source_verification_judge_bundle(
        output, cases=_cases(), report=report, trace_path=trace_path
    )

    assert set(manifest.files) == {
        "cases.json",
        "events.jsonl",
        "report.json",
        "report.md",
        "results.json",
    }
    assert verify_bundle(output).bundle_digest == manifest.bundle_digest
