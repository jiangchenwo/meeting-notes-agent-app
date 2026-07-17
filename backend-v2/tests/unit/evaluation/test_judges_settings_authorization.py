from __future__ import annotations

import json
from pathlib import Path

import pytest

from notes_agent_v2.evaluation.judge_settings import JudgeConfigurationError, load_judge_settings
from notes_agent_v2.evaluation.judges import (
    DataClassification, EvaluationJudgeGateway, JudgeCalibration, JudgeError, JudgeIssueResult,
    JudgeQualificationStatus, PairwiseResult, ScriptedJudgeProvider, qualify_judge,
)
from notes_agent_v2.evaluation.runtime_authorization import AuthorizationError, DevelopmentEvaluationEvidence, qualify_development_runtime
from notes_agent_v2.evaluation.tracing import JsonlTraceRecorder, validate_trace


def config(path: Path, **updates: object) -> Path:
    payload = {
        "schema_version": "judge-settings-v1", "provider": "disabled", "model": None,
        "base_url": None, "timeout_seconds": 30, "max_cost_usd": 0,
        "input_cost_per_million": 0, "output_cost_per_million": 0,
        "temperature": 0, "rubric": "issues-v1",
    }
    payload.update(updates)
    path.write_text(json.dumps(payload))
    return path


def test_judge_settings_precedence_and_secret_exclusion(tmp_path: Path) -> None:
    path = config(tmp_path / "judge.json", timeout_seconds=10)
    env = tmp_path / ".env"
    env.write_text("NOTES_EVAL_JUDGE_TIMEOUT_SECONDS=20\nNOTES_EVAL_JUDGE_MODEL=env-model\n")
    settings = load_judge_settings(path, env_file=env, environ={"NOTES_EVAL_JUDGE_TIMEOUT_SECONDS": "40"})
    assert settings.timeout_seconds == 40
    assert settings.model == "env-model"
    assert "api_token" not in settings.model_dump()
    assert "api_token" not in repr(settings)


def test_judge_settings_load_cost_rates_from_environment(tmp_path: Path) -> None:
    path = config(tmp_path / "judge.json")
    settings = load_judge_settings(path, environ={
        "NOTES_EVAL_JUDGE_INPUT_COST_PER_MILLION": "0.25",
        "NOTES_EVAL_JUDGE_OUTPUT_COST_PER_MILLION": "1.50",
    })
    assert settings.input_cost_per_million == 0.25
    assert settings.output_cost_per_million == 1.50


def test_judge_settings_load_minimum_request_interval_from_environment(tmp_path: Path) -> None:
    path = config(tmp_path / "judge.json")
    settings = load_judge_settings(path, environ={
        "NOTES_EVAL_JUDGE_MIN_INTERVAL_SECONDS": "4.1",
    })
    assert settings.min_interval_seconds == 4.1


def test_checked_in_judge_is_disabled() -> None:
    path = Path(__file__).resolve().parents[3] / "config/evaluation/judge.json"
    assert load_judge_settings(path, environ={}).provider == "disabled"


def test_judge_settings_reject_json_secret_and_incomplete_remote(tmp_path: Path) -> None:
    with pytest.raises(JudgeConfigurationError, match="environment-only"):
        load_judge_settings(config(tmp_path / "secret.json", api_token="bad"), environ={})
    with pytest.raises(JudgeConfigurationError, match="requires"):
        load_judge_settings(config(tmp_path / "remote.json", provider="openai_compatible"), environ={})


def test_judge_gateway_is_opt_in_public_only_budgeted_and_structured(tmp_path: Path) -> None:
    settings = load_judge_settings(config(tmp_path / "judge.json", provider="openai_compatible", model="cheap", base_url="https://example.test/v1", max_cost_usd=1), environ={"NOTES_EVAL_JUDGE_API_TOKEN": "token"})
    provider = ScriptedJudgeProvider([{"issues": [{"code": "unsupported", "severity": "critical", "evidence_refs": ["u1"], "justification": "Claim lacks support."}]}])
    recorder = JsonlTraceRecorder(tmp_path / "events.jsonl")
    gateway = EvaluationJudgeGateway(settings, provider, allow_remote_judge=True, trace_recorder=recorder)
    result = gateway.evaluate_issues(classification=DataClassification.public_benchmark, candidate="notes", reference="reference", estimated_input_tokens=10, max_output_tokens=20)
    assert isinstance(result, JudgeIssueResult)
    assert result.issues[0].code == "unsupported"
    assert gateway.accounting.requests == 1
    assert validate_trace(tmp_path / "events.jsonl").span_count == 1
    system_prompt = provider.requests[0]["messages"][0]["content"]
    assert '"issues"' in system_prompt
    assert "critical" in system_prompt
    assert "material contradiction" in system_prompt
    with pytest.raises(JudgeError, match="public benchmark"):
        gateway.evaluate_issues(classification=DataClassification.private_user, candidate="x", reference="y", estimated_input_tokens=1, max_output_tokens=1)


def test_judge_gateway_runs_anonymized_pairwise_calls_with_shared_budget(tmp_path: Path) -> None:
    settings = load_judge_settings(
        config(
            tmp_path / "judge.json", provider="openai_compatible", model="cheap",
            base_url="https://example.test/v1", max_cost_usd=1,
            input_cost_per_million=1, output_cost_per_million=2,
        ),
        environ={"NOTES_EVAL_JUDGE_API_TOKEN": "token"},
    )
    provider = ScriptedJudgeProvider([
        {"winner": "A", "evidence_refs": ["u1"], "justification": "A matches the decision."},
        {"winner": "B", "evidence_refs": ["u1"], "justification": "B matches the decision."},
    ])
    recorder = JsonlTraceRecorder(tmp_path / "pair-events.jsonl")
    gateway = EvaluationJudgeGateway(settings, provider, allow_remote_judge=True, trace_recorder=recorder)

    first = gateway.evaluate_pairwise(
        classification=DataClassification.public_benchmark, candidate_a="approved 10",
        candidate_b="rejected 10", reference="[u1] approved 10",
        estimated_input_tokens=20, max_output_tokens=30,
    )
    second = gateway.evaluate_pairwise(
        classification=DataClassification.public_benchmark, candidate_a="rejected 10",
        candidate_b="approved 10", reference="[u1] approved 10",
        estimated_input_tokens=20, max_output_tokens=30,
    )

    assert isinstance(first, PairwiseResult)
    assert (first.winner, second.winner) == ("A", "B")
    assert gateway.accounting.requests == 2
    assert gateway.accounting.estimated_cost_usd == pytest.approx(0.00016)
    assert validate_trace(tmp_path / "pair-events.jsonl").request_count == 2
    assert all("candidate_a" in request["messages"][1]["content"] for request in provider.requests)


def test_failed_judge_schema_still_records_reserved_request(tmp_path: Path) -> None:
    settings = load_judge_settings(
        config(
            tmp_path / "judge.json", provider="openai_compatible", model="cheap",
            base_url="https://example.test/v1", max_cost_usd=1,
        ),
        environ={"NOTES_EVAL_JUDGE_API_TOKEN": "token"},
    )
    trace_path = tmp_path / "failed-events.jsonl"
    gateway = EvaluationJudgeGateway(
        settings, ScriptedJudgeProvider([{"invalid": True}]), allow_remote_judge=True,
        trace_recorder=JsonlTraceRecorder(trace_path),
    )

    with pytest.raises(JudgeError, match="schema"):
        gateway.evaluate_issues(
            classification=DataClassification.public_benchmark, candidate="notes",
            reference="reference", estimated_input_tokens=10, max_output_tokens=20,
        )

    validation = validate_trace(trace_path)
    assert validation.request_count == 1
    assert validation.failure_count == 1


def test_judge_gateway_paces_serial_provider_requests(tmp_path: Path) -> None:
    current_time = [100.0]
    waits: list[float] = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        current_time[0] += seconds

    configured = load_judge_settings(
        config(
            tmp_path / "judge.json", provider="openai_compatible", model="cheap",
            base_url="https://example.test/v1", max_cost_usd=1,
        ),
        environ={
            "NOTES_EVAL_JUDGE_API_TOKEN": "token",
            "NOTES_EVAL_JUDGE_MIN_INTERVAL_SECONDS": "4.1",
        },
    )
    provider = ScriptedJudgeProvider([{"issues": []}, {"issues": []}])
    gateway = EvaluationJudgeGateway(
        configured, provider, allow_remote_judge=True,
        clock=lambda: current_time[0], sleeper=sleep,
    )

    for _ in range(2):
        gateway.evaluate_issues(
            classification=DataClassification.public_benchmark, candidate="notes",
            reference="reference", estimated_input_tokens=10, max_output_tokens=20,
        )

    assert waits == [pytest.approx(4.1)]


def test_development_runtime_qualification_has_exact_hard_gates() -> None:
    evidence = DevelopmentEvaluationEvidence(
        runtime_fingerprint="a" * 64, profile_fingerprint="b" * 64, prompt_fingerprint="c" * 64,
        schema_fingerprint="d" * 64, fixture_fingerprint="e" * 64,
        probe_requests=16, structured_valid=4, structured_total=4,
        narrative_valid=4, narrative_total=4, narrative_reasoning_leaks=0, narrative_factual_errors=0,
        tool_correct=3, tool_total=3, tool_calls=9,
        injected_critical_detected=4, injected_critical_total=4,
        clean_critical_false_positives=0, clean_critic_total=2, total_requests=49,
    )
    assert qualify_development_runtime(evidence).status == "development_evaluation_qualified"
    with pytest.raises(AuthorizationError):
        qualify_development_runtime(evidence.model_copy(update={"total_requests": 50}))


def test_judge_qualification_enforces_calibration_thresholds() -> None:
    calibration = JudgeCalibration(
        schema_valid=20, schema_total=20, critical_true_positive=19, critical_total=20,
        clean_critical_false_positive=1, clean_total=20, weighted_kappa=0.72,
        pair_order_agree=18, pair_order_total=20, privacy_failures=0,
    )
    assert qualify_judge(calibration).status is JudgeQualificationStatus.qualified
    assert qualify_judge(calibration.model_copy(update={"weighted_kappa": 0.69})).status is JudgeQualificationStatus.diagnostic_unqualified
