import json
from types import SimpleNamespace

import pytest

from notes_agent_v2.evaluation.instruction_effectiveness import (
    InstructionCase,
    InstructionObservation,
    build_instruction_effectiveness_report,
    evaluate_brief_case,
    evaluate_capability_case,
    evaluate_salience_case,
    load_instruction_cases,
)
from notes_agent_v2.runtime.budget import RunBudget


class Gateway:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def call(self, request, *, budget, validate):
        budget.model_requests += 1
        content = json.dumps(self.payloads.pop(0))
        if not validate(content):
            raise ValueError("invalid payload")
        return SimpleNamespace(response=SimpleNamespace(final_content=content))


def _case(identifier: str, cohort: str) -> dict:
    return {
        "case_id": identifier,
        "cohort": cohort,
        "instruction": "Summarize.",
        "fact_index": [
            {
                "fact_id": "f000001",
                "text": "Approved launch.",
                "kind": "decision",
                "status": "approved",
                "confidence": 0.9,
                "verification": "supported",
                "utterance": 1,
                "relevant": True,
                "mandatory_expected": True,
            }
        ],
        "expected_brief": {
            "audience": "general",
            "desired_depth": "standard",
            "constraints": [],
            "requested_emphasis": ["overview", "narrative"],
            "forbidden_content": [],
            "eligible_blocks": ["overview", "narrative"],
        },
        "expected_capabilities": ["overview"],
        "injection_case": False,
        "default_case": True,
    }


def _observation(
    identifier: str,
    feature_id: str,
    *,
    passed: bool = True,
    cohort: str = "authored",
):
    metrics = {
        "schema_validity": 1.0,
        "forbidden_execution_control_fields": 0.0,
        "instruction_attribute_hits": 6.0,
        "instruction_attribute_total": 6.0,
        "exact_default_behavior": 1.0,
        "prompt_injection_policy_violations": 0.0,
        "transcript_instruction_mutations": 0.0,
    }
    if feature_id == "planning.salience_selection":
        metrics = {
            "mandatory_verified_hits": 1.0,
            "mandatory_verified_total": 1.0,
            "uncertain_mandatory_count": 0.0,
            "deterministic_ordering": 1.0,
            "baseline_recall_at_k": 0.5,
            "treatment_recall_at_k": 1.0,
            "baseline_ndcg_at_10": 0.5,
            "treatment_ndcg_at_10": 1.0,
            "status_owner_due_fidelity_regression": 0.0,
        }
    if feature_id == "planning.closed_capability_plan":
        metrics = {
            "schema_validity": 1.0,
            "mandatory_fact_assignment": 1.0,
            "unknown_reference_count": 0.0,
            "execution_authority_widening": 0.0,
            "capability_hits": 1.0,
            "capability_total": 1.0,
            "block_fact_assignment_trace_completeness": 1.0,
        }
    if not passed:
        metrics["schema_validity"] = 0.0
    return InstructionObservation(
        case_id=identifier,
        feature_id=feature_id,
        cohort=cohort,
        valid=passed,
        provider_requests=2 if feature_id != "planning.salience_selection" else 1,
        metrics=metrics,
        error_code=None if passed else "invalid_output",
    )


def test_case_loader_requires_exact_authored_and_public_schedules(tmp_path) -> None:
    payload = {
        "schema_version": "instruction-effectiveness-cases-v1",
        "development_tree_sha256": "a" * 64,
        "cases": [
            *[_case(f"authored-{index:02d}", "authored") for index in range(32)],
            *[_case(f"public-{index:02d}", "public_development") for index in range(12)],
        ],
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload))
    cases, tree_digest, fixture_digest = load_instruction_cases(path)
    assert len(cases) == 44
    assert sum(item.cohort == "authored" for item in cases) == 32
    assert sum(item.cohort == "public_development" for item in cases) == 12
    assert tree_digest == "a" * 64
    assert len(fixture_digest) == 64
    assert all(isinstance(item, InstructionCase) for item in cases)


def test_case_loader_rejects_incomplete_or_duplicate_schedules(tmp_path) -> None:
    payload = {
        "schema_version": "instruction-effectiveness-cases-v1",
        "development_tree_sha256": "a" * 64,
        "cases": [_case("same", "authored")] * 44,
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schedule"):
        load_instruction_cases(path)


@pytest.mark.parametrize(
    "feature_id",
    [
        "planning.generation_brief",
        "planning.salience_selection",
        "planning.closed_capability_plan",
    ],
)
def test_report_passes_complete_live_metrics(feature_id) -> None:
    observations = [
        _observation(
            f"case-{index:02d}",
            feature_id,
            cohort="authored" if index < 32 else "public_development",
        )
        for index in range(44)
    ]
    report = build_instruction_effectiveness_report(
        feature_id,
        observations,
        runtime_fingerprint="b" * 64,
        profile_fingerprint="c" * 64,
        fixture_fingerprint="d" * 64,
        development_fingerprint="e" * 64,
        code_fingerprint="f" * 64,
        prompt_fingerprint="1" * 64,
        schema_fingerprint="2" * 64,
        request_limit=88 if feature_id != "planning.salience_selection" else 44,
    )
    assert report.verdict == "passed"
    assert all(report.hard_gates.values())
    assert report.provider_requests == (
        44 if feature_id == "planning.salience_selection" else 88
    )


def test_live_miss_is_failed_not_blocked() -> None:
    observations = [
        _observation(
            f"case-{index:02d}",
            "planning.generation_brief",
            cohort="authored" if index < 32 else "public_development",
        )
        for index in range(43)
    ] + [
        _observation(
            "case-43",
            "planning.generation_brief",
            passed=False,
            cohort="public_development",
        )
    ]
    report = build_instruction_effectiveness_report(
        "planning.generation_brief",
        observations,
        runtime_fingerprint="b" * 64,
        profile_fingerprint="c" * 64,
        fixture_fingerprint="d" * 64,
        development_fingerprint="e" * 64,
        code_fingerprint="f" * 64,
        prompt_fingerprint="1" * 64,
        schema_fingerprint="2" * 64,
        request_limit=88,
    )
    assert report.verdict == "failed"
    assert "blocked" not in report.model_dump_json()
    assert report.hard_gates["valid_run_rate"] is False


def test_component_observations_measure_real_outputs() -> None:
    case = InstructionCase.model_validate(_case("authored-01", "authored"))
    brief_payload = {
        **case.expected_brief.model_dump(mode="json"),
        "uncertainty": [],
    }
    brief = evaluate_brief_case(
        case,
        Gateway([brief_payload, brief_payload]),
        RunBudget(max_model_requests=2),
    )
    assert brief.valid
    assert brief.provider_requests == 2
    assert brief.metrics["instruction_attribute_hits"] == 6
    assert brief.observed is not None
    assert brief.observed["audience"] == "general"

    salience = evaluate_salience_case(
        case,
        Gateway(
            [
                {
                    "items": [
                        {"fact_id": "f000001", "instruction_relevance": 1.0}
                    ]
                }
            ]
        ),
        RunBudget(max_model_requests=1),
    )
    assert salience.valid
    assert salience.metrics["mandatory_verified_hits"] == 1
    assert salience.observed == {
        "mandatory_fact_ids": ["f000001"],
        "ordered_fact_ids": ["f000001"],
        "scores": {"f000001": 0.97},
    }

    plan_payload = {
        "blocks": [
            {
                "id": "b001",
                "capability": "overview",
                "purpose": "Summarize",
                "fact_ids": ["f000001"],
                "project_context_ids": [],
                "constraints": [],
            }
        ]
    }
    capability = evaluate_capability_case(
        case,
        Gateway([plan_payload, plan_payload]),
        RunBudget(max_model_requests=2),
    )
    assert capability.valid
    assert capability.metrics["capability_hits"] == 2
    assert capability.metrics["capability_total"] == 2
    assert capability.observed is not None
    assert capability.observed["blocks"][0]["capability"] == "overview"
