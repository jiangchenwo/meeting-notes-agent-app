from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .runtime_authorization import (
    AuthorizationError,
    DevelopmentEvaluationAuthorization,
    DevelopmentEvaluationEvidence,
    qualify_development_runtime,
)


QualificationKind = Literal[
    "structured", "narrative", "tool", "critic_injected", "critic_clean"
]


class QualificationCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    case_id: str
    kind: QualificationKind


class QualificationObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    case_id: str
    passed: bool
    provider_requests: int = Field(ge=1, le=2)


def qualification_schedule() -> tuple[QualificationCase, ...]:
    kinds: tuple[tuple[str, QualificationKind, int], ...] = (
        ("structured", "structured", 4),
        ("narrative", "narrative", 4),
        ("tool", "tool", 3),
        ("critic-injected", "critic_injected", 4),
        ("critic-clean", "critic_clean", 2),
    )
    return tuple(
        QualificationCase(case_id=f"{prefix}-{index:02d}", kind=kind)
        for prefix, kind, count in kinds
        for index in range(1, count + 1)
    )


def build_development_authorization(
    observations: list[QualificationObservation],
    *,
    runtime_fingerprint: str,
    profile_fingerprint: str,
    prompt_fingerprint: str,
    schema_fingerprint: str,
    fixture_fingerprint: str,
    probe_requests: int,
) -> DevelopmentEvaluationAuthorization:
    schedule = qualification_schedule()
    expected = {case.case_id: case for case in schedule}
    observed = {item.case_id: item for item in observations}
    if len(observed) != len(observations) or set(observed) != set(expected):
        raise AuthorizationError("qualification observations are incomplete or duplicated")
    if any(not item.passed for item in observations):
        raise AuthorizationError("qualification observation failed")
    counts = {
        kind: sum(case.kind == kind for case in schedule)
        for kind in ("structured", "narrative", "tool", "critic_injected", "critic_clean")
    }
    evidence = DevelopmentEvaluationEvidence(
        runtime_fingerprint=runtime_fingerprint,
        profile_fingerprint=profile_fingerprint,
        prompt_fingerprint=prompt_fingerprint,
        schema_fingerprint=schema_fingerprint,
        fixture_fingerprint=fixture_fingerprint,
        probe_requests=probe_requests,
        structured_valid=counts["structured"],
        structured_total=counts["structured"],
        narrative_valid=counts["narrative"],
        narrative_total=counts["narrative"],
        narrative_reasoning_leaks=0,
        narrative_factual_errors=0,
        tool_correct=counts["tool"],
        tool_total=counts["tool"],
        tool_calls=counts["tool"],
        injected_critical_detected=counts["critic_injected"],
        injected_critical_total=counts["critic_injected"],
        clean_critical_false_positives=0,
        clean_critic_total=counts["critic_clean"],
        total_requests=probe_requests + sum(item.provider_requests for item in observations),
    )
    return qualify_development_runtime(evidence)
