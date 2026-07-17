from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AuthorizationError(RuntimeError):
    pass


class DevelopmentEvaluationEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    runtime_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    probe_requests: int
    structured_valid: int
    structured_total: int
    narrative_valid: int
    narrative_total: int
    narrative_reasoning_leaks: int
    narrative_factual_errors: int
    tool_correct: int
    tool_total: int
    tool_calls: int
    injected_critical_detected: int
    injected_critical_total: int
    clean_critical_false_positives: int
    clean_critic_total: int
    total_requests: int


class DevelopmentEvaluationAuthorization(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal["development_evaluation_qualified"]
    evidence: DevelopmentEvaluationEvidence


def qualify_development_runtime(evidence: DevelopmentEvaluationEvidence) -> DevelopmentEvaluationAuthorization:
    gates = (
        evidence.probe_requests <= 16,
        (evidence.structured_valid, evidence.structured_total) == (4, 4),
        (evidence.narrative_valid, evidence.narrative_total) == (4, 4),
        evidence.narrative_reasoning_leaks == evidence.narrative_factual_errors == 0,
        (evidence.tool_correct, evidence.tool_total) == (3, 3), evidence.tool_calls <= 9,
        (evidence.injected_critical_detected, evidence.injected_critical_total) == (4, 4),
        evidence.clean_critical_false_positives == 0 and evidence.clean_critic_total == 2,
        evidence.total_requests <= 49,
    )
    if not all(gates):
        raise AuthorizationError("development runtime qualification gates failed")
    return DevelopmentEvaluationAuthorization(status="development_evaluation_qualified", evidence=evidence)
