from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CriticIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^i[0-9]{6}$")
    critic: Literal["deterministic", "claim", "coverage", "structured", "audience", "system"]
    severity: Literal["critical", "warning"]
    category: str = Field(min_length=1)
    block_id: str | None
    claim_id: str | None
    fact_ids: tuple[str, ...]
    message: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def valid_target(self) -> CriticIssue:
        if self.claim_id is not None and self.block_id is None:
            raise ValueError("claim issue also requires a block target")
        if self.critic != "system" and self.block_id is None and not self.fact_ids:
            raise ValueError("critic issue requires a block or fact target")
        return self


class QualityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    disposition: Literal["accepted", "rejected", "review_required"]
    issues: tuple[CriticIssue, ...]
    mandatory_coverage: float = Field(ge=0, le=1)
    total_coverage: float = Field(ge=0, le=1)
    evidence_link_rate: float = Field(ge=0, le=1)
    unsupported_claim_count: int = Field(ge=0)
    critic_failure_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    revision_count: int = Field(ge=0, le=2)

    @model_validator(mode="after")
    def disposition_invariants(self) -> QualityReport:
        warnings = sum(item.severity == "warning" for item in self.issues)
        critic_failures = sum(item.critic == "system" and item.category == "critic_failure" for item in self.issues)
        critical = any(item.severity == "critical" for item in self.issues)
        if self.warning_count != warnings or self.critic_failure_count != critic_failures:
            raise ValueError("quality issue counts do not match issue instances")
        if self.critic_failure_count and self.disposition != "review_required":
            raise ValueError("critic failure requires review_required disposition")
        if self.disposition == "accepted" and (
            critical
            or self.mandatory_coverage != 1
            or self.evidence_link_rate != 1
            or self.unsupported_claim_count
            or self.warning_count > 5
        ):
            raise ValueError("accepted quality report violates deterministic gates")
        if self.disposition == "rejected" and not (
            critical or self.mandatory_coverage < 1 or self.evidence_link_rate < 1 or self.unsupported_claim_count
        ):
            raise ValueError("rejected quality report requires a rejection gate")
        return self
