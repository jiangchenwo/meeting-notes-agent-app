from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AuditStatus(StrEnum):
    qualified = "qualified"
    diagnostic_unqualified = "diagnostic_unqualified"


class AuditRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    packet_id: str = Field(min_length=1)
    expected_valid: bool
    observed_valid: bool
    schema_valid: bool = True
    privacy_valid: bool = True


class AuditQualification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: AuditStatus
    record_count: int
    correct_count: int


def qualify_auditor(records: list[AuditRecord]) -> AuditQualification:
    if len(records) < 6:
        raise ValueError("auditor qualification requires at least six packets")
    correct = sum(item.expected_valid == item.observed_valid and item.schema_valid and item.privacy_valid for item in records)
    status = AuditStatus.qualified if correct == len(records) else AuditStatus.diagnostic_unqualified
    return AuditQualification(status=status, record_count=len(records), correct_count=correct)
