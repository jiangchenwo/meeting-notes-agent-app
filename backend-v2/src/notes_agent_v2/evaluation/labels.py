from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LabelProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_type: Literal["qmsum", "ami", "meetingbank"]
    release: str = Field(min_length=1)
    license: str = Field(min_length=1)
    upstream_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def nonzero_digest(self) -> LabelProvenance:
        if self.upstream_digest == "0" * 64:
            raise ValueError("upstream digest must not be zero")
        return self


class ReferenceGold(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reference_id: str = Field(min_length=1)
    meeting_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_ids: tuple[str, ...]
    provenance: LabelProvenance
    available_utterance_ids: tuple[str, ...] = Field(exclude=True, repr=False)
    applicability: Literal["applicable", "not_applicable"] = "applicable"
    status: Literal["provisional", "model_assisted_verified"] = "provisional"

    @model_validator(mode="after")
    def evidence_resolves(self) -> ReferenceGold:
        missing = set(self.evidence_ids) - set(self.available_utterance_ids)
        if missing:
            raise ValueError("evidence IDs do not resolve to supplied utterances")
        if self.applicability == "applicable" and not self.evidence_ids:
            raise ValueError("applicable reference requires evidence")
        return self
