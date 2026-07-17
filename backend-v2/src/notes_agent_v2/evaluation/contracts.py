from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .labels import LabelProvenance


class EvaluationContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Utterance(EvaluationContract):
    id: str = Field(min_length=1)
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def ordered_times(self) -> Utterance:
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("utterance time range is reversed")
        return self


class ImportedMeeting(EvaluationContract):
    meeting_id: str = Field(min_length=1)
    source_type: Literal["qmsum", "ami", "meetingbank"]
    utterances: tuple[Utterance, ...] = Field(min_length=1)
    provenance: LabelProvenance
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    canonical_digest: str = ""

    @model_validator(mode="after")
    def derive_digest(self) -> ImportedMeeting:
        ids = [item.id for item in self.utterances]
        if len(ids) != len(set(ids)):
            raise ValueError("utterance IDs must be unique")
        payload = self.model_dump(mode="json", exclude={"canonical_digest"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        object.__setattr__(self, "canonical_digest", hashlib.sha256(canonical.encode()).hexdigest())
        return self


class ImportedReference(EvaluationContract):
    reference_id: str = Field(min_length=1)
    meeting_id: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    query: str | None = None
