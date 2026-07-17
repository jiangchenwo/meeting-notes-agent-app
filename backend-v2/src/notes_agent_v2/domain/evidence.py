from __future__ import annotations

from datetime import datetime
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .transcript import Utterance


def canonical_digest(value: object) -> str:
    if isinstance(value, str):
        payload = value.encode()
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    utterance_ids: tuple[str, ...] = Field(min_length=1)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_span(self) -> EvidenceSpan:
        if len(self.utterance_ids) != len(set(self.utterance_ids)):
            raise ValueError("evidence utterance IDs must be unique")
        if any(not item.startswith("u") or len(item) != 7 or not item[1:].isdigit() for item in self.utterance_ids):
            raise ValueError("evidence utterance ID is invalid")
        if not self.quote.strip():
            raise ValueError("evidence quote must not be blank")
        return self


class EvidenceChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^ec[0-9]{6}$")
    utterance_ids: tuple[str, ...] = Field(min_length=1)
    rendered_token_count: int = Field(gt=0)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def digest_matches(self) -> EvidenceChunk:
        if len(self.utterance_ids) != len(set(self.utterance_ids)):
            raise ValueError("evidence chunk utterance IDs must be unique")
        if any(not item.startswith("u") or len(item) != 7 or not item[1:].isdigit() for item in self.utterance_ids):
            raise ValueError("evidence chunk utterance ID is invalid")
        payload = {
            "utterance_ids": list(self.utterance_ids),
            "rendered_token_count": self.rendered_token_count,
        }
        if self.digest != canonical_digest(payload):
            raise ValueError("evidence chunk digest mismatch")
        return self


class ProjectContextRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^pc[0-9]{6}$")
    note_id: str = Field(pattern=r"^n[0-9]{6}$")
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_at: datetime

    @model_validator(mode="after")
    def approved_and_bound(self) -> ProjectContextRecord:
        if not self.title.strip() or not self.content.strip():
            raise ValueError("project context title and content must not be blank")
        if self.approved_at.tzinfo is None or self.approved_at.utcoffset() is None:
            raise ValueError("project context approval must be timezone-aware")
        if self.digest != canonical_digest(self.content):
            raise ValueError("project context digest mismatch")
        return self


class ProjectContextCitation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(pattern=r"^pc[0-9]{6}$")
    quote: str = Field(min_length=1)


class ExtractedFactCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^fc[0-9]{6}$")
    text: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    status: str = Field(min_length=1)
    speaker_ids: tuple[str, ...]
    owner: str | None
    due_text: str | None
    evidence: tuple[EvidenceSpan, ...] = Field(min_length=1)


class Fact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^f[0-9]{6}$")
    text: str = Field(min_length=1)
    kind: Literal["fact", "decision", "action", "proposal", "question", "risk", "correction"]
    status: Literal["asserted", "proposed", "approved", "rejected", "completed", "uncertain"]
    speaker_ids: tuple[str, ...]
    owner: str | None
    due_text: str | None
    confidence: float = Field(ge=0, le=1)
    verification: Literal["supported", "uncertain"]
    evidence: tuple[EvidenceSpan, ...] = Field(min_length=1)
    source_candidate_ids: tuple[str, ...] = Field(min_length=1)
    supersedes_fact_ids: tuple[str, ...]
    conflicts_with_fact_ids: tuple[str, ...]

    @model_validator(mode="after")
    def semantic_invariants(self) -> Fact:
        if not self.text.strip():
            raise ValueError("fact text must not be blank")
        if (self.status == "uncertain") != (self.verification == "uncertain"):
            raise ValueError("uncertain fact status and verification must agree")
        if (self.owner is not None or self.due_text is not None) and self.kind != "action":
            raise ValueError("owner and due text are allowed only on actions")
        references = self.supersedes_fact_ids + self.conflicts_with_fact_ids
        if self.id in references or len(references) != len(set(references)):
            raise ValueError("fact relationships must be unique and cannot reference self")
        return self


def validate_fact_graph(facts: tuple[Fact, ...], utterances: tuple[Utterance, ...]) -> None:
    fact_by_id = {item.id: item for item in facts}
    if len(fact_by_id) != len(facts):
        raise ValueError("fact IDs must be unique")
    expected_fact_ids = tuple(f"f{index:06d}" for index in range(1, len(facts) + 1))
    if tuple(item.id for item in facts) != expected_fact_ids:
        raise ValueError("fact IDs must be unique and monotonically increasing from f000001")
    utterance_by_id = {item.id: item for item in utterances}
    utterance_order = {item.id: index for index, item in enumerate(utterances)}
    first_positions: list[int] = []
    for item in facts:
        for reference in item.supersedes_fact_ids + item.conflicts_with_fact_ids:
            if reference not in fact_by_id:
                raise ValueError("fact relationship references an unknown fact")
        for conflict in item.conflicts_with_fact_ids:
            if item.id not in fact_by_id[conflict].conflicts_with_fact_ids:
                raise ValueError("fact conflict links must be symmetric")
        positions: list[int] = []
        for span in item.evidence:
            try:
                source = "\n".join(utterance_by_id[identifier].text for identifier in span.utterance_ids)
                positions.extend(utterance_order[identifier] for identifier in span.utterance_ids)
            except KeyError as exc:
                raise ValueError("fact evidence references an unknown utterance") from exc
            if span.quote not in source:
                raise ValueError("evidence quote is absent from its source utterances")
        first_positions.append(min(positions))
    if first_positions != sorted(first_positions):
        raise ValueError("facts must be ordered by first evidence")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(identifier: str) -> None:
        if identifier in visiting:
            raise ValueError("fact supersession links must be acyclic")
        if identifier in visited:
            return
        visiting.add(identifier)
        for parent in fact_by_id[identifier].supersedes_fact_ids:
            visit(parent)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in fact_by_id:
        visit(identifier)
