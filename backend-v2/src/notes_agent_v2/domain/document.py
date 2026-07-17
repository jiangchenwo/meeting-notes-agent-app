from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence import Fact, ProjectContextCitation, ProjectContextRecord


class DocumentClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^c[0-9]{6}$")
    text: str = Field(min_length=1)
    fact_ids: tuple[str, ...]
    project_context_citations: tuple[ProjectContextCitation, ...]

    @model_validator(mode="after")
    def has_support(self) -> DocumentClaim:
        if not self.text.strip():
            raise ValueError("document claim must not be blank")
        if not self.fact_ids and not self.project_context_citations:
            raise ValueError("document claim requires meeting facts or approved context")
        return self


class StructuredItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^si[0-9]{6}$")
    kind: Literal["decision", "action", "risk", "question"]
    text: str = Field(min_length=1)
    fact_ids: tuple[str, ...] = Field(min_length=1)
    status: str = Field(min_length=1)
    owner: str | None
    due_text: str | None


class DocumentBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^b[0-9]{6}$")
    capability: str = Field(min_length=1)
    title: str = Field(min_length=1)
    claims: tuple[DocumentClaim, ...]
    structured_items: tuple[StructuredItem, ...]


class NotesDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^d[0-9]{6}$")
    run_id: str = Field(pattern=r"^r[0-9]{6}$")
    version: int = Field(gt=0)
    parent_id: str | None
    title: str = Field(min_length=1)
    blocks: tuple[DocumentBlock, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def version_integrity(self) -> NotesDocument:
        if (self.version == 1) != (self.parent_id is None):
            raise ValueError("first document version has no parent; later versions require one")
        block_ids = [item.id for item in self.blocks]
        claim_ids = [claim.id for block in self.blocks for claim in block.claims]
        item_ids = [item.id for block in self.blocks for item in block.structured_items]
        if any(len(values) != len(set(values)) for values in (block_ids, claim_ids, item_ids)):
            raise ValueError("document block, claim, and structured item IDs must be unique")
        return self


def validate_document_integrity(
    document: NotesDocument,
    *,
    facts: tuple[Fact, ...],
    project_context: tuple[ProjectContextRecord, ...],
    note_id: str,
) -> None:
    fact_by_id = {item.id: item for item in facts}
    context_by_id = {item.id: item for item in project_context}
    for block in document.blocks:
        for claim in block.claims:
            for identifier in claim.fact_ids:
                if identifier not in fact_by_id or fact_by_id[identifier].verification != "supported":
                    raise ValueError("document claim references an unavailable supported fact")
            for citation in claim.project_context_citations:
                record = context_by_id.get(citation.record_id)
                if record is None or record.note_id != note_id:
                    raise ValueError("document claim references project context from another note")
                if citation.quote not in record.content:
                    raise ValueError("project context citation quote is absent from its record")
        for item in block.structured_items:
            if not item.fact_ids:
                raise ValueError("structured item requires meeting fact support")
            for identifier in item.fact_ids:
                source = fact_by_id.get(identifier)
                if source is None or source.verification != "supported":
                    raise ValueError("structured item references an unavailable supported fact")
                if item.kind in {"decision", "action"} and source.kind != item.kind:
                    raise ValueError("structured item kind does not match its fact")
