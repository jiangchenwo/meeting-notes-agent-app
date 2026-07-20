from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from notes_agent_v2.domain.document import (
    DocumentBlock,
    DocumentClaim,
    NotesDocument,
    StructuredItem,
    validate_document_integrity,
)
from notes_agent_v2.domain.evidence import Fact, ProjectContextRecord
from notes_agent_v2.workflow.planner import CapabilityPlan


class AssemblyError(ValueError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class SourceMapEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    output_id: str = Field(min_length=1)
    fact_ids: tuple[str, ...]
    utterance_ids: tuple[str, ...]
    quotes: tuple[str, ...]
    project_context_ids: tuple[str, ...]


class AssemblyResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document: NotesDocument
    source_map: tuple[SourceMapEntry, ...]
    display_markdown: str


def assemble_document(
    *,
    run_id: str,
    document_id: str,
    title: str,
    plan: CapabilityPlan,
    written_blocks: Mapping[str, DocumentBlock | None],
    facts: Sequence[Fact],
    project_context: Sequence[ProjectContextRecord],
    note_id: str = "n000001",
    version: int = 1,
    parent_id: str | None = None,
) -> AssemblyResult:
    planned_ids = {item.id for item in plan.blocks}
    if set(written_blocks) != planned_ids:
        raise AssemblyError("block_result_mismatch")
    ordered = [written_blocks[item.id] for item in plan.blocks]
    present = [item for item in ordered if item is not None]
    if not any(item.capability in {"overview", "narrative"} for item in present):
        raise AssemblyError("required_narrative_missing")
    normalized = _renumber(present)
    document = NotesDocument(
        id=document_id,
        run_id=run_id,
        version=version,
        parent_id=parent_id,
        title=title.strip(),
        blocks=normalized,
    )
    try:
        validate_document_integrity(
            document,
            facts=tuple(facts),
            project_context=tuple(project_context),
            note_id=note_id,
        )
    except ValueError as exc:
        raise AssemblyError("document_integrity_failure") from exc
    source_map = _source_map(document, facts)
    return AssemblyResult(
        document=document,
        source_map=source_map,
        display_markdown=_render(document),
    )


def _renumber(blocks: Sequence[DocumentBlock]) -> tuple[DocumentBlock, ...]:
    claim_sequence = 0
    item_sequence = 0
    normalized: list[DocumentBlock] = []
    for block_sequence, block in enumerate(blocks, start=1):
        claims: list[DocumentClaim] = []
        for claim in block.claims:
            claim_sequence += 1
            claims.append(claim.model_copy(update={"id": f"c{claim_sequence:06d}"}))
        items: list[StructuredItem] = []
        for item in block.structured_items:
            item_sequence += 1
            items.append(item.model_copy(update={"id": f"si{item_sequence:06d}"}))
        normalized.append(
            block.model_copy(
                update={
                    "id": f"b{block_sequence:06d}",
                    "claims": tuple(claims),
                    "structured_items": tuple(items),
                }
            )
        )
    return tuple(normalized)


def _source_map(
    document: NotesDocument, facts: Sequence[Fact]
) -> tuple[SourceMapEntry, ...]:
    fact_by_id = {item.id: item for item in facts}
    entries: list[SourceMapEntry] = []
    for block in document.blocks:
        for output in (*block.claims, *block.structured_items):
            sources = [fact_by_id[identifier] for identifier in output.fact_ids]
            entries.append(
                SourceMapEntry(
                    output_id=output.id,
                    fact_ids=output.fact_ids,
                    utterance_ids=tuple(
                        dict.fromkeys(
                            identifier
                            for fact in sources
                            for span in fact.evidence
                            for identifier in span.utterance_ids
                        )
                    ),
                    quotes=tuple(
                        dict.fromkeys(span.quote for fact in sources for span in fact.evidence)
                    ),
                    project_context_ids=(
                        tuple(
                            dict.fromkeys(
                                citation.record_id
                                for citation in output.project_context_citations
                            )
                        )
                        if isinstance(output, DocumentClaim)
                        else ()
                    ),
                )
            )
    return tuple(entries)


def _render(document: NotesDocument) -> str:
    lines = [f"# {document.title}"]
    for block in document.blocks:
        lines.extend(("", f"## {block.title}"))
        lines.extend(claim.text for claim in block.claims)
        for item in block.structured_items:
            details = [item.text, f"status: {item.status}"]
            if item.owner is not None:
                details.append(f"owner: {item.owner}")
            if item.due_text is not None:
                details.append(f"due: {item.due_text}")
            lines.append(f"- {'; '.join(details)}")
    return "\n".join(lines).strip() + "\n"
