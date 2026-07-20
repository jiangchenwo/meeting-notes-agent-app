from __future__ import annotations

import json
import re
from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from notes_agent_v2.domain.document import DocumentBlock, DocumentClaim, StructuredItem
from notes_agent_v2.domain.evidence import Fact, ProjectContextCitation, ProjectContextRecord
from notes_agent_v2.runtime.tools import ToolPolicy
from notes_agent_v2.workflow.dispatcher import RoleRequest, SafeMessage
from notes_agent_v2.workflow.planner import CapabilityBlock


_CITATION = re.compile(r"\[\[(f[0-9]{6})\]\]")
_EXPLICIT_CLAIM = re.compile(r"^\[(c[0-9]{6})\]\s*")
_ENTITY = re.compile(
    r"https?://[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d[\w:/.%-]*\b|\b[A-Z][a-z]{2,}\b"
)
_ENTITY_ALLOWLIST = frozenset(
    {"A", "An", "And", "As", "At", "For", "From", "In", "It", "Next", "On", "The", "This", "To"}
)
_UNCITED_TRANSITIONS = frozenset({"In summary.", "Next steps follow."})
_STRUCTURED_KIND = {
    "decisions": "decision",
    "actions": "action",
    "risks": "risk",
    "questions": "question",
}


class WriterPolicyError(ValueError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class WriterDispatcher(Protocol):
    def dispatch(self, request: RoleRequest, *, validate): ...


class BlockWriteResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    assignment_id: str = Field(pattern=r"^b[0-9]{3}$")
    status: Literal["ready", "omitted", "writing_failed"]
    block: DocumentBlock | None = None
    error_code: str | None = None

    @model_validator(mode="after")
    def result_shape(self) -> BlockWriteResult:
        if (self.status == "ready") != (self.block is not None):
            raise ValueError("only ready writer results contain a block")
        if (self.status == "writing_failed") != (self.error_code is not None):
            raise ValueError("only failed writer results contain an error code")
        return self


class _StructuredDraftItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str | None = Field(default=None, pattern=r"^si[0-9]{6}$")
    text: str = Field(min_length=1)
    fact_ids: tuple[str, ...] = Field(min_length=1)
    status: str = Field(min_length=1)
    owner: str | None = None
    due_text: str | None = None

    @model_validator(mode="after")
    def unique_facts(self) -> _StructuredDraftItem:
        if len(self.fact_ids) != len(set(self.fact_ids)):
            raise ValueError("structured item fact IDs must be unique")
        return self


class _StructuredDraft(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(min_length=1)
    items: tuple[_StructuredDraftItem, ...]


def writer_tool_policy(
    *, run_id: str, allowed_entity_ids: Sequence[str]
) -> ToolPolicy:
    return ToolPolicy(
        run_id=run_id,
        stage="write",
        allowed_tools=frozenset(
            {
                "get_fact_details",
                "get_project_context",
                "get_generation_constraints",
            }
        ),
        allowed_entity_ids=frozenset(allowed_entity_ids),
        max_rounds=2,
        max_calls=2,
        max_result_tokens=2048,
    )


def write_blocks_serially(
    *,
    run_id: str,
    instruction: str,
    assignments: Sequence[CapabilityBlock],
    facts: Sequence[Fact],
    project_context: Sequence[ProjectContextRecord],
    dispatcher: WriterDispatcher,
    output_limit: int = 8192,
) -> tuple[BlockWriteResult, ...]:
    return tuple(
        _write_block(
            run_id=run_id,
            instruction=instruction,
            assignment=assignment,
            facts=facts,
            project_context=project_context,
            dispatcher=dispatcher,
            output_limit=output_limit,
        )
        for assignment in assignments
    )


def _write_block(
    *,
    run_id: str,
    instruction: str,
    assignment: CapabilityBlock,
    facts: Sequence[Fact],
    project_context: Sequence[ProjectContextRecord],
    dispatcher: WriterDispatcher,
    output_limit: int,
) -> BlockWriteResult:
    structured = assignment.capability in _STRUCTURED_KIND
    if structured and not assignment.fact_ids:
        return BlockWriteResult(assignment_id=assignment.id, status="omitted")
    fact_by_id = {item.id: item for item in facts}
    context_by_id = {item.id: item for item in project_context}
    try:
        assigned_facts = tuple(fact_by_id[item] for item in assignment.fact_ids)
        assigned_context = tuple(
            context_by_id[item] for item in assignment.project_context_ids
        )
    except KeyError:
        return BlockWriteResult(
            assignment_id=assignment.id,
            status="writing_failed",
            error_code="assignment_reference_missing",
        )
    if any(item.verification != "supported" for item in assigned_facts):
        return BlockWriteResult(
            assignment_id=assignment.id,
            status="writing_failed",
            error_code="uncertain_fact_reference",
        )
    authoritative = json.dumps(
        {
            "instruction": instruction,
            "assignment": assignment.model_dump(mode="json"),
            "facts": [item.model_dump(mode="json") for item in assigned_facts],
            "project_context": [
                {
                    "id": item.id,
                    "title": item.title,
                    "content": item.content,
                    "digest": item.digest,
                }
                for item in assigned_context
            ],
        },
        sort_keys=True,
    )
    stage = "write_structured" if structured else "write_narrative"
    profile = "structured_off" if structured else "narrative_reasoned"
    system = (
        "Return one JSON object with title and items. Preserve each supporting "
        "fact's kind, status, owner, due_text, and fact_ids exactly. Do not infer "
        "missing assignments."
        if structured
        else "Write only the assigned block. End every factual sentence or list item "
        "with one or more assigned fact markers such as [[f000001]]. Do not cite "
        "uncertain or unassigned facts."
    )

    def validate(content: str) -> bool:
        try:
            if structured:
                parse_structured_block(
                    content,
                    assignment=assignment,
                    facts=assigned_facts,
                    output_limit=output_limit,
                )
            else:
                parse_cited_narrative(
                    content,
                    assignment=assignment,
                    facts=assigned_facts,
                    project_context=assigned_context,
                    instruction=instruction,
                    output_limit=output_limit,
                )
        except Exception:
            return False
        return True

    request = RoleRequest(
        run_id=run_id,
        stage=stage,
        role="writer",
        profile_name=profile,
        messages=(
            SafeMessage(role="system", content=system),
            SafeMessage(role="user", content=authoritative),
        ),
        allowed_tools=(),
        output_schema=_StructuredDraft.model_json_schema() if structured else None,
    )
    try:
        result = dispatcher.dispatch(request, validate=validate)
        output_tokens = result.response.usage.output_tokens
        block = (
            parse_structured_block(
                result.response.final_content,
                assignment=assignment,
                facts=assigned_facts,
                output_tokens=output_tokens,
                output_limit=output_limit,
            )
            if structured
            else parse_cited_narrative(
                result.response.final_content,
                assignment=assignment,
                facts=assigned_facts,
                project_context=assigned_context,
                instruction=instruction,
                output_tokens=output_tokens,
                output_limit=output_limit,
            )
        )
    except Exception as exc:
        return BlockWriteResult(
            assignment_id=assignment.id,
            status="writing_failed",
            error_code=getattr(exc, "error_code", "invalid_writer_output"),
        )
    if block is None:
        return BlockWriteResult(assignment_id=assignment.id, status="omitted")
    return BlockWriteResult(assignment_id=assignment.id, status="ready", block=block)


def parse_structured_block(
    content: str,
    *,
    assignment: CapabilityBlock,
    facts: Sequence[Fact],
    output_tokens: int = 0,
    output_limit: int = 8192,
) -> DocumentBlock | None:
    _check_output_limit(output_tokens, output_limit)
    kind = _STRUCTURED_KIND.get(assignment.capability)
    if kind is None:
        raise WriterPolicyError("unsupported_structured_capability")
    try:
        draft = _StructuredDraft.model_validate_json(content)
    except ValidationError as exc:
        raise WriterPolicyError("invalid_writer_output") from exc
    if not draft.items:
        if assignment.fact_ids:
            raise WriterPolicyError("assigned_facts_omitted")
        return None
    explicit_ids = [item.id for item in draft.items if item.id is not None]
    if len(explicit_ids) != len(set(explicit_ids)):
        raise WriterPolicyError("duplicate_item_id")
    fact_by_id = {item.id: item for item in facts}
    items: list[StructuredItem] = []
    for index, item in enumerate(draft.items, start=1):
        sources = _resolve_assigned_facts(item.fact_ids, assignment, fact_by_id)
        if any(source.kind != kind for source in sources):
            raise WriterPolicyError("kind_mismatch")
        if item.status not in {source.status for source in sources}:
            raise WriterPolicyError("status_mismatch")
        if item.owner not in {source.owner for source in sources}:
            raise WriterPolicyError("owner_mismatch")
        if item.due_text not in {source.due_text for source in sources}:
            raise WriterPolicyError("due_mismatch")
        _validate_entities(item.text, sources, (), "")
        items.append(
            StructuredItem(
                id=item.id or f"si{index:06d}",
                kind=kind,  # type: ignore[arg-type]
                text=item.text.strip(),
                fact_ids=item.fact_ids,
                status=item.status,
                owner=item.owner,
                due_text=item.due_text,
            )
        )
    return DocumentBlock(
        id=_document_block_id(assignment.id),
        capability=assignment.capability,
        title=draft.title.strip(),
        claims=(),
        structured_items=tuple(items),
    )


def parse_cited_narrative(
    content: str,
    *,
    assignment: CapabilityBlock,
    facts: Sequence[Fact],
    instruction: str,
    project_context: Sequence[ProjectContextRecord] = (),
    context_citations: Sequence[ProjectContextCitation] = (),
    output_tokens: int = 0,
    output_limit: int = 8192,
) -> DocumentBlock:
    _check_output_limit(output_tokens, output_limit)
    if assignment.capability not in {"overview", "narrative", "custom"}:
        raise WriterPolicyError("unsupported_narrative_capability")
    fact_by_id = {item.id: item for item in facts}
    context_by_id = {item.id: item for item in project_context}
    for citation in context_citations:
        if citation.record_id not in assignment.project_context_ids:
            raise WriterPolicyError("unassigned_context_reference")
        record = context_by_id.get(citation.record_id)
        if record is None:
            raise WriterPolicyError("unknown_context_reference")
        if citation.quote not in record.content:
            raise WriterPolicyError("invalid_context_quote")

    title = assignment.purpose
    claims: list[DocumentClaim] = []
    seen_ids: set[str] = set()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                title = heading
            continue
        explicit = _EXPLICIT_CLAIM.match(line)
        claim_id = explicit.group(1) if explicit else f"c{len(claims) + 1:06d}"
        if explicit:
            line = line[explicit.end() :]
        if claim_id in seen_ids:
            raise WriterPolicyError("duplicate_claim_id")
        seen_ids.add(claim_id)
        fact_ids = tuple(dict.fromkeys(_CITATION.findall(line)))
        text = _CITATION.sub("", line).strip()
        if not fact_ids:
            if text in _UNCITED_TRANSITIONS:
                continue
            raise WriterPolicyError("uncited_factual_content")
        sources = _resolve_assigned_facts(fact_ids, assignment, fact_by_id)
        _validate_entities(text, sources, project_context, instruction)
        claims.append(
            DocumentClaim(
                id=claim_id,
                text=text,
                fact_ids=fact_ids,
                project_context_citations=tuple(context_citations),
            )
        )
    if not claims:
        raise WriterPolicyError("empty_required_block")
    return DocumentBlock(
        id=_document_block_id(assignment.id),
        capability=assignment.capability,
        title=title,
        claims=tuple(claims),
        structured_items=(),
    )


def _resolve_assigned_facts(
    identifiers: Sequence[str],
    assignment: CapabilityBlock,
    fact_by_id: dict[str, Fact],
) -> tuple[Fact, ...]:
    resolved: list[Fact] = []
    for identifier in identifiers:
        fact = fact_by_id.get(identifier)
        if fact is None:
            raise WriterPolicyError("unknown_fact_reference")
        if identifier not in assignment.fact_ids:
            raise WriterPolicyError("unassigned_fact_reference")
        if fact.verification != "supported" or fact.status == "uncertain":
            raise WriterPolicyError("uncertain_fact_reference")
        resolved.append(fact)
    return tuple(resolved)


def _validate_entities(
    text: str,
    facts: Sequence[Fact],
    project_context: Sequence[ProjectContextRecord],
    instruction: str,
) -> None:
    corpus = "\n".join(
        [*(item.text for item in facts), *(item.content for item in project_context), instruction]
    ).casefold()
    for match in _ENTITY.findall(text):
        if match in _ENTITY_ALLOWLIST:
            continue
        if match.casefold() not in corpus:
            raise WriterPolicyError("unsupported_entity")


def _check_output_limit(output_tokens: int, output_limit: int) -> None:
    if output_tokens < 0 or output_limit <= 0 or output_tokens > output_limit:
        raise WriterPolicyError("output_limit_exceeded")


def _document_block_id(identifier: str) -> str:
    return f"b{int(identifier[1:]):06d}"
