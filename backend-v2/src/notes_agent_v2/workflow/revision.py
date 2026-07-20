from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from notes_agent_v2.domain.document import (
    DocumentBlock,
    NotesDocument,
    validate_document_integrity,
)
from notes_agent_v2.domain.evidence import Fact, ProjectContextRecord
from notes_agent_v2.domain.quality import CriticIssue
from notes_agent_v2.runtime.tools import ToolPolicy
from notes_agent_v2.workflow.dispatcher import RoleRequest, SafeMessage
from notes_agent_v2.workflow.planner import CapabilityBlock
from notes_agent_v2.workflow.writers import (
    parse_cited_narrative,
    parse_structured_block,
)


class RevisionError(ValueError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class RevisionScope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    block_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]


class RevisionDispatcher(Protocol):
    def dispatch(self, request: RoleRequest, *, validate): ...


class RevisionAttempt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["revised", "revision_failed", "review_required"]
    document: NotesDocument
    error_code: str | None = None


def revision_scope(issues: Sequence[CriticIssue]) -> RevisionScope:
    block_ids = tuple(
        dict.fromkeys(item.block_id for item in issues if item.block_id is not None)
    )
    fact_ids = tuple(
        dict.fromkeys(identifier for item in issues for identifier in item.fact_ids)
    )
    return RevisionScope(block_ids=block_ids, fact_ids=fact_ids)


def revision_tool_policy(*, run_id: str, scope: RevisionScope) -> ToolPolicy:
    return ToolPolicy(
        run_id=run_id,
        stage="revise",
        allowed_tools=frozenset(
            {
                "get_fact_details",
                "get_project_context",
                "get_generation_constraints",
            }
        ),
        allowed_entity_ids=frozenset((*scope.block_ids, *scope.fact_ids)),
        max_rounds=2,
        max_calls=2,
        max_result_tokens=2048,
    )


def revise_document(
    *,
    parent: NotesDocument,
    document_id: str,
    issues: Sequence[CriticIssue],
    facts: Sequence[Fact],
    mandatory_fact_ids: Sequence[str],
    instruction: str,
    project_context: Sequence[ProjectContextRecord],
    dispatcher: RevisionDispatcher,
    output_limit: int = 8192,
    note_id: str = "n000001",
) -> RevisionAttempt:
    if any(
        item.critic == "system" and item.category == "critic_failure"
        for item in issues
    ):
        return RevisionAttempt(
            status="review_required",
            document=parent,
            error_code="critic_failure_requires_review",
        )
    if parent.version >= 3:
        return RevisionAttempt(
            status="revision_failed",
            document=parent,
            error_code="revision_limit_exceeded",
        )
    scope = revision_scope(issues)
    parent_by_id = {item.id: item for item in parent.blocks}
    if not scope.block_ids or any(item not in parent_by_id for item in scope.block_ids):
        return RevisionAttempt(
            status="revision_failed",
            document=parent,
            error_code="revision_scope_mismatch",
        )
    fact_by_id = {item.id: item for item in facts}
    context_by_id = {item.id: item for item in project_context}
    revised_blocks: dict[str, DocumentBlock] = {}
    try:
        for block_id in scope.block_ids:
            block = parent_by_id[block_id]
            block_fact_ids = tuple(
                dict.fromkeys(
                    identifier
                    for output in (*block.claims, *block.structured_items)
                    for identifier in output.fact_ids
                )
            )
            allowed_fact_ids = tuple(dict.fromkeys((*scope.fact_ids, *block_fact_ids)))
            assigned_facts = tuple(fact_by_id[item] for item in allowed_fact_ids)
            context_ids = tuple(
                dict.fromkeys(
                    citation.record_id
                    for claim in block.claims
                    for citation in claim.project_context_citations
                )
            )
            assigned_context = tuple(context_by_id[item] for item in context_ids)
            assignment = CapabilityBlock(
                id=f"b{int(block.id[1:]):03d}",
                capability=block.capability,
                purpose=block.title,
                fact_ids=allowed_fact_ids,
                project_context_ids=context_ids,
                constraints=() if block.capability != "custom" else (instruction,),
            )
            effective_scope = RevisionScope(
                block_ids=(block_id,), fact_ids=allowed_fact_ids
            )
            policy = revision_tool_policy(run_id=parent.run_id, scope=effective_scope)
            structured = block.capability in {
                "decisions",
                "actions",
                "risks",
                "questions",
            }
            payload = json.dumps(
                {
                    "instruction": instruction,
                    "issues": [
                        item.model_dump(mode="json")
                        for item in issues
                        if item.block_id == block_id
                    ],
                    "current_block": block.model_dump(mode="json"),
                    "facts": [item.model_dump(mode="json") for item in assigned_facts],
                    "project_context": [
                        item.model_dump(mode="json") for item in assigned_context
                    ],
                },
                sort_keys=True,
            )

            def parse(content: str) -> DocumentBlock | None:
                if structured:
                    return parse_structured_block(
                        content,
                        assignment=assignment,
                        facts=assigned_facts,
                        output_limit=output_limit,
                    )
                return parse_cited_narrative(
                    content,
                    assignment=assignment,
                    facts=assigned_facts,
                    project_context=assigned_context,
                    instruction=instruction,
                    output_limit=output_limit,
                )

            request = RoleRequest(
                run_id=parent.run_id,
                stage="revise",
                role="reviser",
                profile_name="tool_reasoned",
                messages=(
                    SafeMessage(
                        role="system",
                        content=(
                            "Revise only the supplied block to resolve the listed issues. "
                            "Preserve all supported facts and cite every narrative claim "
                            "with assigned markers such as [[f000001]]."
                        ),
                    ),
                    SafeMessage(role="user", content=payload),
                ),
                allowed_tools=tuple(sorted(policy.allowed_tools)),
                allowed_entity_ids=tuple(sorted(policy.allowed_entity_ids)),
                max_tool_rounds=policy.max_rounds,
                max_tool_calls=policy.max_calls,
                max_tool_result_tokens=policy.max_result_tokens,
            )
            response = dispatcher.dispatch(
                request, validate=lambda content: _valid_revision(content, parse)
            ).response
            replacement = parse(response.final_content)
            if replacement is None:
                raise RevisionError("revised_block_omitted")
            revised_blocks[block_id] = replacement.model_copy(
                update={"id": block_id}
            )
        revised = apply_targeted_revision(
            parent=parent,
            document_id=document_id,
            issues=issues,
            revised_blocks=revised_blocks,
            facts=facts,
            mandatory_fact_ids=mandatory_fact_ids,
            project_context=project_context,
            note_id=note_id,
        )
    except Exception as exc:
        return RevisionAttempt(
            status="revision_failed",
            document=parent,
            error_code=getattr(exc, "error_code", "revision_dispatch_failed"),
        )
    return RevisionAttempt(status="revised", document=revised)


def _valid_revision(content: str, parse) -> bool:
    try:
        return parse(content) is not None
    except Exception:
        return False


def apply_targeted_revision(
    *,
    parent: NotesDocument,
    document_id: str,
    issues: Sequence[CriticIssue],
    revised_blocks: Mapping[str, DocumentBlock],
    facts: Sequence[Fact],
    mandatory_fact_ids: Sequence[str],
    project_context: Sequence[ProjectContextRecord] = (),
    note_id: str = "n000001",
) -> NotesDocument:
    if any(
        item.critic == "system" and item.category == "critic_failure"
        for item in issues
    ):
        raise RevisionError("critic_failure_requires_review")
    if parent.version >= 3:
        raise RevisionError("revision_limit_exceeded")
    scope = revision_scope(issues)
    if not scope.block_ids or set(revised_blocks) != set(scope.block_ids):
        raise RevisionError("revision_scope_mismatch")
    parent_by_id = {item.id: item for item in parent.blocks}
    if any(identifier not in parent_by_id for identifier in scope.block_ids):
        raise RevisionError("unknown_revision_block")
    blocks: list[DocumentBlock] = []
    for block in parent.blocks:
        replacement = revised_blocks.get(block.id)
        if replacement is None:
            blocks.append(block)
            continue
        if replacement.id != block.id or replacement.capability != block.capability:
            raise RevisionError("revision_block_identity_changed")
        blocks.append(replacement)
    covered = {
        identifier
        for block in blocks
        for output in (*block.claims, *block.structured_items)
        for identifier in output.fact_ids
    }
    if any(identifier not in covered for identifier in mandatory_fact_ids):
        raise RevisionError("mandatory_fact_deleted")
    allowed_facts = set(scope.fact_ids)
    for identifier in scope.block_ids:
        source = parent_by_id[identifier]
        allowed_facts.update(
            fact_id
            for output in (*source.claims, *source.structured_items)
            for fact_id in output.fact_ids
        )
    revised_fact_ids = {
        fact_id
        for identifier in scope.block_ids
        for output in (
            *revised_blocks[identifier].claims,
            *revised_blocks[identifier].structured_items,
        )
        for fact_id in output.fact_ids
    }
    if not revised_fact_ids.issubset(allowed_facts):
        raise RevisionError("revision_scope_widening")
    revised = NotesDocument(
        id=document_id,
        run_id=parent.run_id,
        version=parent.version + 1,
        parent_id=parent.id,
        title=parent.title,
        blocks=tuple(blocks),
    )
    try:
        validate_document_integrity(
            revised,
            facts=tuple(facts),
            project_context=tuple(project_context),
            note_id=note_id,
        )
    except ValueError as exc:
        raise RevisionError("revised_document_invalid") from exc
    for previous, current in zip(parent.blocks, revised.blocks, strict=True):
        if previous.id not in scope.block_ids and previous.model_dump_json() != current.model_dump_json():
            raise RevisionError("unchanged_block_modified")
    return revised
