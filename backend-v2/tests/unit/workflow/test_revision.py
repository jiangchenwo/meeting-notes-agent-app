from __future__ import annotations

from types import SimpleNamespace

import pytest

from notes_agent_v2.domain.document import DocumentBlock, DocumentClaim, NotesDocument
from notes_agent_v2.domain.evidence import EvidenceSpan, Fact
from notes_agent_v2.domain.quality import CriticIssue
from notes_agent_v2.workflow.revision import (
    RevisionError,
    apply_targeted_revision,
    revise_document,
    revision_scope,
    revision_tool_policy,
)


def _fact(identifier: str) -> Fact:
    text = f"Fact {identifier}"
    return Fact(
        id=identifier,
        text=text,
        kind="fact",
        status="asserted",
        speaker_ids=("s1",),
        owner=None,
        due_text=None,
        confidence=1,
        verification="supported",
        evidence=(EvidenceSpan(utterance_ids=("u000001",), quote=text),),
        source_candidate_ids=("fc000001",),
        supersedes_fact_ids=(),
        conflicts_with_fact_ids=(),
    )


def _block(identifier: str, fact_id: str, text: str | None = None) -> DocumentBlock:
    return DocumentBlock(
        id=identifier,
        capability="overview" if identifier == "b000001" else "narrative",
        title="Overview" if identifier == "b000001" else "Details",
        claims=(
            DocumentClaim(
                id="c000001" if identifier == "b000001" else "c000002",
                text=text or f"Fact {fact_id}",
                fact_ids=(fact_id,),
                project_context_citations=(),
            ),
        ),
        structured_items=(),
    )


def _document(*, version: int = 1) -> NotesDocument:
    return NotesDocument(
        id=f"d{version:06d}",
        run_id="r000001",
        version=version,
        parent_id=None if version == 1 else f"d{version - 1:06d}",
        title="Notes",
        blocks=(
            _block("b000001", "f000001"),
            _block("b000002", "f000002"),
        ),
    )


def _issue(*, critic: str = "claim", block_id: str | None = "b000001") -> CriticIssue:
    return CriticIssue(
        id="i000001",
        critic=critic,
        severity="critical",
        category="critic_failure" if critic == "system" else "contradiction",
        block_id=block_id,
        claim_id="c000001" if block_id else None,
        fact_ids=("f000001",) if block_id else (),
        message="Issue",
        confidence=None,
    )


def test_revision_scope_and_tool_policy_are_exact() -> None:
    scope = revision_scope((_issue(),))
    assert scope.block_ids == ("b000001",)
    assert scope.fact_ids == ("f000001",)
    policy = revision_tool_policy(run_id="r000001", scope=scope)
    assert policy.allowed_entity_ids == frozenset({"b000001", "f000001"})
    assert policy.allowed_tools == frozenset(
        {"get_fact_details", "get_project_context", "get_generation_constraints"}
    )
    assert (policy.max_rounds, policy.max_calls, policy.max_result_tokens) == (2, 2, 2048)


def test_targeted_revision_keeps_unchanged_blocks_byte_identical_and_links_parent() -> None:
    parent = _document()
    replacement = _block("b000001", "f000001", text="Revised fact f000001")
    revised = apply_targeted_revision(
        parent=parent,
        document_id="d000002",
        issues=(_issue(),),
        revised_blocks={"b000001": replacement},
        facts=(_fact("f000001"), _fact("f000002")),
        mandatory_fact_ids=("f000001", "f000002"),
    )
    assert (revised.version, revised.parent_id) == (2, parent.id)
    assert revised.blocks[0].claims[0].text == "Revised fact f000001"
    assert revised.blocks[1].model_dump_json() == parent.blocks[1].model_dump_json()


def test_revision_rejects_scope_widening_mandatory_deletion_and_third_revision() -> None:
    parent = _document()
    with pytest.raises(RevisionError, match="revision_scope_mismatch"):
        apply_targeted_revision(
            parent=parent,
            document_id="d000002",
            issues=(_issue(),),
            revised_blocks={"b000002": _block("b000002", "f000002", "Changed")},
            facts=(_fact("f000001"), _fact("f000002")),
            mandatory_fact_ids=("f000001", "f000002"),
        )
    with pytest.raises(RevisionError, match="mandatory_fact_deleted"):
        apply_targeted_revision(
            parent=parent,
            document_id="d000002",
            issues=(_issue(),),
            revised_blocks={"b000001": _block("b000001", "f000002", "Wrong scope")},
            facts=(_fact("f000001"), _fact("f000002")),
            mandatory_fact_ids=("f000001", "f000002"),
        )
    with pytest.raises(RevisionError, match="revision_limit_exceeded"):
        apply_targeted_revision(
            parent=_document(version=3),
            document_id="d000004",
            issues=(_issue(),),
            revised_blocks={"b000001": _block("b000001", "f000001", "Changed")},
            facts=(_fact("f000001"), _fact("f000002")),
            mandatory_fact_ids=("f000001", "f000002"),
        )


def test_critic_failure_never_triggers_automatic_revision_and_parent_is_unchanged() -> None:
    parent = _document()
    before = parent.model_dump_json()
    with pytest.raises(RevisionError, match="critic_failure_requires_review"):
        apply_targeted_revision(
            parent=parent,
            document_id="d000002",
            issues=(_issue(critic="system", block_id=None),),
            revised_blocks={},
            facts=(_fact("f000001"), _fact("f000002")),
            mandatory_fact_ids=("f000001", "f000002"),
        )
    assert parent.model_dump_json() == before


def test_reviser_dispatches_only_affected_blocks_and_preserves_parent_on_failure() -> None:
    parent = _document()

    class Dispatcher:
        def __init__(self, content: str | Exception) -> None:
            self.content = content
            self.requests = []

        def dispatch(self, request, *, validate):
            self.requests.append(request)
            if isinstance(self.content, Exception):
                raise self.content
            assert validate(self.content)
            return SimpleNamespace(
                response=SimpleNamespace(
                    final_content=self.content,
                    usage=SimpleNamespace(output_tokens=20),
                )
            )

    dispatcher = Dispatcher("# Overview\nFact f000001 [[f000001]]")
    result = revise_document(
        parent=parent,
        document_id="d000002",
        issues=(_issue(),),
        facts=(_fact("f000001"), _fact("f000002")),
        mandatory_fact_ids=("f000001", "f000002"),
        instruction="Summarize the meeting.",
        project_context=(),
        dispatcher=dispatcher,
    )
    assert result.status == "revised"
    assert result.document.parent_id == parent.id
    assert result.document.blocks[1].model_dump_json() == parent.blocks[1].model_dump_json()
    assert len(dispatcher.requests) == 1
    request = dispatcher.requests[0]
    assert (request.role, request.stage, request.profile_name) == (
        "reviser",
        "revise",
        "tool_reasoned",
    )
    assert set(request.allowed_entity_ids) == {"b000001", "f000001"}

    failed = revise_document(
        parent=parent,
        document_id="d000002",
        issues=(_issue(),),
        facts=(_fact("f000001"), _fact("f000002")),
        mandatory_fact_ids=("f000001", "f000002"),
        instruction="Summarize the meeting.",
        project_context=(),
        dispatcher=Dispatcher(RuntimeError("timeout")),
    )
    assert failed.status == "revision_failed"
    assert failed.document is parent
    assert failed.error_code == "revision_dispatch_failed"
