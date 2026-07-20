from __future__ import annotations

import pytest

from notes_agent_v2.domain.document import DocumentBlock, DocumentClaim, StructuredItem
from notes_agent_v2.domain.evidence import EvidenceSpan, Fact
from notes_agent_v2.workflow.assembly import AssemblyError, assemble_document
from notes_agent_v2.workflow.planner import CapabilityBlock, CapabilityPlan


def _fact(identifier: str, text: str, *, kind: str = "fact") -> Fact:
    return Fact(
        id=identifier,
        text=text,
        kind=kind,
        status="approved" if kind in {"decision", "action"} else "asserted",
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


def _plan() -> CapabilityPlan:
    return CapabilityPlan(
        blocks=(
            CapabilityBlock(
                id="b001",
                capability="overview",
                purpose="Overview",
                fact_ids=("f000001",),
                project_context_ids=(),
                constraints=(),
            ),
            CapabilityBlock(
                id="b002",
                capability="risks",
                purpose="Risks",
                fact_ids=(),
                project_context_ids=(),
                constraints=(),
            ),
            CapabilityBlock(
                id="b003",
                capability="decisions",
                purpose="Decisions",
                fact_ids=("f000002",),
                project_context_ids=(),
                constraints=(),
            ),
        )
    )


def test_assembly_is_ordered_omits_empty_optional_and_rebuilds_source_map() -> None:
    facts = (
        _fact("f000001", "The rollout begins Friday."),
        _fact("f000002", "The proposal was approved.", kind="decision"),
    )
    overview = DocumentBlock(
        id="b000099",
        capability="overview",
        title="Overview",
        claims=(
            DocumentClaim(
                id="c000099",
                text="The rollout begins Friday.",
                fact_ids=("f000001",),
                project_context_citations=(),
            ),
        ),
        structured_items=(),
    )
    decisions = DocumentBlock(
        id="b000098",
        capability="decisions",
        title="Decisions",
        claims=(),
        structured_items=(
            StructuredItem(
                id="si000099",
                kind="decision",
                text="The proposal was approved.",
                fact_ids=("f000002",),
                status="approved",
                owner=None,
                due_text=None,
            ),
        ),
    )
    result = assemble_document(
        run_id="r000001",
        document_id="d000001",
        title="Meeting notes",
        plan=_plan(),
        written_blocks={"b003": decisions, "b002": None, "b001": overview},
        facts=facts,
        project_context=(),
    )
    assert [block.capability for block in result.document.blocks] == ["overview", "decisions"]
    assert [block.id for block in result.document.blocks] == ["b000001", "b000002"]
    assert result.document.blocks[0].claims[0].id == "c000001"
    assert result.document.blocks[1].structured_items[0].id == "si000001"
    assert result.source_map[0].utterance_ids == ("u000001",)
    assert "[[" not in result.display_markdown
    assert "The rollout begins Friday." in result.display_markdown


def test_assembly_requires_overview_or_narrative_output() -> None:
    decision = _fact("f000002", "The proposal was approved.", kind="decision")
    with pytest.raises(AssemblyError, match="required_narrative_missing"):
        assemble_document(
            run_id="r000001",
            document_id="d000001",
            title="Meeting notes",
            plan=_plan(),
            written_blocks={
                "b001": None,
                "b002": None,
                "b003": DocumentBlock(
                    id="b000001",
                    capability="decisions",
                    title="Decisions",
                    claims=(),
                    structured_items=(
                        StructuredItem(
                            id="si000001",
                            kind="decision",
                            text=decision.text,
                            fact_ids=(decision.id,),
                            status="approved",
                            owner=None,
                            due_text=None,
                        ),
                    ),
                ),
            },
            facts=(decision,),
            project_context=(),
        )


def test_assembly_rejects_missing_or_extra_block_results() -> None:
    fact = _fact("f000001", "The rollout begins Friday.")
    block = DocumentBlock(
        id="b000001",
        capability="overview",
        title="Overview",
        claims=(
            DocumentClaim(
                id="c000001",
                text=fact.text,
                fact_ids=(fact.id,),
                project_context_citations=(),
            ),
        ),
        structured_items=(),
    )
    for written in ({"b001": block}, {"b001": block, "b002": None, "b003": None, "b999": block}):
        with pytest.raises(AssemblyError, match="block_result_mismatch"):
            assemble_document(
                run_id="r000001",
                document_id="d000001",
                title="Meeting notes",
                plan=_plan(),
                written_blocks=written,
                facts=(fact,),
                project_context=(),
            )
