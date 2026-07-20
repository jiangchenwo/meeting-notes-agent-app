from __future__ import annotations

import pytest
from pydantic import ValidationError

from notes_agent_v2.workflow.audience import GenerationBrief
from notes_agent_v2.workflow.outline import (
    DocumentOutline,
    OutlineClaim,
    OutlinePolicyError,
    build_fact_covered_outline,
    outline_tool_policy,
)
from notes_agent_v2.workflow.planner import CapabilityBlock, CapabilityPlan
from notes_agent_v2.workflow.salience import SalienceRecord


def _brief(depth: str = "standard") -> GenerationBrief:
    return GenerationBrief(
        audience="engineering leads",
        desired_depth=depth,
        constraints=(),
        requested_emphasis=("overview", "decisions", "actions"),
        forbidden_content=(),
        uncertainty=(),
        eligible_blocks=("overview", "decisions", "actions"),
    )


def _salience(
    identifier: str,
    *,
    kind: str = "fact",
    status: str = "asserted",
    mandatory: bool = False,
    verification: str = "supported",
) -> SalienceRecord:
    return SalienceRecord(
        fact_id=identifier,
        kind=kind,
        status=status,
        verification=verification,
        instruction_relevance=1,
        meeting_importance=1,
        decision_action_weight=1 if kind in {"decision", "action"} else 0,
        recency_correction_weight=1 if kind == "correction" else 0,
        confidence=1,
        score=1,
        mandatory=mandatory,
    )


def _plan() -> CapabilityPlan:
    return CapabilityPlan(
        blocks=(
            CapabilityBlock(
                id="b001",
                capability="overview",
                purpose="Summarize the current state",
                fact_ids=("f000001", "f000002"),
                project_context_ids=("pc000001",),
                constraints=(),
            ),
            CapabilityBlock(
                id="b002",
                capability="actions",
                purpose="List assigned follow-up work",
                fact_ids=("f000003",),
                project_context_ids=(),
                constraints=(),
            ),
        )
    )


def test_outline_contract_requires_unique_monotonic_ids_and_orders() -> None:
    claim = OutlineClaim(
        id="oc000001",
        purpose="overview",
        fact_ids=("f000001",),
        project_context_ids=(),
        block_id="b001",
        order=1,
        context_only=False,
    )
    with pytest.raises(ValidationError):
        DocumentOutline(title="Notes", claims=(claim, claim))
    with pytest.raises(ValidationError):
        DocumentOutline(
            title="Notes",
            claims=(claim, claim.model_copy(update={"id": "oc000002", "order": 3})),
        )


def test_context_only_claims_are_explicit_and_cannot_pose_as_status() -> None:
    with pytest.raises(ValidationError):
        OutlineClaim(
            id="oc000001",
            purpose="decision",
            fact_ids=(),
            project_context_ids=("pc000001",),
            block_id="b001",
            order=1,
            context_only=True,
        )
    with pytest.raises(ValidationError):
        OutlineClaim(
            id="oc000001",
            purpose="context",
            fact_ids=(),
            project_context_ids=("pc000001",),
            block_id="b001",
            order=1,
            context_only=False,
        )


def test_builder_rejects_unknown_uncertain_and_unapproved_references() -> None:
    salience = (
        _salience("f000001"),
        _salience("f000002", verification="uncertain", status="uncertain"),
        _salience("f000003", kind="action", mandatory=True),
    )
    for update, code in (
        ({"fact_ids": ("f999999",)}, "unknown_fact_reference"),
        ({"fact_ids": ("f000002",)}, "uncertain_fact_reference"),
        ({"project_context_ids": ("pc999999",)}, "unapproved_context_reference"),
        ({"block_id": "b999"}, "unknown_block_reference"),
    ):
        proposed = DocumentOutline(
            title="Notes",
            claims=(
                OutlineClaim(
                    id="oc000001",
                    purpose="overview",
                    fact_ids=("f000001",),
                    project_context_ids=(),
                    block_id="b001",
                    order=1,
                    context_only=False,
                ).model_copy(update=update),
                OutlineClaim(
                    id="oc000002",
                    purpose="action",
                    fact_ids=("f000003",),
                    project_context_ids=(),
                    block_id="b002",
                    order=2,
                    context_only=False,
                ),
            ),
        )
        with pytest.raises(OutlinePolicyError, match=code):
            build_fact_covered_outline(
                plan=_plan(),
                brief=_brief(),
                salience=salience,
                approved_project_context_ids=("pc000001",),
                proposed=proposed,
            )


def test_builder_appends_only_omitted_mandatory_facts_to_compatible_blocks() -> None:
    proposed = DocumentOutline(
        title="Engineering notes",
        claims=(
            OutlineClaim(
                id="oc000001",
                purpose="overview",
                fact_ids=("f000001",),
                project_context_ids=(),
                block_id="b001",
                order=1,
                context_only=False,
            ),
        ),
    )
    result = build_fact_covered_outline(
        plan=_plan(),
        brief=_brief(),
        salience=(
            _salience("f000001"),
            _salience("f000002", kind="correction", mandatory=True),
            _salience("f000003", kind="action", mandatory=True),
        ),
        approved_project_context_ids=("pc000001",),
        proposed=proposed,
    )
    assert [claim.fact_ids for claim in result.claims] == [
        ("f000001",),
        ("f000002",),
        ("f000003",),
    ]
    assert result.claims[1].block_id == "b001"
    assert result.claims[2].block_id == "b002"
    assert [claim.order for claim in result.claims] == [1, 2, 3]


def test_builder_preserves_source_order_for_corrections_and_current_state() -> None:
    proposed = DocumentOutline(
        title="Notes",
        claims=(
            OutlineClaim(
                id="oc000001",
                purpose="correction",
                fact_ids=("f000002",),
                project_context_ids=(),
                block_id="b001",
                order=1,
                context_only=False,
            ),
            OutlineClaim(
                id="oc000002",
                purpose="overview",
                fact_ids=("f000001",),
                project_context_ids=(),
                block_id="b001",
                order=2,
                context_only=False,
            ),
            OutlineClaim(
                id="oc000003",
                purpose="action",
                fact_ids=("f000003",),
                project_context_ids=(),
                block_id="b002",
                order=3,
                context_only=False,
            ),
        ),
    )
    with pytest.raises(OutlinePolicyError, match="fact_order_violation"):
        build_fact_covered_outline(
            plan=_plan(),
            brief=_brief(),
            salience=(
                _salience("f000001"),
                _salience("f000002", kind="correction", mandatory=True),
                _salience("f000003", kind="action", mandatory=True),
            ),
            approved_project_context_ids=("pc000001",),
            proposed=proposed,
        )


def test_complete_plan_builds_deterministically_without_tools() -> None:
    outline = build_fact_covered_outline(
        plan=_plan(),
        brief=_brief("detailed"),
        salience=(
            _salience("f000001"),
            _salience("f000002", kind="correction", mandatory=True),
            _salience("f000003", kind="action", mandatory=True),
        ),
        approved_project_context_ids=("pc000001",),
    )
    assert [claim.fact_ids for claim in outline.claims if claim.fact_ids] == [
        ("f000001",),
        ("f000002",),
        ("f000003",),
    ]
    assert outline.claims[-1].block_id == "b002"


def test_outline_tool_policy_is_closed_and_bounded() -> None:
    policy = outline_tool_policy(
        run_id="r000001",
        allowed_entity_ids=("f000001", "pc000001"),
    )
    assert policy.allowed_tools == frozenset(
        {"get_fact_details", "get_project_context", "get_generation_constraints"}
    )
    assert policy.max_calls == 2
    assert policy.max_rounds == 1
    assert policy.max_result_tokens == 2048
