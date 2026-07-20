from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from notes_agent_v2.runtime.tools import ToolPolicy
from notes_agent_v2.workflow.audience import GenerationBrief
from notes_agent_v2.workflow.planner import CapabilityBlock, CapabilityPlan
from notes_agent_v2.workflow.salience import SalienceRecord


ClaimPurpose = Literal[
    "overview",
    "narrative",
    "decision",
    "action",
    "risk",
    "question",
    "correction",
    "custom",
    "context",
]


class OutlinePolicyError(ValueError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class OutlineClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^oc[0-9]{6}$")
    purpose: ClaimPurpose
    fact_ids: tuple[str, ...]
    project_context_ids: tuple[str, ...]
    block_id: str = Field(pattern=r"^b[0-9]{3}$")
    order: int = Field(gt=0)
    context_only: bool

    @model_validator(mode="after")
    def valid_support(self) -> OutlineClaim:
        if len(self.fact_ids) != len(set(self.fact_ids)):
            raise ValueError("outline fact IDs must be unique")
        if len(self.project_context_ids) != len(set(self.project_context_ids)):
            raise ValueError("outline context IDs must be unique")
        if not self.fact_ids and not self.project_context_ids:
            raise ValueError("outline claim requires support")
        if self.context_only:
            if self.fact_ids or not self.project_context_ids or self.purpose != "context":
                raise ValueError("context-only outline claims must be explicitly labeled")
        elif self.purpose == "context":
            raise ValueError("context purpose requires context-only labeling")
        return self


class DocumentOutline(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(min_length=1, max_length=240)
    claims: tuple[OutlineClaim, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_unique_claims(self) -> DocumentOutline:
        identifiers = tuple(item.id for item in self.claims)
        orders = tuple(item.order for item in self.claims)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("outline claim IDs must be unique")
        if orders != tuple(range(1, len(self.claims) + 1)):
            raise ValueError("outline claim order must be contiguous")
        return self


def outline_tool_policy(
    *, run_id: str, allowed_entity_ids: Sequence[str]
) -> ToolPolicy:
    return ToolPolicy(
        run_id=run_id,
        stage="outline",
        allowed_tools=frozenset(
            {
                "get_fact_details",
                "get_project_context",
                "get_generation_constraints",
            }
        ),
        allowed_entity_ids=frozenset(allowed_entity_ids),
        max_rounds=1,
        max_calls=2,
        max_result_tokens=2048,
    )


def build_fact_covered_outline(
    *,
    plan: CapabilityPlan,
    brief: GenerationBrief,
    salience: Sequence[SalienceRecord],
    approved_project_context_ids: Sequence[str],
    proposed: DocumentOutline | None = None,
) -> DocumentOutline:
    fact_by_id = {item.fact_id: item for item in salience}
    block_by_id = {item.id: item for item in plan.blocks}
    approved_context = set(approved_project_context_ids)
    outline = proposed or _outline_from_plan(plan, brief, fact_by_id)
    _validate_references(
        outline,
        block_by_id=block_by_id,
        fact_by_id=fact_by_id,
        approved_context=approved_context,
    )
    outline = _append_mandatory(outline, plan, salience)
    _validate_references(
        outline,
        block_by_id=block_by_id,
        fact_by_id=fact_by_id,
        approved_context=approved_context,
    )
    _validate_fact_order(outline, salience)
    return outline


def _outline_from_plan(
    plan: CapabilityPlan,
    brief: GenerationBrief,
    fact_by_id: dict[str, SalienceRecord],
) -> DocumentOutline:
    claims: list[OutlineClaim] = []
    for block in plan.blocks:
        for fact_id in block.fact_ids:
            fact = fact_by_id.get(fact_id)
            if fact is None:
                raise OutlinePolicyError("unknown_fact_reference")
            claims.append(
                OutlineClaim(
                    id=f"oc{len(claims) + 1:06d}",
                    purpose=_purpose(block, fact),
                    fact_ids=(fact_id,),
                    project_context_ids=(),
                    block_id=block.id,
                    order=len(claims) + 1,
                    context_only=False,
                )
            )
        for context_id in block.project_context_ids:
            claims.append(
                OutlineClaim(
                    id=f"oc{len(claims) + 1:06d}",
                    purpose="context",
                    fact_ids=(),
                    project_context_ids=(context_id,),
                    block_id=block.id,
                    order=len(claims) + 1,
                    context_only=True,
                )
            )
    if not claims:
        raise OutlinePolicyError("empty_outline")
    audience = brief.audience.strip().capitalize()
    return DocumentOutline(title=f"{audience} meeting notes", claims=tuple(claims))


def _purpose(block: CapabilityBlock, fact: SalienceRecord) -> ClaimPurpose:
    if fact.kind in {"decision", "action", "risk", "question", "correction"}:
        return fact.kind  # type: ignore[return-value]
    if block.capability in {"overview", "narrative", "custom"}:
        return block.capability
    return "narrative"


def _validate_references(
    outline: DocumentOutline,
    *,
    block_by_id: dict[str, CapabilityBlock],
    fact_by_id: dict[str, SalienceRecord],
    approved_context: set[str],
) -> None:
    for claim in outline.claims:
        block = block_by_id.get(claim.block_id)
        if block is None:
            raise OutlinePolicyError("unknown_block_reference")
        for identifier in claim.fact_ids:
            fact = fact_by_id.get(identifier)
            if fact is None:
                raise OutlinePolicyError("unknown_fact_reference")
            if fact.verification != "supported":
                raise OutlinePolicyError("uncertain_fact_reference")
            if identifier not in block.fact_ids:
                raise OutlinePolicyError("unassigned_fact_reference")
        for identifier in claim.project_context_ids:
            if identifier not in approved_context:
                raise OutlinePolicyError("unapproved_context_reference")
            if identifier not in block.project_context_ids:
                raise OutlinePolicyError("unassigned_context_reference")


def _append_mandatory(
    outline: DocumentOutline,
    plan: CapabilityPlan,
    salience: Sequence[SalienceRecord],
) -> DocumentOutline:
    claims = list(outline.claims)
    assigned = {identifier for claim in claims for identifier in claim.fact_ids}
    preferred = {
        "decision": "decisions",
        "action": "actions",
        "correction": "overview",
        "risk": "risks",
        "question": "questions",
    }
    for fact in salience:
        if not fact.mandatory or fact.fact_id in assigned:
            continue
        capability = preferred.get(fact.kind)
        block = next(
            (
                item
                for item in plan.blocks
                if fact.fact_id in item.fact_ids and item.capability == capability
            ),
            None,
        ) or next(
            (item for item in plan.blocks if fact.fact_id in item.fact_ids),
            None,
        )
        if block is None:
            raise OutlinePolicyError("mandatory_fact_unassigned")
        claims.append(
            OutlineClaim(
                id=f"oc{len(claims) + 1:06d}",
                purpose=_purpose(block, fact),
                fact_ids=(fact.fact_id,),
                project_context_ids=(),
                block_id=block.id,
                order=len(claims) + 1,
                context_only=False,
            )
        )
        assigned.add(fact.fact_id)
    return DocumentOutline(title=outline.title, claims=tuple(claims))


def _validate_fact_order(
    outline: DocumentOutline, salience: Sequence[SalienceRecord]
) -> None:
    source_order = {item.fact_id: index for index, item in enumerate(salience)}
    observed = [
        source_order[identifier]
        for claim in outline.claims
        for identifier in claim.fact_ids
    ]
    if observed != sorted(observed):
        raise OutlinePolicyError("fact_order_violation")
