from __future__ import annotations

import json
from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from notes_agent_v2.runtime.budget import RunBudget
from notes_agent_v2.runtime.gateway import GatewayRequest
from notes_agent_v2.workflow.audience import GenerationBrief
from notes_agent_v2.workflow.salience import SalienceRecord


Capability = Literal[
    "overview", "narrative", "decisions", "actions", "risks", "questions", "custom"
]


class CapabilityBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^b[0-9]{3}$")
    capability: Capability
    purpose: str = Field(min_length=1, max_length=240)
    fact_ids: tuple[str, ...]
    project_context_ids: tuple[str, ...]
    constraints: tuple[str, ...]

    @model_validator(mode="after")
    def valid_block(self) -> CapabilityBlock:
        if not self.purpose.strip():
            raise ValueError("block purpose must not be blank")
        if len(self.fact_ids) != len(set(self.fact_ids)):
            raise ValueError("block fact IDs must be unique")
        if len(self.project_context_ids) != len(set(self.project_context_ids)):
            raise ValueError("block project context IDs must be unique")
        if self.capability == "custom" and not self.constraints:
            raise ValueError("custom blocks require explicit constraints")
        return self


class CapabilityPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    blocks: tuple[CapabilityBlock, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique_block_ids(self) -> CapabilityPlan:
        identifiers = tuple(item.id for item in self.blocks)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("block IDs must be unique")
        if not any(item.capability in {"overview", "narrative"} for item in self.blocks):
            raise ValueError("an overview or narrative block is required")
        return self


class CapabilityPlanResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ready", "planning_failed"]
    plan: CapabilityPlan | None = None
    error_code: str | None = None


class PlannerGateway(Protocol):
    def call(self, request: GatewayRequest, *, budget: RunBudget, validate): ...


class PlanningPolicyError(ValueError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


def _valid_shape(content: str) -> bool:
    try:
        CapabilityPlan.model_validate_json(content)
    except Exception:
        return False
    return True


def _repair_mandatory_assignments(
    plan: CapabilityPlan, salience: Sequence[SalienceRecord]
) -> CapabilityPlan:
    blocks = list(plan.blocks)
    assigned = {fact_id for block in blocks for fact_id in block.fact_ids}
    preferred = {
        "decision": "decisions",
        "action": "actions",
        "correction": "decisions",
        "risk": "risks",
        "question": "questions",
    }
    for item in salience:
        if not item.mandatory or item.fact_id in assigned:
            continue
        wanted = preferred.get(item.kind)
        target = next(
            (index for index, block in enumerate(blocks) if block.capability == wanted),
            None,
        )
        if target is None:
            target = next(
                index
                for index, block in enumerate(blocks)
                if block.capability in {"overview", "narrative"}
            )
        block = blocks[target]
        blocks[target] = block.model_copy(
            update={"fact_ids": block.fact_ids + (item.fact_id,)}
        )
        assigned.add(item.fact_id)
    return CapabilityPlan(blocks=tuple(blocks))


def _validate_application_policy(
    plan: CapabilityPlan,
    *,
    brief: GenerationBrief,
    salience: Sequence[SalienceRecord],
    approved_project_context_ids: Sequence[str],
) -> None:
    eligible = set(brief.eligible_blocks)
    fact_by_id = {item.fact_id: item for item in salience}
    approved_context = set(approved_project_context_ids)
    for block in plan.blocks:
        if block.capability not in eligible:
            raise PlanningPolicyError("ineligible_capability")
        if (
            block.capability not in {"overview", "narrative", "custom"}
            and not block.fact_ids
        ):
            raise PlanningPolicyError("empty_optional_block")
        for identifier in block.fact_ids:
            item = fact_by_id.get(identifier)
            if item is None:
                raise PlanningPolicyError("unknown_fact_reference")
            if item.verification != "supported":
                raise PlanningPolicyError("uncertain_fact_reference")
        if not set(block.project_context_ids).issubset(approved_context):
            raise PlanningPolicyError("unapproved_context_reference")
    assigned = {fact_id for block in plan.blocks for fact_id in block.fact_ids}
    if any(item.mandatory and item.fact_id not in assigned for item in salience):
        raise PlanningPolicyError("mandatory_fact_unassigned")


def create_capability_plan(
    *,
    run_id: str,
    instruction: str,
    brief: GenerationBrief,
    salience: Sequence[SalienceRecord],
    approved_project_context_ids: Sequence[str],
    gateway: PlannerGateway,
    budget: RunBudget,
) -> CapabilityPlanResult:
    authoritative = {
        "instruction": instruction.strip() or "Summarize the meeting.",
        "brief": brief.model_dump(mode="json"),
        "ranked_fact_index": [item.model_dump(mode="json") for item in salience],
        "allowed_fact_ids": [
            item.fact_id for item in salience if item.verification == "supported"
        ],
        "allowed_project_context_ids": list(approved_project_context_ids),
    }
    analysis_request = GatewayRequest(
        run_id=run_id,
        stage="capability_analysis",
        role="planner",
        profile_name="planning_reasoned",
        messages=(
            {
                "role": "system",
                "content": (
                    "Select at most eight blocks from the closed capability schema. "
                    "fact_ids must contain only allowed_fact_ids; uncertain facts remain "
                    "visible for context but must never be referenced. project_context_ids "
                    "must contain only allowed_project_context_ids and must be empty when "
                    "that allowlist is empty. Include an overview or narrative. Do not "
                    "select models, profiles, tools, retries, budgets, workflows, or "
                    "executable operations. Return JSON only."
                ),
            },
            {"role": "user", "content": json.dumps(authoritative, sort_keys=True)},
        ),
        output_schema=CapabilityPlan.model_json_schema(),
    )
    try:
        analysis_result = gateway.call(
            analysis_request, budget=budget, validate=_valid_shape
        )
        analysis = CapabilityPlan.model_validate_json(
            analysis_result.response.final_content
        )
    except Exception:
        return CapabilityPlanResult(
            status="planning_failed", error_code="invalid_capability_analysis"
        )

    final_request = GatewayRequest(
        run_id=run_id,
        stage="capability_finalization",
        role="planner",
        profile_name="planning_structured_off",
        messages=(
            {
                "role": "system",
                "content": (
                    "Finalize the closed capability structure using only the authoritative "
                    "input and validated analysis. fact_ids must contain only "
                    "allowed_fact_ids. project_context_ids must contain only "
                    "allowed_project_context_ids. Delete every other reference rather than "
                    "copying it from the analysis. Preserve allowed IDs exactly. Return JSON "
                    "only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "authoritative_input": authoritative,
                        "analysis_summary": analysis.model_dump(mode="json"),
                    },
                    sort_keys=True,
                ),
            },
        ),
        output_schema=CapabilityPlan.model_json_schema(),
    )
    try:
        final_result = gateway.call(final_request, budget=budget, validate=_valid_shape)
        plan = CapabilityPlan.model_validate_json(final_result.response.final_content)
        plan = _repair_mandatory_assignments(plan, salience)
        _validate_application_policy(
            plan,
            brief=brief,
            salience=salience,
            approved_project_context_ids=approved_project_context_ids,
        )
    except Exception as exc:
        return CapabilityPlanResult(
            status="planning_failed",
            error_code=getattr(exc, "error_code", "invalid_capability_plan"),
        )
    return CapabilityPlanResult(status="ready", plan=plan)
