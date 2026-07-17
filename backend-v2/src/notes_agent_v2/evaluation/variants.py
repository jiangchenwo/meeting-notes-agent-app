from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class EvaluationVariant(StrEnum):
    full_v2 = "full_v2"
    single_pass_evidence = "single_pass_evidence"
    without_verification = "without_verification"
    without_consolidation = "without_consolidation"
    fixed_plan = "fixed_plan"
    without_critics = "without_critics"
    without_revision = "without_revision"


class WorkflowPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    extraction_passes: int = 2
    verification: bool = True
    consolidation: bool = True
    dynamic_plan: bool = True
    critics: bool = True
    revisions: bool = True


def workflow_policy_for(variant: EvaluationVariant) -> WorkflowPolicy:
    updates = {
        EvaluationVariant.full_v2: {},
        EvaluationVariant.single_pass_evidence: {"extraction_passes": 1},
        EvaluationVariant.without_verification: {"verification": False},
        EvaluationVariant.without_consolidation: {"consolidation": False},
        EvaluationVariant.fixed_plan: {"dynamic_plan": False},
        EvaluationVariant.without_critics: {"critics": False},
        EvaluationVariant.without_revision: {"revisions": False},
    }
    return WorkflowPolicy.model_validate(updates[variant])
