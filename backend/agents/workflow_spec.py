"""Workflow definitions as validated data.

DOMAIN_WORKFLOWS are the built-in per-domain plans; a Template.workflow_config
JSON blob overrides them. The write path (templates API) validates strictly
with WorkflowSpec; the read path (select_workflow) stays lenient — anything
invalid falls back to the domain default, as the legacy engine did.
"""
import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

AgentName = Literal[
    "Summarizer", "ActionItemExtractor", "DecisionLogger", "InterviewAgent", "LectureAgent"
]


class StepSpec(BaseModel):
    agent: AgentName
    prompt_override: str | None = Field(
        default=None, description="Replaces the template prompt for this step only"
    )


class WorkflowSpec(BaseModel):
    steps: list[StepSpec] = Field(min_length=1, max_length=8)
    critique_steps: list[AgentName] = []
    critique_threshold: float = Field(default=8.0, ge=0, le=10)
    max_retries: int = Field(default=2, ge=0, le=3)

    @field_validator("steps", mode="before")
    @classmethod
    def _coerce_legacy_steps(cls, v):
        # Legacy shape: steps: ["Summarizer", ...]
        if isinstance(v, list):
            return [{"agent": s} if isinstance(s, str) else s for s in v]
        return v

    @model_validator(mode="after")
    def _critique_steps_must_run(self):
        run_agents = {s.agent for s in self.steps}
        unknown = [c for c in self.critique_steps if c not in run_agents]
        if unknown:
            raise ValueError(f"critique_steps must also appear in steps: {unknown}")
        return self

    @property
    def step_names(self) -> list[str]:
        return [s.agent for s in self.steps]


def _spec(steps: list[str], critique_steps: list[str]) -> WorkflowSpec:
    return WorkflowSpec.model_validate({"steps": steps, "critique_steps": critique_steps})


DOMAIN_WORKFLOWS: dict[str, WorkflowSpec] = {
    # Priority 0: baseline for any conversation that doesn't fit a specific
    # domain. Decisions are worth extracting from any meeting, not just
    # Project ones.
    "General": _spec(
        ["Summarizer", "ActionItemExtractor", "DecisionLogger"],
        ["Summarizer"],
    ),
    # The lecture extraction is this domain's signature output — critique it
    # alongside the summary.
    "Education": _spec(
        ["Summarizer", "LectureAgent", "ActionItemExtractor"],
        ["Summarizer", "LectureAgent"],
    ),
    # Accuracy-critical domain: follow-ups and instructions in the action
    # items get a quality pass too. (No clinical extraction agent on purpose —
    # structured medical output from a small local model is what the
    # RiskClassifier's needs_review flag guards against, not something to
    # generate more of.)
    "Healthcare": _spec(
        ["Summarizer", "ActionItemExtractor"],
        ["Summarizer", "ActionItemExtractor"],
    ),
    # Both the narrative summary and the candidate assessment are user-facing.
    "Interview": _spec(
        ["Summarizer", "InterviewAgent"],
        ["Summarizer", "InterviewAgent"],
    ),
    "Project": _spec(
        ["Summarizer", "ActionItemExtractor", "DecisionLogger"],
        ["Summarizer"],
    ),
}

# Fallback for domains not in DOMAIN_WORKFLOWS.
DEFAULT_WORKFLOW = _spec(
    ["Summarizer", "ActionItemExtractor", "DecisionLogger"],
    ["Summarizer"],
)


def select_workflow(domain_name: str, template_workflow_config: str | None) -> WorkflowSpec:
    if template_workflow_config:
        try:
            override = json.loads(template_workflow_config)
            if isinstance(override, dict) and "steps" in override:
                # Legacy merge semantics: override fields win over the default plan.
                merged = {**DEFAULT_WORKFLOW.model_dump(), **override}
                return WorkflowSpec.model_validate(merged)
        except Exception:
            pass
    return DOMAIN_WORKFLOWS.get(domain_name, DEFAULT_WORKFLOW)
