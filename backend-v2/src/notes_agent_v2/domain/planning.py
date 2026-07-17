from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Capability = Literal["overview", "narrative", "decisions", "actions", "risks", "questions", "custom"]


class GenerationBrief(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    instruction: str = Field(min_length=1)
    audience: str = Field(min_length=1)
    desired_depth: Literal["brief", "standard", "detailed"]
    constraints: tuple[str, ...]
    requested_emphasis: tuple[str, ...]
    forbidden_content: tuple[str, ...]
    uncertainties: tuple[str, ...]


class PlannedBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^b[0-9]{6}$")
    capability: Capability
    title: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    fact_ids: tuple[str, ...]
    project_context_ids: tuple[str, ...]
    required: bool

    @model_validator(mode="after")
    def unique_references(self) -> PlannedBlock:
        if len(self.fact_ids) != len(set(self.fact_ids)) or len(self.project_context_ids) != len(set(self.project_context_ids)):
            raise ValueError("planned block references must be unique")
        return self


class CapabilityPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    capabilities: tuple[Capability, ...] = Field(min_length=1)
    blocks: tuple[PlannedBlock, ...] = Field(min_length=1, max_length=8)
    required_fact_ids: tuple[str, ...]

    @model_validator(mode="after")
    def complete_plan(self) -> CapabilityPlan:
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError("plan capabilities must be unique")
        if not ({"overview", "narrative"} & set(self.capabilities)):
            raise ValueError("plan requires overview or narrative capability")
        if len({item.id for item in self.blocks}) != len(self.blocks):
            raise ValueError("planned block IDs must be unique")
        if any(item.capability not in self.capabilities for item in self.blocks):
            raise ValueError("planned block capability must be declared")
        assigned = {identifier for item in self.blocks for identifier in item.fact_ids}
        missing = set(self.required_fact_ids) - assigned
        if missing:
            raise ValueError("required fact is not assigned to a block")
        return self
