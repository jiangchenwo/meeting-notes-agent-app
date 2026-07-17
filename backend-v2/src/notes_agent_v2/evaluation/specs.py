from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MetricGate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    metric: str = Field(min_length=1)
    operation: Literal["eq", "gte", "lte"]
    threshold: float


class FeatureEvaluationSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    feature_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]+$")
    hypothesis: str = Field(min_length=10)
    baseline: str = Field(min_length=1)
    treatment: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    metrics: tuple[MetricGate, ...] = Field(min_length=1)
    trace_requirements: tuple[str, ...] = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=1)
    max_requests: int = Field(ge=0)
    max_cost_usd: float = Field(ge=0)
    invalidation_conditions: tuple[str, ...] = Field(min_length=1)
    report_owner: str = Field(min_length=1)


def load_feature_specs(path: Path) -> dict[str, FeatureEvaluationSpec]:
    payload = json.loads(path.read_text())
    specs = [FeatureEvaluationSpec.model_validate(item) for item in payload.get("features", [])]
    result = {spec.feature_id: spec for spec in specs}
    if len(result) != len(specs):
        raise ValueError("feature IDs must be unique")
    return result
