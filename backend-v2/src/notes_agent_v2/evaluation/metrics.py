from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ReferenceItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    applicable: bool = True
    category: str = "fact"
    owner: str | None = None
    due: str | None = None
    status: str | None = None


class ReferenceScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    true_positive: int
    false_positive: int
    false_negative: int
    applicable_count: int
    precision: float | None
    recall: float | None
    f1: float | None


def score_reference_items(expected: tuple[ReferenceItem, ...], predicted_ids: tuple[str, ...]) -> ReferenceScore:
    applicable = {item.id for item in expected if item.applicable}
    predicted = set(predicted_ids)
    tp = len(applicable & predicted)
    fp = len(predicted - applicable)
    fn = len(applicable - predicted)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0 if precision is not None and recall is not None else None
    return ReferenceScore(true_positive=tp, false_positive=fp, false_negative=fn, applicable_count=len(applicable), precision=precision, recall=recall, f1=f1)
