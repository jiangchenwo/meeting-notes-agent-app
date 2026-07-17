from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class QagError(RuntimeError):
    pass


class QagDirection(StrEnum):
    coverage = "coverage"
    factual_alignment = "factual_alignment"


class QagQuestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    binding_id: str = Field(min_length=1)


class QagDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    question: QagQuestion
    answer: str = Field(min_length=1)
    evidence_ids: tuple[str, ...]
    supported: bool


class QagResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    direction: QagDirection
    question_count: int
    supported_count: int
    score: float
    decisions: tuple[QagDecision, ...]


class QagProvider(Protocol):
    def generate(self, direction: QagDirection, source: str) -> list[QagQuestion]: ...
    def answer(self, direction: QagDirection, question: QagQuestion, context: str) -> dict[str, Any]: ...


def evaluate_qag(provider: QagProvider, *, direction: QagDirection, generation_source: str, answer_context: str, allowed_evidence_ids: set[str]) -> QagResult:
    if generation_source == answer_context:
        raise QagError("generation and answer contexts must be independent")
    questions = provider.generate(direction, generation_source)
    if len(questions) > 8:
        raise QagError("QAG permits at most eight questions per direction")
    if not questions:
        raise QagError("QAG generation returned no questions")
    decisions = []
    for question in questions:
        try:
            payload = provider.answer(direction, question, answer_context)
            decision = QagDecision(question=question, answer=payload["answer"], evidence_ids=tuple(payload["evidence_ids"]), supported=payload["supported"])
        except Exception as exc:
            raise QagError("QAG answer was malformed") from exc
        if set(decision.evidence_ids) - allowed_evidence_ids:
            raise QagError("QAG answer contains unresolved evidence")
        decisions.append(decision)
    supported = sum(item.supported for item in decisions)
    return QagResult(direction=direction, question_count=len(decisions), supported_count=supported, score=supported / len(decisions), decisions=tuple(decisions))
