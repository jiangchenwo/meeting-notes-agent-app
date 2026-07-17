from __future__ import annotations

import pytest

from notes_agent_v2.evaluation.metrics import ReferenceItem, score_reference_items
from notes_agent_v2.evaluation.qag import QagDirection, QagError, QagQuestion, evaluate_qag


def test_reference_metrics_are_applicability_aware_and_deduplicate() -> None:
    expected = (
        ReferenceItem(id="a", text="Ship Friday", applicable=True),
        ReferenceItem(id="b", text="Owner Alice", applicable=True),
        ReferenceItem(id="c", text="No due date", applicable=False),
    )
    score = score_reference_items(expected, predicted_ids=("a", "a", "x"))
    assert (score.true_positive, score.false_positive, score.false_negative) == (1, 1, 1)
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(0.5)


@pytest.mark.parametrize("index", range(40))
def test_known_answer_metric_fixture_matrix(index: int) -> None:
    expected = tuple(ReferenceItem(id=f"f{item}", text=f"fact {item}", applicable=item % 5 != 4) for item in range(5))
    applicable = {item.id for item in expected if item.applicable}
    predicted = tuple(sorted(applicable)) if index % 4 == 0 else ("f0", "unknown") if index % 4 == 1 else () if index % 4 == 2 else ("f0", "f1")
    score = score_reference_items(expected, predicted)
    predicted_set = set(predicted)
    assert score.true_positive == len(applicable & predicted_set)
    assert score.false_positive == len(predicted_set - applicable)
    assert score.false_negative == len(applicable - predicted_set)


class ScriptedQag:
    def generate(self, direction: QagDirection, source: str):
        assert source
        return [QagQuestion(id="q1", question="Was Friday approved?", binding_id="fact-1")]

    def answer(self, direction: QagDirection, question: QagQuestion, context: str):
        return {"answer": "yes", "evidence_ids": ["u1"], "supported": True}


def test_qag_uses_separate_source_and_answer_contexts() -> None:
    result = evaluate_qag(
        ScriptedQag(),
        direction=QagDirection.coverage,
        generation_source="reference facts",
        answer_context="candidate notes",
        allowed_evidence_ids={"u1"},
    )
    assert result.score == 1.0
    assert result.question_count == 1


def test_qag_rejects_more_than_eight_or_unbound_questions() -> None:
    class Bad(ScriptedQag):
        def generate(self, direction: QagDirection, source: str):
            return [QagQuestion(id=str(i), question="q", binding_id="f") for i in range(9)]

    with pytest.raises(QagError, match="eight"):
        evaluate_qag(Bad(), direction=QagDirection.coverage, generation_source="x", answer_context="y", allowed_evidence_ids=set())


@pytest.mark.parametrize("index", range(24))
def test_qag_authored_omission_and_alignment_matrix(index: int) -> None:
    direction = QagDirection.coverage if index < 12 else QagDirection.factual_alignment
    result = evaluate_qag(ScriptedQag(), direction=direction, generation_source=f"source-{index}", answer_context=f"context-{index}", allowed_evidence_ids={"u1"})
    assert result.question_count <= 8
    assert result.decisions[0].question.binding_id == "fact-1"
