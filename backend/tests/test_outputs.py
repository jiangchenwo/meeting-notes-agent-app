"""Contract tests freezing the output-model JSON shapes.

These keys are load-bearing: the frontend and assembly logic read them out of
WorkflowStepResult.result_json / Summary.raw_sections_json. If one of these
tests breaks, the persisted JSON shape changed.
"""
import pytest
from pydantic import ValidationError

from agents.outputs import (
    AGENT_OUTPUT_TYPES,
    ActionItem,
    ActionItemsOutput,
    CritiqueDimensions,
    CritiqueOutput,
    Decision,
    DecisionsOutput,
    FALLBACK_CRITIQUE,
    InterviewOutput,
    LectureOutput,
    SummaryOutput,
)


def test_registry_covers_all_five_agents():
    assert set(AGENT_OUTPUT_TYPES) == {
        "Summarizer", "ActionItemExtractor", "DecisionLogger", "InterviewAgent", "LectureAgent",
    }


def test_summary_output_shape():
    assert SummaryOutput(summary="# Notes").model_dump() == {"summary": "# Notes"}


def test_action_items_shape_and_defaults():
    out = ActionItemsOutput(action_items=[ActionItem(task="Ship it")])
    dumped = out.model_dump()
    assert list(dumped) == ["action_items"]
    assert dumped["action_items"][0] == {
        "task": "Ship it", "owner": "TBD", "deadline": None, "priority": "medium",
    }
    assert ActionItemsOutput().model_dump() == {"action_items": []}


def test_decisions_shape_and_defaults():
    out = DecisionsOutput(decisions=[Decision(decision="Use SQLite")])
    assert out.model_dump()["decisions"][0] == {
        "decision": "Use SQLite", "rationale": "", "made_by": "group",
    }


def test_interview_output_keys():
    dumped = InterviewOutput().model_dump()
    assert set(dumped) == {
        "questions_asked", "candidate_highlights", "red_flags", "green_flags",
        "suggested_followups",
    }
    parsed = InterviewOutput.model_validate(
        {"questions_asked": [{"question": "Why us?", "type": "behavioral"}]}
    )
    assert parsed.model_dump()["questions_asked"] == [
        {"question": "Why us?", "type": "behavioral"}
    ]


def test_lecture_output_keys():
    dumped = LectureOutput.model_validate({
        "key_concepts": [{"concept": "Entropy"}],
        "assignments": [{"task": "Read ch. 3"}],
        "quiz_questions": [{"question": "Define entropy", "answer": "..."}],
    }).model_dump()
    assert set(dumped) == {"key_concepts", "learning_objectives", "assignments", "quiz_questions"}
    assert dumped["key_concepts"][0] == {"concept": "Entropy", "definition": "", "importance": "medium"}
    assert dumped["assignments"][0] == {"task": "Read ch. 3", "due": "", "notes": ""}


def test_critique_score_recomputed_from_dimensions():
    critique = CritiqueOutput(
        dimensions=CritiqueDimensions(coverage=3, accuracy=3, specificity=1, structure=1),
        issues=["Missing the budget decision"],
    )
    assert critique.score == 8.0
    dumped = critique.model_dump()
    assert dumped["score"] == 8.0  # serialized for result_json parity
    assert dumped["issues"] == ["Missing the budget decision"]


def test_critique_llm_total_is_ignored():
    # Even if the model emits its own (wrong) score, only dimensions count.
    critique = CritiqueOutput.model_validate({
        "dimensions": {"coverage": 2, "accuracy": 2, "specificity": 1, "structure": 0},
        "score": 10,
        "issues": [],
    })
    assert critique.score == 5.0


def test_critique_dimension_bounds_enforced():
    with pytest.raises(ValidationError):
        CritiqueDimensions(coverage=5, accuracy=0, specificity=0, structure=0)
    with pytest.raises(ValidationError):
        CritiqueDimensions(coverage=0, accuracy=-1, specificity=0, structure=0)


def test_fallback_critique_shape():
    assert FALLBACK_CRITIQUE == {
        "dimensions": {}, "score": 5.0, "issues": ["Could not parse critique response"],
    }
