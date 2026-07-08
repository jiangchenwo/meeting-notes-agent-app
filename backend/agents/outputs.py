"""Typed output models for every agent step.

Field names are load-bearing: `model_dump()` keys must match the JSON shapes
stored in WorkflowStepResult.result_json / Summary.raw_sections_json, which the
frontend and the assembly logic read.
"""
from pydantic import BaseModel, Field, computed_field


class SummaryOutput(BaseModel):
    summary: str = Field(description="The full meeting summary in Markdown")


class ActionItem(BaseModel):
    task: str = Field(description="One specific task, starting with a verb")
    owner: str = Field("TBD", description='Person explicitly assigned, or "TBD"')
    deadline: str | None = Field(None, description="Date or phrase if mentioned, null otherwise")
    priority: str = Field("medium", description='"high" if urgent/blocking, "medium" otherwise')


class ActionItemsOutput(BaseModel):
    action_items: list[ActionItem] = []


class Decision(BaseModel):
    decision: str
    rationale: str = Field("", description="Rationale if stated; empty string if not mentioned")
    made_by: str = Field("group", description='Name(s) or "group" for a collective decision')


class DecisionsOutput(BaseModel):
    decisions: list[Decision] = []


class InterviewQuestion(BaseModel):
    question: str
    type: str = Field("other", description="behavioral | technical | situational | other")


class InterviewOutput(BaseModel):
    questions_asked: list[InterviewQuestion] = []
    candidate_highlights: list[str] = []
    red_flags: list[str] = Field(
        [], description="Concrete concerns (vague answers, gaps, contradictions)"
    )
    green_flags: list[str] = Field(
        [], description="Concrete positives (specific examples, depth, clarity)"
    )
    suggested_followups: list[str] = Field(
        [], description="Questions that would help further assess the candidate"
    )


class KeyConcept(BaseModel):
    concept: str
    definition: str = ""
    importance: str = Field("medium", description="high | medium | low")


class Assignment(BaseModel):
    task: str
    due: str = ""
    notes: str = ""


class QuizQuestion(BaseModel):
    question: str
    answer: str = ""


class LectureOutput(BaseModel):
    key_concepts: list[KeyConcept] = []
    learning_objectives: list[str] = []
    assignments: list[Assignment] = []
    quiz_questions: list[QuizQuestion] = []


class CritiqueDimensions(BaseModel):
    coverage: int = Field(ge=0, le=4)
    accuracy: int = Field(ge=0, le=3)
    specificity: int = Field(ge=0, le=2)
    structure: int = Field(ge=0, le=1)


class CritiqueOutput(BaseModel):
    dimensions: CritiqueDimensions
    issues: list[str] = Field(
        [],
        description="For each deduction: the specific missing topic or wrong claim, "
        "and exactly what to add or correct",
    )

    @computed_field
    @property
    def score(self) -> float:
        """Sum of the four dimensions — the LLM's own total is never trusted."""
        d = self.dimensions
        return float(d.coverage + d.accuracy + d.specificity + d.structure)


# Legacy semantics: an unparseable critique scores 5 and stays advisory.
FALLBACK_CRITIQUE = {
    "dimensions": {},
    "score": 5.0,
    "issues": ["Could not parse critique response"],
}

AGENT_OUTPUT_TYPES: dict[str, type[BaseModel]] = {
    "Summarizer": SummaryOutput,
    "ActionItemExtractor": ActionItemsOutput,
    "DecisionLogger": DecisionsOutput,
    "InterviewAgent": InterviewOutput,
    "LectureAgent": LectureOutput,
}
