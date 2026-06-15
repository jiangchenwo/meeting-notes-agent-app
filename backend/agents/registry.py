from dataclasses import dataclass

from .base import LLMAgent
from .core_agents import ActionItemExtractor, DecisionLogger, Summarizer
from .domain_agents import InterviewAgent, LectureAgent


@dataclass(frozen=True)
class AgentSpec:
    name: str
    purpose: str
    domains: frozenset[str]
    reads: frozenset[str]
    writes: frozenset[str]
    output_schema: type
    can_run_parallel: bool = False
    critique_required: bool = False
    human_review_required: bool = False
    model_profile: str = "default"


_AGENT_CLASSES: dict[str, type[LLMAgent]] = {
    "Summarizer": Summarizer,
    "ActionItemExtractor": ActionItemExtractor,
    "DecisionLogger": DecisionLogger,
    "InterviewAgent": InterviewAgent,
    "LectureAgent": LectureAgent,
}


AGENT_SPECS: dict[str, AgentSpec] = {
    "Summarizer": AgentSpec(
        name="Summarizer",
        purpose="Write the narrative meeting summary.",
        domains=frozenset({"General", "Education", "Healthcare", "Interview", "Project"}),
        reads=frozenset({"transcript", "template_prompt", "project_context"}),
        writes=frozenset({"summary"}),
        output_schema=dict,
        can_run_parallel=True,
        critique_required=True,
    ),
    "ActionItemExtractor": AgentSpec(
        name="ActionItemExtractor",
        purpose="Extract concrete committed tasks.",
        domains=frozenset({"General", "Education", "Healthcare", "Project"}),
        reads=frozenset({"transcript", "summary"}),
        writes=frozenset({"action_items"}),
        output_schema=dict,
        can_run_parallel=True,
        critique_required=True,
    ),
    "DecisionLogger": AgentSpec(
        name="DecisionLogger",
        purpose="Extract explicit decisions and rationale.",
        domains=frozenset({"General", "Project"}),
        reads=frozenset({"transcript"}),
        writes=frozenset({"decisions"}),
        output_schema=dict,
        can_run_parallel=True,
    ),
    "InterviewAgent": AgentSpec(
        name="InterviewAgent",
        purpose="Extract interview questions, highlights, concerns, and follow-ups.",
        domains=frozenset({"Interview"}),
        reads=frozenset({"transcript"}),
        writes=frozenset({"questions_asked", "candidate_highlights", "red_flags", "green_flags", "suggested_followups"}),
        output_schema=dict,
        can_run_parallel=True,
        critique_required=True,
        human_review_required=True,
    ),
    "LectureAgent": AgentSpec(
        name="LectureAgent",
        purpose="Extract lecture concepts, learning objectives, assignments, and quiz questions.",
        domains=frozenset({"Education"}),
        reads=frozenset({"transcript"}),
        writes=frozenset({"key_concepts", "learning_objectives", "assignments", "quiz_questions"}),
        output_schema=dict,
        can_run_parallel=True,
    ),
}


def create_agent(name: str) -> LLMAgent:
    return _AGENT_CLASSES[name]()


def create_agent_registry() -> dict[str, LLMAgent]:
    return {name: create_agent(name) for name in AGENT_SPECS}
