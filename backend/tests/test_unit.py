"""
Unit tests — no LLM calls required. Tests tools, base class helpers,
and orchestrator pure-Python functions.
"""

import json
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.tools import search_knowledge_base, extract_date_mentions
from agents.base import LLMAgent, WorkflowContext
from agents.orchestrator import _select_workflow, _assemble_suggestions, DOMAIN_WORKFLOWS
from agents.registry import AGENT_SPECS, AgentSpec, create_agent_registry
from agents.workflows import (
    _DEFAULT_WORKFLOW as WORKFLOW_DEFAULT,
    DOMAIN_WORKFLOWS as WORKFLOW_DEFINITIONS,
    select_workflow,
)
from seed import DOMAINS as SEED_DOMAINS, TEMPLATES as SEED_TEMPLATES


PRIORITY_DOMAINS = {"Education", "Healthcare", "Interview", "Project"}
ALL_KNOWN_DOMAINS = PRIORITY_DOMAINS | {"General"}


# ---------------------------------------------------------------------------
# WorkflowContext helper
# ---------------------------------------------------------------------------

def make_ctx(domain="Project", transcript="Meeting transcript here.", cfg=None):
    return WorkflowContext(
        note_id=1,
        transcript=transcript,
        domain_name=domain,
        template_name="Default",
        template_prompt="Summarize the meeting.",
        project_system_prompt="",
        project_knowledge_base="",
        cfg=cfg or {"base_url": "http://localhost:1234/v1", "model": "m", "max_tokens": 4096, "max_response_tokens": 1024},
    )


# ---------------------------------------------------------------------------
# search_knowledge_base
# ---------------------------------------------------------------------------

def test_search_kb_finds_relevant_lines():
    kb = "The database is PostgreSQL.\nWe use Redis for caching.\nAuth uses JWT tokens."
    result = search_knowledge_base(kb, "database postgres")
    assert "PostgreSQL" in result


def test_search_kb_returns_prefix_on_empty_query():
    kb = "Line one.\nLine two.\nLine three."
    result = search_knowledge_base(kb, "")
    assert result == kb[:600]


def test_search_kb_empty_kb_returns_empty():
    assert search_knowledge_base("", "query") == ""


def test_search_kb_respects_max_chars():
    kb = "relevant info\n" * 100
    result = search_knowledge_base(kb, "relevant")
    assert len(result) <= 600


def test_search_kb_no_match_returns_empty():
    kb = "cats and dogs.\nweather is nice."
    result = search_knowledge_base(kb, "kubernetes deployment")
    assert result == ""


# ---------------------------------------------------------------------------
# extract_date_mentions
# ---------------------------------------------------------------------------

def test_extract_dates_iso():
    text = "The deadline is 2026-06-15."
    dates = extract_date_mentions(text)
    assert "2026-06-15" in dates


def test_extract_dates_relative():
    text = "Deliver by end of week. Follow-up next friday."
    dates = extract_date_mentions(text)
    assert any("end" in d or "friday" in d for d in dates)


def test_extract_dates_quarter():
    text = "Launch in Q3 2026."
    dates = extract_date_mentions(text)
    assert "q3 2026" in dates


def test_extract_dates_empty():
    assert extract_date_mentions("no dates here") == []


# ---------------------------------------------------------------------------
# LLMAgent._parse_json
# ---------------------------------------------------------------------------

class _ConcreteAgent(LLMAgent):
    name = "Test"
    def run(self, ctx):
        return {}


_agent = _ConcreteAgent()


def test_parse_json_valid():
    result = _agent._parse_json('{"key": "value"}', {"key": "fallback"})
    assert result == {"key": "value"}


def test_parse_json_code_fenced():
    content = '```json\n{"summary": "Meeting notes"}\n```'
    result = _agent._parse_json(content, {"summary": ""})
    assert result["summary"] == "Meeting notes"


def test_parse_json_embedded_in_text():
    content = 'Here is the result:\n{"action_items": [{"task": "Do something"}]}'
    result = _agent._parse_json(content, {"action_items": []})
    assert len(result["action_items"]) == 1


def test_parse_json_fallback_on_garbage():
    result = _agent._parse_json("This is not JSON at all!", {"default": True})
    assert result == {"default": True}


def test_parse_json_nested_objects():
    content = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
    result = _agent._parse_json(content, {})
    assert result["outer"]["inner"] == "value"
    assert result["list"] == [1, 2, 3]


def test_parse_json_returns_fallback_on_truncated():
    content = '{"key": "value'  # truncated — missing closing quote and brace
    result = _agent._parse_json(content, {"key": "default"})
    assert result == {"key": "default"}


# ---------------------------------------------------------------------------
# LLMAgent._truncate_transcript
# ---------------------------------------------------------------------------

def test_truncate_short_transcript_unchanged(cfg):
    agent = _ConcreteAgent()
    short = "Short transcript."
    assert agent._truncate_transcript(short, cfg) == short


def test_truncate_long_transcript(cfg):
    agent = _ConcreteAgent()
    long_text = "x" * 20_000
    result = agent._truncate_transcript(long_text, cfg)
    assert len(result) < len(long_text)
    assert result.endswith("[Transcript truncated due to context limit]")


def test_truncate_uses_max_tokens_from_cfg():
    agent = _ConcreteAgent()
    small_cfg = {"max_tokens": 1000, "max_response_tokens": 512}
    large_cfg = {"max_tokens": 8000, "max_response_tokens": 512}
    text = "w " * 5000
    small_result = agent._truncate_transcript(text, small_cfg)
    large_result = agent._truncate_transcript(text, large_cfg)
    # Larger context window should allow more of the transcript
    assert len(large_result) >= len(small_result)


# ---------------------------------------------------------------------------
# _select_workflow
# ---------------------------------------------------------------------------

def test_select_workflow_known_domain():
    wf = _select_workflow("Project", None)
    assert "DecisionLogger" in wf["steps"]


def test_select_workflow_unknown_domain_falls_back():
    wf = _select_workflow("Cooking", None)
    assert "Summarizer" in wf["steps"]
    assert "ActionItemExtractor" in wf["steps"]


def test_select_workflow_template_override_replaces_steps():
    override = json.dumps({
        "steps": ["Summarizer", "ActionItemExtractor", "LectureAgent"],
        "critique_threshold": 9,
    })
    wf = _select_workflow("Project", override)
    assert "LectureAgent" in wf["steps"]
    assert "DecisionLogger" not in wf["steps"]
    assert wf["critique_threshold"] == 9


def test_select_workflow_template_override_invalid_json_uses_domain():
    wf = _select_workflow("Project", "not valid json {{")
    assert "DecisionLogger" in wf["steps"]


def test_select_workflow_template_override_no_steps_key_uses_domain():
    # Override without 'steps' key should fall back to domain workflow
    override = json.dumps({"critique_threshold": 9})
    wf = _select_workflow("Project", override)
    assert "DecisionLogger" in wf["steps"]


def test_all_domains_have_required_keys():
    required = {"steps", "critique_steps", "critique_threshold", "max_retries"}
    for domain, wf in DOMAIN_WORKFLOWS.items():
        missing = required - set(wf.keys())
        assert not missing, f"{domain} workflow missing keys: {missing}"


def test_all_known_domains_have_builtin_workflows_and_seed_data():
    # General is Priority 0 (baseline) + 4 specialized domains
    assert set(DOMAIN_WORKFLOWS) == ALL_KNOWN_DOMAINS
    assert {name for name, _ in SEED_DOMAINS} == ALL_KNOWN_DOMAINS
    assert {domain_name for _, domain_name, _, _ in SEED_TEMPLATES} == ALL_KNOWN_DOMAINS


# ---------------------------------------------------------------------------
# AgentSpec registry
# ---------------------------------------------------------------------------

def test_agent_specs_cover_configured_workflow_steps():
    workflow_agents = set(WORKFLOW_DEFAULT["steps"])
    for wf in WORKFLOW_DEFINITIONS.values():
        workflow_agents.update(wf["steps"])

    missing = workflow_agents - set(AGENT_SPECS)
    assert not missing, f"Workflow agents missing specs: {sorted(missing)}"
    for agent_name in workflow_agents:
        assert isinstance(AGENT_SPECS[agent_name], AgentSpec)


def test_agent_specs_name_their_outputs():
    assert AGENT_SPECS["Summarizer"].writes == {"summary"}
    assert AGENT_SPECS["ActionItemExtractor"].writes == {"action_items"}
    assert AGENT_SPECS["DecisionLogger"].writes == {"decisions"}
    assert AGENT_SPECS["InterviewAgent"].human_review_required is True


def test_agent_specs_only_advertise_known_domains():
    for spec in AGENT_SPECS.values():
        assert spec.domains <= ALL_KNOWN_DOMAINS


def test_create_agent_registry_matches_configured_specs():
    registry = create_agent_registry()
    assert set(registry) == set(AGENT_SPECS)
    assert registry["Summarizer"].name == "Summarizer"
    assert registry["ActionItemExtractor"].name == "ActionItemExtractor"


# ---------------------------------------------------------------------------
# agents.workflows compatibility
# ---------------------------------------------------------------------------

def test_workflows_module_preserves_domain_selection():
    assert select_workflow("Project", None) == _select_workflow("Project", None)


def test_workflows_module_preserves_template_override_behavior():
    override = json.dumps({
        "steps": ["Summarizer", "LectureAgent"],
        "critique_threshold": 9,
    })
    assert select_workflow("Project", override) == _select_workflow("Project", override)


# ---------------------------------------------------------------------------
# _assemble_suggestions
# ---------------------------------------------------------------------------

def test_assemble_project_includes_decisions():
    results = {
        "DecisionLogger": {
            "decisions": [{"decision": "Use PostgreSQL", "rationale": "Performance", "made_by": "group"}]
        },
    }
    out = _assemble_suggestions(results, "Project")
    assert "PostgreSQL" in out
    assert "Performance" in out


def test_assemble_interview_sections():
    results = {
        "InterviewAgent": {
            "red_flags": ["Vague answers about past failures"],
            "green_flags": ["Detailed system design example"],
            "candidate_highlights": ["Strong Redis knowledge"],
            "suggested_followups": ["Ask about team conflict"],
        }
    }
    out = _assemble_suggestions(results, "Interview")
    assert "Red Flags" in out
    assert "Vague answers" in out
    assert "Strong Redis" in out


def test_assemble_empty_results_returns_empty():
    out = _assemble_suggestions({}, "Project")
    assert out == ""


def test_assemble_education_includes_concepts():
    results = {
        "LectureAgent": {
            "key_concepts": [{"concept": "Gradient Descent", "definition": "Minimize loss function", "importance": "high"}],
            "learning_objectives": ["Understand backpropagation"],
            "quiz_questions": [{"question": "What is learning rate?", "answer": "Step size for gradient update"}],
            "assignments": [],
        }
    }
    out = _assemble_suggestions(results, "Education")
    assert "Gradient Descent" in out
    assert "backpropagation" in out
    assert "learning rate" in out.lower()
