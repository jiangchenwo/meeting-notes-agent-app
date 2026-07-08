"""Unit tests for pure-Python helpers: tools, transcript sizing, assembly,
and seed/workflow coverage. No LLM calls."""

import json

from agents.pipeline import _assemble_suggestions, _truncate_transcript
from agents.tools import extract_date_mentions, search_knowledge_base
from agents.workflow_spec import DOMAIN_WORKFLOWS
from seed import DOMAINS as SEED_DOMAINS, TEMPLATES as SEED_TEMPLATES

ALL_KNOWN_DOMAINS = {"General", "Education", "Healthcare", "Interview", "Project"}


# ---------------------------------------------------------------------------
# search_knowledge_base
# ---------------------------------------------------------------------------

def test_search_kb_finds_relevant_lines():
    kb = "The database is PostgreSQL.\nWe use Redis for caching.\nAuth uses JWT tokens."
    result = search_knowledge_base(kb, "database postgres")
    assert "PostgreSQL" in result


def test_search_kb_returns_prefix_on_empty_query():
    kb = "Line one.\nLine two.\nLine three."
    assert search_knowledge_base(kb, "") == kb[:600]


def test_search_kb_empty_kb_returns_empty():
    assert search_knowledge_base("", "query") == ""


def test_search_kb_respects_max_chars():
    kb = "relevant info\n" * 100
    assert len(search_knowledge_base(kb, "relevant")) <= 600


def test_search_kb_no_match_returns_empty():
    kb = "cats and dogs.\nweather is nice."
    assert search_knowledge_base(kb, "kubernetes deployment") == ""


# ---------------------------------------------------------------------------
# extract_date_mentions
# ---------------------------------------------------------------------------

def test_extract_dates_iso():
    assert "2026-06-15" in extract_date_mentions("The deadline is 2026-06-15.")


def test_extract_dates_relative():
    dates = extract_date_mentions("Deliver by end of week. Follow-up next friday.")
    assert any("end" in d or "friday" in d for d in dates)


def test_extract_dates_quarter():
    assert "q3 2026" in extract_date_mentions("Launch in Q3 2026.")


def test_extract_dates_empty():
    assert extract_date_mentions("no dates here") == []


# ---------------------------------------------------------------------------
# _truncate_transcript
# ---------------------------------------------------------------------------

def test_truncate_short_transcript_unchanged(cfg):
    assert _truncate_transcript("Short transcript.", cfg) == "Short transcript."


def test_truncate_long_transcript(cfg):
    long_text = "x" * 20_000
    result = _truncate_transcript(long_text, cfg)
    assert len(result) < len(long_text)
    assert result.endswith("[Transcript truncated due to context limit]")


def test_truncate_uses_max_tokens_from_cfg():
    text = "w " * 5000
    small = _truncate_transcript(text, {"max_tokens": 1000})
    large = _truncate_transcript(text, {"max_tokens": 8000})
    assert len(large) >= len(small)


# ---------------------------------------------------------------------------
# Seed data covers all built-in workflows
# ---------------------------------------------------------------------------

def test_all_known_domains_have_builtin_workflows_and_seed_data():
    assert set(DOMAIN_WORKFLOWS) == ALL_KNOWN_DOMAINS
    assert {name for name, _ in SEED_DOMAINS} == ALL_KNOWN_DOMAINS
    assert {domain_name for _, domain_name, _, _ in SEED_TEMPLATES} == ALL_KNOWN_DOMAINS


# ---------------------------------------------------------------------------
# _assemble_suggestions
# ---------------------------------------------------------------------------

def test_assemble_decisions_render_for_any_domain():
    # Assembly is driven by which agents ran, not by domain name, so custom
    # workflows mixing agents across domains still produce output.
    results = {
        "DecisionLogger": {
            "decisions": [{"decision": "Use PostgreSQL", "rationale": "Performance", "made_by": "group"}]
        },
    }
    out = _assemble_suggestions(results)
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
    out = _assemble_suggestions(results)
    assert "Red Flags" in out
    assert "Vague answers" in out
    assert "Strong Redis" in out


def test_assemble_empty_results_returns_empty():
    assert _assemble_suggestions({}) == ""


def test_assemble_education_includes_concepts():
    results = {
        "LectureAgent": {
            "key_concepts": [{"concept": "Gradient Descent", "definition": "Minimize loss function", "importance": "high"}],
            "learning_objectives": ["Understand backpropagation"],
            "quiz_questions": [{"question": "What is learning rate?", "answer": "Step size for gradient update"}],
            "assignments": [],
        }
    }
    out = _assemble_suggestions(results)
    assert "Gradient Descent" in out
    assert "backpropagation" in out
    assert "learning rate" in out.lower()
