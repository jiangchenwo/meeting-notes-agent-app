"""
Contract tests — mock the LLM call and verify each agent:
  1. Produces the required output keys with correct types
  2. Handles malformed / missing JSON gracefully (no exception raised)
  3. Handles missing keys in the LLM response with safe defaults
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import MOCK_CFG
from agents.base import WorkflowContext
from agents.core_agents import Summarizer, ActionItemExtractor, DecisionLogger, Critic
from agents.domain_agents import InterviewAgent, LectureAgent


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_ctx(**overrides):
    defaults = dict(
        note_id=1,
        transcript="Alice said we should use PostgreSQL. Bob agreed and will set it up by Friday.",
        domain_name="Project",
        template_name="Default",
        template_prompt="Summarize the meeting.",
        project_system_prompt="",
        project_knowledge_base="",
        cfg=MOCK_CFG.copy(),
    )
    defaults.update(overrides)
    return WorkflowContext(**defaults)


def mock_llm_response(content: str):
    """Return a (content, input_tokens, output_tokens) tuple like _call_llm."""
    return content, 100, 50


def patch_llm(content: str):
    return patch(
        "agents.base.LLMAgent._call_llm",
        return_value=mock_llm_response(content),
    )


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

class TestSummarizer:
    def test_happy_path_returns_summary_key(self):
        resp = json.dumps({"summary": "## Meeting Notes\nDecided to use PostgreSQL."})
        with patch_llm(resp):
            result = Summarizer().run(make_ctx())
        assert "summary" in result
        assert "PostgreSQL" in result["summary"]

    def test_malformed_json_uses_raw_content_as_summary(self):
        with patch_llm("PostgreSQL was chosen as the database. Action: Bob sets it up."):
            result = Summarizer().run(make_ctx())
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_json_missing_summary_key_falls_back_to_empty(self):
        resp = json.dumps({"wrong_key": "value"})
        with patch_llm(resp):
            result = Summarizer().run(make_ctx())
        assert "summary" in result

    def test_uses_project_knowledge_base_in_prompt(self):
        """KB search should augment the prompt; we verify _call_llm sees it."""
        calls = []
        def capture_call(system, user, cfg):
            calls.append((system, user))
            return json.dumps({"summary": "ok"}), 100, 50

        ctx = make_ctx(
            project_knowledge_base="Our database stack is Postgres 16 with read replicas.",
            template_prompt="database performance"
        )
        with patch("agents.base.LLMAgent._call_llm", side_effect=capture_call):
            Summarizer().run(ctx)
        # KB snippet should appear in the user message
        assert any("Postgres 16" in u for _, u in calls)


# ---------------------------------------------------------------------------
# ActionItemExtractor
# ---------------------------------------------------------------------------

class TestActionItemExtractor:
    def test_happy_path_returns_list(self):
        resp = json.dumps({"action_items": [
            {"task": "Set up Postgres", "owner": "Bob", "deadline": "Friday", "priority": "high"}
        ]})
        with patch_llm(resp):
            result = ActionItemExtractor().run(make_ctx())
        assert isinstance(result["action_items"], list)
        assert result["action_items"][0]["owner"] == "Bob"

    def test_non_list_action_items_returns_empty_list(self):
        resp = json.dumps({"action_items": "Bob sets up Postgres by Friday"})
        with patch_llm(resp):
            result = ActionItemExtractor().run(make_ctx())
        assert result["action_items"] == []

    def test_missing_action_items_key_returns_empty(self):
        resp = json.dumps({"tasks": []})
        with patch_llm(resp):
            result = ActionItemExtractor().run(make_ctx())
        assert result["action_items"] == []

    def test_items_as_strings_are_coerced_to_dicts(self):
        resp = json.dumps({"action_items": ["Set up Postgres", "Write migrations"]})
        with patch_llm(resp):
            result = ActionItemExtractor().run(make_ctx())
        for item in result["action_items"]:
            assert isinstance(item, dict)
            assert "task" in item

    def test_uses_summarizer_result_as_hint(self):
        """If Summarizer already ran, its output should appear in the user prompt."""
        calls = []
        def capture_call(system, user, cfg):
            calls.append((system, user))
            return json.dumps({"action_items": []}), 100, 50

        ctx = make_ctx()
        ctx.results["Summarizer"] = {"summary": "## Key Decision\nUse PostgreSQL for new service."}
        with patch("agents.base.LLMAgent._call_llm", side_effect=capture_call):
            ActionItemExtractor().run(ctx)
        assert any("PostgreSQL" in u for _, u in calls)

    def test_malformed_json_returns_empty_list(self):
        with patch_llm("There are no action items discussed."):
            result = ActionItemExtractor().run(make_ctx())
        assert result["action_items"] == []


# ---------------------------------------------------------------------------
# DecisionLogger
# ---------------------------------------------------------------------------

class TestDecisionLogger:
    def test_happy_path_schema(self):
        resp = json.dumps({"decisions": [
            {"decision": "Use PostgreSQL", "rationale": "Faster JSON queries", "made_by": "group"}
        ]})
        with patch_llm(resp):
            result = DecisionLogger().run(make_ctx())
        assert "decisions" in result
        assert isinstance(result["decisions"], list)
        d = result["decisions"][0]
        assert "decision" in d and "rationale" in d and "made_by" in d

    def test_empty_decisions_on_no_decisions(self):
        resp = json.dumps({"decisions": []})
        with patch_llm(resp):
            result = DecisionLogger().run(make_ctx())
        assert result["decisions"] == []

    def test_malformed_returns_fallback(self):
        with patch_llm("No decisions were made in this meeting."):
            result = DecisionLogger().run(make_ctx())
        assert "decisions" in result


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class TestCritic:
    def test_run_critique_returns_score_and_issues(self):
        resp = json.dumps({
            "dimensions": {"coverage": 3, "accuracy": 3, "specificity": 2, "structure": 1},
            "score": 9,
            "issues": [],
        })
        with patch_llm(resp):
            result = Critic().run_critique(make_ctx(), "Summarizer", "Initial summary text")
        # score recomputed from dimensions: 3+3+2+1 = 9
        assert result["score"] == 9.0
        assert isinstance(result["issues"], list)
        assert "dimensions" in result
        assert "improved_version" not in result

    def test_dimensions_recompute_score(self):
        # LLM may inflate the total; we recompute from dimensions to prevent bias
        resp = json.dumps({
            "dimensions": {"coverage": 2, "accuracy": 2, "specificity": 1, "structure": 1},
            "score": 9,  # LLM inflated — should be overridden to 6
            "issues": ["Missing budget discussion"],
        })
        with patch_llm(resp):
            result = Critic().run_critique(make_ctx(), "Summarizer", "text")
        assert result["score"] == 6.0  # 2+2+1+1

    def test_score_fallback_when_no_dimensions(self):
        resp = json.dumps({"score": "7", "issues": []})
        with patch_llm(resp):
            result = Critic().run_critique(make_ctx(), "Summarizer", "text")
        assert isinstance(result["score"], float)
        assert result["score"] == 7.0

    def test_fallback_score_on_malformed(self):
        with patch_llm("This looks good, score 8"):
            result = Critic().run_critique(make_ctx(), "Summarizer", "text")
        # Fallback score is 5 (below threshold) so retries will trigger
        assert result["score"] == 5.0

    def test_run_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Critic().run(make_ctx())


# ---------------------------------------------------------------------------
# InterviewAgent
# ---------------------------------------------------------------------------

class TestInterviewAgent:
    def test_schema_happy_path(self):
        resp = json.dumps({
            "questions_asked": [{"question": "Tell me about a system failure", "type": "behavioral"}],
            "candidate_highlights": ["Strong Redis knowledge"],
            "red_flags": ["Vague answers about past failures"],
            "green_flags": ["Detailed sliding window implementation"],
            "suggested_followups": ["Ask about largest scale system"],
        })
        with patch_llm(resp):
            result = InterviewAgent().run(make_ctx(domain_name="Interview"))
        for key in ("questions_asked", "candidate_highlights", "red_flags", "green_flags", "suggested_followups"):
            assert key in result
            assert isinstance(result[key], list)

    def test_empty_arrays_when_no_content(self):
        resp = json.dumps({
            "questions_asked": [], "candidate_highlights": [],
            "red_flags": [], "green_flags": [], "suggested_followups": [],
        })
        with patch_llm(resp):
            result = InterviewAgent().run(make_ctx(domain_name="Interview"))
        assert all(result[k] == [] for k in ("red_flags", "green_flags"))


# ---------------------------------------------------------------------------
# LectureAgent
# ---------------------------------------------------------------------------

class TestLectureAgent:
    def test_schema_happy_path(self):
        resp = json.dumps({
            "key_concepts": [{"concept": "Gradient Descent", "definition": "Optimization algorithm", "importance": "high"}],
            "learning_objectives": ["Understand backpropagation"],
            "assignments": [{"task": "Implement SGD", "due": "Thursday", "notes": "Use NumPy"}],
            "quiz_questions": [{"question": "What is learning rate?", "answer": "Step size"}],
        })
        with patch_llm(resp):
            result = LectureAgent().run(make_ctx(domain_name="Education"))
        for key in ("key_concepts", "learning_objectives", "assignments", "quiz_questions"):
            assert key in result
            assert isinstance(result[key], list)

    def test_assignment_due_dates_captured(self):
        resp = json.dumps({
            "key_concepts": [],
            "learning_objectives": [],
            "assignments": [{"task": "Problem set 3", "due": "next Thursday", "notes": ""}],
            "quiz_questions": [],
        })
        with patch_llm(resp):
            result = LectureAgent().run(make_ctx(domain_name="Education"))
        assert result["assignments"][0]["due"] == "next Thursday"
