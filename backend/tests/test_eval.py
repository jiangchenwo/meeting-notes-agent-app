"""
Tests for eval/metrics.py — pure-function scoring utilities.
No LLM calls, no DB access.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.metrics import (
    coverage_score,
    action_recall,
    hallucination_check,
    build_full_output_text,
)


# ---------------------------------------------------------------------------
# coverage_score
# ---------------------------------------------------------------------------

def test_coverage_all_facts_present():
    facts = ["PostgreSQL", "Bob owns the setup"]
    output = "The team chose PostgreSQL. Bob owns the setup task."
    assert coverage_score(output, facts) == 1.0


def test_coverage_partial():
    facts = ["PostgreSQL was chosen", "Bob owns the setup", "deadline is Friday"]
    output = "PostgreSQL was chosen."
    score = coverage_score(output, facts)
    assert 0.0 < score < 1.0
    assert round(score, 3) == round(1 / 3, 3)


def test_coverage_none_present():
    facts = ["Redis was chosen", "Alice leads the project"]
    output = "We talked about databases and ownership."
    assert coverage_score(output, facts) == 0.0


def test_coverage_empty_facts_returns_one():
    assert coverage_score("anything", []) == 1.0


def test_coverage_case_insensitive():
    facts = ["postgresql was chosen"]
    output = "PostgreSQL was chosen."
    assert coverage_score(output, facts) == 1.0


# ---------------------------------------------------------------------------
# action_recall
# ---------------------------------------------------------------------------

def test_action_recall_full_match():
    items = [{"task": "Set up Postgres", "owner": "Bob"}]
    assert action_recall(items, ["Bob"], ["Set up Postgres"]) == 1.0


def test_action_recall_partial():
    items = [{"task": "Set up Postgres", "owner": "Bob"}]
    score = action_recall(items, ["Bob", "Alice"], ["Set up Postgres"])
    assert score is not None
    assert 0.0 < score < 1.0


def test_action_recall_no_expected_returns_none():
    items = [{"task": "Do something"}]
    assert action_recall(items, [], []) is None


def test_action_recall_empty_items_returns_zero():
    score = action_recall([], ["Bob"], ["Set up Postgres"])
    assert score is not None
    assert score == 0.0


def test_action_recall_case_insensitive():
    items = [{"task": "set up postgres", "owner": "bob"}]
    assert action_recall(items, ["Bob"], ["Set up Postgres"]) == 1.0


# ---------------------------------------------------------------------------
# hallucination_check
# ---------------------------------------------------------------------------

def test_hallucination_no_false_positives_when_names_in_transcript():
    transcript = "Alice and Bob discussed PostgreSQL performance."
    output = "Alice proposed PostgreSQL. Bob agreed."
    result = hallucination_check(output, transcript)
    assert "Alice" not in result
    assert "Bob" not in result
    assert "PostgreSQL" not in result


def test_hallucination_flags_unknown_proper_noun():
    transcript = "we discussed database options"
    output = "Miroslav proposed switching to Cassandra."
    result = hallucination_check(output, transcript)
    assert "Miroslav" in result or "Cassandra" in result


def test_hallucination_stopwords_not_flagged():
    transcript = "we had a meeting"
    output = "Summary of the Meeting Notes covering Action Items and Decisions."
    result = hallucination_check(output, transcript)
    # All of these are in the stopword list
    for w in ["Summary", "Meeting", "Notes", "Action", "Items", "Decisions"]:
        assert w not in result


def test_hallucination_empty_output():
    assert hallucination_check("", "some transcript") == []


def test_hallucination_empty_transcript_flags_everything():
    output = "Miroslav and Katarzyna agreed on Redis."
    result = hallucination_check(output, "")
    assert len(result) > 0


# ---------------------------------------------------------------------------
# build_full_output_text
# ---------------------------------------------------------------------------

def test_build_full_output_text_joins_all():
    results = {
        "Summarizer": {"summary": "Postgres chosen"},
        "DecisionLogger": {"decisions": [{"decision": "use Postgres"}]},
    }
    text = build_full_output_text(results, suggestions_text="## Decisions\n- use Postgres", summary_text="Postgres chosen")
    assert "Postgres chosen" in text
    assert "use Postgres" in text


def test_build_full_output_text_empty_inputs():
    text = build_full_output_text({}, "", "")
    assert isinstance(text, str)


def test_build_full_output_text_skips_non_dict_values():
    results = {"bad": "not a dict", "ok": {"key": "value"}}
    text = build_full_output_text(results, "", "")
    assert "value" in text
    assert "bad" not in text or "not a dict" not in text
