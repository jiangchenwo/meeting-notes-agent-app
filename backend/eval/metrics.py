"""
Evaluation metric helpers shared between the lab runner and test suite.

All functions are pure Python — no LLM calls, no DB access.
"""
import json
import re


def coverage_score(output_text: str, facts: list[str]) -> float:
    """Fraction of ground-truth facts found anywhere in output_text (case-insensitive)."""
    if not facts:
        return 1.0
    text_lower = output_text.lower()
    return round(sum(1 for f in facts if f.lower() in text_lower) / len(facts), 3)


def action_recall(
    action_items: list[dict],
    expected_owners: list[str],
    expected_tasks: list[str],
) -> float | None:
    """
    Fraction of expected owners and tasks mentioned across all action items.
    Returns None when no expected values are provided (unscored, not 0).
    """
    expected = expected_owners + expected_tasks
    if not expected:
        return None
    ai_text = json.dumps(action_items).lower()
    return round(sum(1 for e in expected if e.lower() in ai_text) / len(expected), 3)


def hallucination_check(output_text: str, transcript: str) -> list[str]:
    """
    Heuristic: capitalized proper nouns in output that don't appear in the transcript.
    Returns suspicious tokens for review — not definitive hallucination detection.
    """
    transcript_lower = transcript.lower()
    words = re.findall(r"\b[A-Z][a-z]{2,}\b", output_text)
    stopwords = {
        "Summary", "Action", "Items", "Notes", "Meeting", "Decisions", "Risks",
        "Obligations", "Highlights", "Flags", "Concepts", "Objectives", "Tickets",
        "Suggested", "Follow", "Tech", "Debt", "Architecture", "CRM", "Update",
        "Questions", "Learning", "Key", "Quiz", "Green", "Red", "Email", "Draft",
        "Healthcare", "Education", "Interview", "Project", "General",
    }
    return sorted({w for w in words if w.lower() not in transcript_lower and w not in stopwords})


def build_full_output_text(results: dict, suggestions_text: str, summary_text: str) -> str:
    """Flatten all agent outputs into a single string for coverage scoring."""
    parts = [summary_text, suggestions_text]
    for v in results.values():
        if isinstance(v, dict):
            parts.append(json.dumps(v))
    return " ".join(parts)
