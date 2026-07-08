"""
Evaluation metric helpers for the offline eval harness.

All functions are pure Python — no LLM calls, no DB access.
"""
import json
import re


_FACT_STOPWORDS = {
    "the", "was", "were", "are", "is", "be", "a", "an", "of", "to", "in", "on",
    "and", "for", "by", "with", "at", "from", "into", "has", "have", "had",
}


def coverage_score(output_text: str, facts: list[str]) -> float:
    """Fraction of ground-truth facts covered by output_text.

    A fact counts as covered when >= 70% of its significant tokens (length > 2,
    stopwords removed) appear in the output — full-phrase substring matching
    scores 0 for correct paraphrases ("budget of $900,000" vs the fact
    "Q3 budget was 900,000"), which made the metric useless.
    """
    if not facts:
        return 1.0
    text = output_text.lower()
    text_tokens = set(re.findall(r"[a-z0-9']+", text))
    covered = 0
    for fact in facts:
        tokens = [
            t for t in re.findall(r"[a-z0-9']+", fact.lower())
            if len(t) > 2 and t not in _FACT_STOPWORDS
        ]
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in text_tokens or t in text)
        if hits / len(tokens) >= 0.7:
            covered += 1
    return round(covered / len(facts), 3)


def action_recall(
    action_items: list[dict],
    expected_owners: list[str],
    expected_tasks: list[str],
) -> float | None:
    """
    Fraction of expected owners and tasks mentioned across all action items,
    using the same token-level matching as coverage_score (verbatim phrase
    matching scores 0 for correct rewordings).
    Returns None when no expected values are provided (unscored, not 0).
    """
    expected = expected_owners + expected_tasks
    if not expected:
        return None
    ai_text = json.dumps(action_items).lower()
    ai_tokens = set(re.findall(r"[a-z0-9']+", ai_text))
    covered = 0
    for e in expected:
        tokens = [
            t for t in re.findall(r"[a-z0-9']+", e.lower())
            if len(t) > 2 and t not in _FACT_STOPWORDS
        ]
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in ai_tokens or t in ai_text)
        if hits / len(tokens) >= 0.7:
            covered += 1
    return round(covered / len(expected), 3)


def summary_alignment(summary_text: str, reference: str) -> dict[str, float] | None:
    """ROUGE-1-style content-token overlap between the generated summary and a
    human reference summary (public datasets): recall = fraction of reference
    content tokens present in the summary, precision = the reverse. Stopwords
    and tokens of length <= 2 are ignored, matching the other metrics here.
    Returns None when either side has no content tokens (unscored, not 0).
    """
    def _content_tokens(text: str) -> set[str]:
        return {
            t for t in re.findall(r"[a-z0-9']+", text.lower())
            if len(t) > 2 and t not in _FACT_STOPWORDS
        }

    ref_tokens = _content_tokens(reference)
    out_tokens = _content_tokens(summary_text)
    if not ref_tokens or not out_tokens:
        return None
    overlap = len(ref_tokens & out_tokens)
    return {
        "reference_recall": round(overlap / len(ref_tokens), 3),
        "reference_precision": round(overlap / len(out_tokens), 3),
    }


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
