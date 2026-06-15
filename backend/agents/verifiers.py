"""
Non-LLM verification steps — run after agent extraction.

SchemaVerifier: checks required output fields and types. Fast, deterministic.
RiskClassifier: heuristic keyword scan for content warranting human review.
Neither makes LLM calls.
"""
import re


_REQUIRED_FIELDS: dict[str, set[str]] = {
    "Summarizer":          {"summary"},
    "ActionItemExtractor": {"action_items"},
    "DecisionLogger":      {"decisions"},
    "InterviewAgent": {
        "questions_asked", "candidate_highlights",
        "red_flags", "green_flags", "suggested_followups",
    },
    "LectureAgent": {
        "key_concepts", "learning_objectives", "assignments", "quiz_questions",
    },
}

_REQUIRED_TYPES: dict[str, dict[str, type]] = {
    "Summarizer":          {"summary": str},
    "ActionItemExtractor": {"action_items": list},
    "DecisionLogger":      {"decisions": list},
    "InterviewAgent": {
        "questions_asked": list, "candidate_highlights": list,
        "red_flags": list, "green_flags": list, "suggested_followups": list,
    },
    "LectureAgent": {
        "key_concepts": list, "learning_objectives": list,
        "assignments": list, "quiz_questions": list,
    },
}

# Risk patterns keyed by domain name
_RISK_PATTERNS: dict[str, list[str]] = {
    "Healthcare": [
        r"\b(suicid\w*|self[- ]harm|crisis|overdos\w*|danger|emergency|abus\w*|"
        r"neglect|hospitali[sz]\w*|diagnos\w*|prescri\w*|medication|treatment\s+plan|"
        r"HIPAA|PHI)\b",
    ],
    "Interview": [
        r"\b(national\s+origin|protected\s+class|marital\s+status)\b",
        r"\b(personality\s+type|likabilit\w*)\b",
    ],
}


class SchemaVerifier:
    """Validate required output fields and types after each extraction step."""
    name = "SchemaVerifier"

    def verify(self, agent_name: str, result: dict) -> dict:
        required_fields = _REQUIRED_FIELDS.get(agent_name, set())
        required_types  = _REQUIRED_TYPES.get(agent_name, {})

        missing = [f for f in required_fields if f not in result]
        type_errors = [
            f"{field} must be {t.__name__}, got {type(result[field]).__name__}"
            for field, t in required_types.items()
            if field in result and not isinstance(result[field], t)
        ]
        return {
            "pass": not missing and not type_errors,
            "missing": missing,
            "type_errors": type_errors,
        }

    def verify_all(self, results: dict) -> dict[str, dict]:
        """Verify every agent result that has a known schema."""
        return {
            name: self.verify(name, result)
            for name, result in results.items()
            if name in _REQUIRED_FIELDS
        }


class RiskClassifier:
    """
    Heuristic keyword scanner for content that warrants human review.
    Returns needs_review bool + list of matched risk tokens.
    """
    name = "RiskClassifier"

    def classify(self, domain_name: str, output_text: str) -> dict:
        patterns = _RISK_PATTERNS.get(domain_name, [])
        flags: list[str] = []
        for pattern in patterns:
            flags.extend(
                m.lower() for m in re.findall(pattern, output_text, re.IGNORECASE)
            )
        unique_flags = sorted(set(flags))
        return {
            "needs_review": bool(unique_flags),
            "risk_flags": unique_flags,
            "domain": domain_name,
        }
