import re


def search_knowledge_base(kb_text: str, query: str, max_chars: int = 600) -> str:
    """Keyword search over project knowledge base text. Returns top matching lines."""
    if not kb_text:
        return ""
    if not query:
        return kb_text[:max_chars]
    query_terms = [t for t in query.lower().split() if len(t) > 2]
    if not query_terms:
        return kb_text[:max_chars]
    lines = kb_text.split("\n")
    scored = []
    for line in lines:
        line_lower = line.lower()
        score = sum(1 for term in query_terms if term in line_lower)
        if score > 0:
            scored.append((score, line))
    scored.sort(reverse=True)
    result = "\n".join(line for _, line in scored[:10])
    return result[:max_chars]


def extract_date_mentions(text: str) -> list[str]:
    """Extract date and time references from text."""
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b",
        r"\b(?:next|this)\s+(?:week|month|quarter|sprint|friday|monday|tuesday|wednesday|thursday)\b",
        r"\bby\s+(?:eod|eow|end\s+of\s+(?:day|week|month|quarter))\b",
        r"\bin\s+\d+\s+(?:days?|weeks?|months?)\b",
        r"\b(?:q[1-4])\s+\d{4}\b",
    ]
    found = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text.lower()):
            found.add(match.group(0))
    return sorted(found)
