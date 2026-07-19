from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import re
import string
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from notes_agent_v2.domain.evidence import ExtractedFactCandidate, Fact, validate_fact_graph
from notes_agent_v2.domain.transcript import Utterance


class SemanticGrouper(Protocol):
    def propose(
        self, candidates: Sequence[ExtractedFactCandidate]
    ) -> Sequence[Sequence[str]]: ...


class ConsolidationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    facts: tuple[Fact, ...]
    candidate_to_fact: dict[str, str]

    @property
    def facts_by_id(self) -> dict[str, Fact]:
        return {item.id: item for item in self.facts}


_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "in", "is", "it",
    "of", "on", "the", "to", "we", "will", "was", "were",
}
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
}


def _ordered_tokens(text: str) -> tuple[str, ...]:
    cleaned = text.casefold().translate(str.maketrans("", "", string.punctuation))
    return tuple(
        _NUMBER_WORDS.get(token, token)
        for token in cleaned.split()
        if token not in _STOPWORDS
    )


def _tokens(text: str) -> set[str]:
    return set(_ordered_tokens(text))


def _normalized(text: str) -> str:
    return " ".join(_ordered_tokens(text))


def _numbers(text: str) -> set[str]:
    tokens = _tokens(text)
    return {item for item in tokens if re.fullmatch(r"\d+(?:\.\d+)?", item)}


def _similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _first_position(
    candidates: Sequence[ExtractedFactCandidate], order: Mapping[str, int]
) -> int:
    return min(
        order[identifier]
        for candidate in candidates
        for span in candidate.evidence
        for identifier in span.utterance_ids
    )


def _semantic_merge_allowed(
    left: ExtractedFactCandidate, right: ExtractedFactCandidate
) -> bool:
    if left.kind != right.kind or left.status != right.status:
        return False
    if left.owner and right.owner and left.owner.casefold() != right.owner.casefold():
        return False
    if left.due_text and right.due_text and left.due_text.casefold() != right.due_text.casefold():
        return False
    left_numbers = _numbers(left.text)
    right_numbers = _numbers(right.text)
    if left_numbers and right_numbers and left_numbers != right_numbers:
        return False
    left_tokens = _tokens(left.text)
    right_tokens = _tokens(right.text)
    if "risk" in left_tokens and "risk" in right_tokens:
        left_subject = left_tokens - {"risk", "latency"}
        right_subject = right_tokens - {"risk", "latency"}
        if left_subject and right_subject and left_subject.isdisjoint(right_subject):
            return False
    return _similarity(left.text, right.text) >= 0.35


def consolidate_candidates(
    candidates: Sequence[ExtractedFactCandidate],
    utterances: Sequence[Utterance],
    *,
    semantic_grouper: SemanticGrouper | None = None,
    verification: Mapping[
        str, Literal["supported", "uncertain", "contradicted"]
    ] | None = None,
) -> ConsolidationResult:
    """Create loss-aware fact memory from supported and uncertain candidates."""

    verification = verification or {}
    included = [
        item
        for item in candidates
        if verification.get(item.id, "supported") != "contradicted"
    ]
    by_id = {item.id: item for item in included}
    if len(by_id) != len(included):
        raise ValueError("candidate IDs must be unique")
    order = {item.id: index for index, item in enumerate(utterances)}

    parent = {item.id: item.id for item in included}

    def find(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    exact_groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    evidence_groups: dict[tuple[tuple[tuple[str, ...], str], ...], list[str]] = defaultdict(list)
    for item in included:
        exact_groups[(_normalized(item.text), item.kind, item.status)].append(item.id)
        evidence_key = tuple(
            (span.utterance_ids, span.quote) for span in item.evidence
        )
        evidence_groups[evidence_key].append(item.id)
    for identifiers in exact_groups.values():
        for identifier in identifiers[1:]:
            union(identifiers[0], identifier)
    for identifiers in evidence_groups.values():
        for left_index, left in enumerate(identifiers):
            for right in identifiers[left_index + 1 :]:
                left_item = by_id[left]
                right_item = by_id[right]
                if (
                    left_item.kind == right_item.kind
                    and left_item.status == right_item.status
                    and _similarity(left_item.text, right_item.text) >= 0.8
                ):
                    union(left, right)

    if semantic_grouper is not None:
        for proposed in semantic_grouper.propose(tuple(included)):
            identifiers = [item for item in proposed if item in by_id]
            if len(identifiers) < 2:
                continue
            anchor = by_id[identifiers[0]]
            if all(
                _semantic_merge_allowed(anchor, by_id[identifier])
                for identifier in identifiers[1:]
            ):
                for identifier in identifiers[1:]:
                    union(identifiers[0], identifier)

    grouped: dict[str, list[ExtractedFactCandidate]] = defaultdict(list)
    for item in included:
        grouped[find(item.id)].append(item)
    groups = sorted(
        grouped.values(), key=lambda items: _first_position(items, order)
    )

    fact_inputs: list[dict[str, object]] = []
    candidate_to_fact: dict[str, str] = {}
    for index, items in enumerate(groups, start=1):
        items.sort(key=lambda item: (_first_position((item,), order), item.id))
        identifier = f"f{index:06d}"
        for item in items:
            candidate_to_fact[item.id] = identifier
        evidence = []
        evidence_keys: set[str] = set()
        for item in items:
            for span in item.evidence:
                key = span.model_dump_json()
                if key not in evidence_keys:
                    evidence_keys.add(key)
                    evidence.append(span)
        uncertain = any(
            verification.get(item.id, "supported") == "uncertain" for item in items
        )
        richest = max(items, key=lambda item: len(item.text))
        fact_inputs.append(
            {
                "id": identifier,
                "text": richest.text,
                "kind": items[0].kind,
                "status": "uncertain" if uncertain else items[0].status,
                "speaker_ids": tuple(
                    dict.fromkeys(
                        speaker for item in items for speaker in item.speaker_ids
                    )
                ),
                "owner": next((item.owner for item in reversed(items) if item.owner), None),
                "due_text": next(
                    (item.due_text for item in reversed(items) if item.due_text), None
                ),
                "confidence": 0.5 if uncertain else 1.0,
                "verification": "uncertain" if uncertain else "supported",
                "evidence": tuple(evidence),
                "source_candidate_ids": tuple(sorted(item.id for item in items)),
                "supersedes_fact_ids": (),
                "conflicts_with_fact_ids": (),
            }
        )

    supersedes: dict[str, set[str]] = defaultdict(set)
    conflicts: dict[str, set[str]] = defaultdict(set)
    for later_index, later in enumerate(fact_inputs):
        for earlier in fact_inputs[:later_index]:
            similarity = _similarity(str(later["text"]), str(earlier["text"]))
            if similarity < 0.30:
                continue
            later_id = str(later["id"])
            earlier_id = str(earlier["id"])
            later_status = str(later["status"])
            earlier_status = str(earlier["status"])
            if later["kind"] == "correction" or (
                earlier_status in {"proposed", "asserted"}
                and later_status in {"approved", "rejected", "completed"}
            ):
                supersedes[later_id].add(earlier_id)
                continue
            later_numbers = _numbers(str(later["text"]))
            earlier_numbers = _numbers(str(earlier["text"]))
            later_speakers = set(later["speaker_ids"])
            earlier_speakers = set(earlier["speaker_ids"])
            if (
                later_numbers
                and earlier_numbers
                and later_numbers != earlier_numbers
                and later_speakers.isdisjoint(earlier_speakers)
            ):
                conflicts[later_id].add(earlier_id)
                conflicts[earlier_id].add(later_id)

    facts = tuple(
        Fact(
            **{
                **item,
                "supersedes_fact_ids": tuple(sorted(supersedes[str(item["id"])])),
                "conflicts_with_fact_ids": tuple(sorted(conflicts[str(item["id"])])),
            }
        )
        for item in fact_inputs
    )
    validate_fact_graph(facts, tuple(utterances))
    return ConsolidationResult(facts=facts, candidate_to_fact=candidate_to_fact)
