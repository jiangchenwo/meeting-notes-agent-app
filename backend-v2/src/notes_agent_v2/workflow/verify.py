from __future__ import annotations

from collections.abc import Callable, Sequence
import json
import re
import string
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from notes_agent_v2.domain.evidence import ExtractedFactCandidate
from notes_agent_v2.domain.transcript import Utterance
from notes_agent_v2.runtime.budget import RunBudget
from notes_agent_v2.runtime.gateway import GatewayRequest


class VerificationGateway(Protocol):
    def call(
        self,
        request: GatewayRequest,
        *,
        budget: RunBudget,
        validate: Callable[[str], bool],
    ) -> object: ...


class _SemanticPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["supported", "contradicted", "uncertain"]
    evidence_ids: tuple[str, ...]


class VerificationDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    status: Literal["supported", "contradicted", "uncertain"]
    evidence_ids: tuple[str, ...]
    deterministic_findings: tuple[str, ...]
    semantic_finding: Literal["supported", "contradicted", "uncertain"] | None
    error_code: str | None = None


_NUMBER = re.compile(r"(?<![\w@])\d+(?:\.\d+)?%?(?![\w@])")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s;,]+", re.IGNORECASE)
_DATE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}\b",
    re.IGNORECASE,
)


def _entities(pattern: re.Pattern[str], text: str) -> set[str]:
    return {item.casefold().rstrip(string.punctuation) for item in pattern.findall(text)}


def _normalized_statement(text: str) -> str:
    return " ".join(
        text.casefold().translate(str.maketrans("", "", string.punctuation)).split()
    )


def _source_for_candidate(
    candidate: ExtractedFactCandidate,
    utterance_by_id: dict[str, Utterance],
) -> tuple[str, tuple[str, ...], list[Utterance]]:
    ids: list[str] = []
    sources: list[str] = []
    for span in candidate.evidence:
        if any(identifier not in utterance_by_id for identifier in span.utterance_ids):
            return "", (), []
        source = "\n".join(
            utterance_by_id[identifier].text for identifier in span.utterance_ids
        )
        if span.quote not in source:
            return "", (), []
        sources.append(source)
        ids.extend(span.utterance_ids)
    ordered_ids = tuple(dict.fromkeys(ids))
    return (
        "\n".join(sources),
        ordered_ids,
        [utterance_by_id[identifier] for identifier in ordered_ids],
    )


def _deterministic_findings(
    candidate: ExtractedFactCandidate,
    utterance_by_id: dict[str, Utterance],
) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    source, evidence_ids, source_utterances = _source_for_candidate(
        candidate, utterance_by_id
    )
    if not source:
        return ("quote_or_scope_mismatch",), source, evidence_ids
    findings: list[str] = []
    stated_speakers = {item.speaker_id for item in source_utterances if item.speaker_id}
    if any(item not in stated_speakers for item in candidate.speaker_ids):
        findings.append("speaker_attribution_mismatch")

    candidate_dates = _entities(_DATE, candidate.text)
    source_dates = _entities(_DATE, source)
    if not candidate_dates.issubset(source_dates):
        findings.append("date_mismatch")
    candidate_numbers = _entities(_NUMBER, candidate.text)
    source_numbers = _entities(_NUMBER, source)
    if not candidate_numbers.issubset(source_numbers):
        findings.append("number_mismatch")
    if not _entities(_EMAIL, candidate.text).issubset(_entities(_EMAIL, source)):
        findings.append("email_mismatch")
    if not _entities(_URL, candidate.text).issubset(_entities(_URL, source)):
        findings.append("url_mismatch")

    normalized_source = _normalized_statement(source)
    normalized_candidate = _normalized_statement(candidate.text)
    negative_source = any(
        marker in f" {normalized_source} " for marker in (" not ", " no ", " never ")
    )
    negative_candidate = any(
        marker in f" {normalized_candidate} "
        for marker in (" not ", " no ", " never ")
    )
    if negative_source != negative_candidate and candidate.status in {
        "approved",
        "completed",
        "asserted",
    }:
        findings.append("status_or_negation_mismatch")
    status_markers = {
        "approved": ("approved", "agreed", "decided", "accepted"),
        "rejected": ("rejected", "declined"),
        "completed": ("completed", "finished", "done"),
        "proposed": ("propose", "proposal", "suggest"),
    }
    required_markers = status_markers.get(candidate.status)
    if required_markers and not any(
        marker in normalized_source for marker in required_markers
    ):
        findings.append("status_or_negation_mismatch")

    if candidate.kind == "correction":
        correction_marked = any(
            marker in normalized_source for marker in ("correction", "actually", "not")
        )
        not_match = re.search(r"\b([^,.;]+),?\s+not\s+([^,.;]+)", source, re.IGNORECASE)
        if not correction_marked or (
            not_match is not None
            and _normalized_statement(not_match.group(2)) in normalized_candidate
            and _normalized_statement(not_match.group(1)) not in normalized_candidate
        ):
            findings.append("correction_mismatch")

    if candidate.owner is not None and candidate.owner.casefold() not in source.casefold():
        findings.append("owner_not_cooccurring")
    if candidate.due_text is not None:
        due = candidate.due_text.casefold()
        due_utterances = [item for item in source_utterances if due in item.text.casefold()]
        if not due_utterances:
            findings.append("due_date_mismatch")
        elif all("meeting date" in item.text.casefold() for item in due_utterances):
            findings.append("meeting_date_as_due")
        elif candidate.owner is not None and not any(
            candidate.owner.casefold() in item.text.casefold() and due in item.text.casefold()
            for item in source_utterances
        ):
            findings.append("owner_due_not_cooccurring")
    return tuple(dict.fromkeys(findings)), source, evidence_ids


def _window(
    evidence_ids: tuple[str, ...], utterances: Sequence[Utterance]
) -> tuple[Utterance, ...]:
    order = {item.id: index for index, item in enumerate(utterances)}
    positions = [order[item] for item in evidence_ids if item in order]
    if not positions:
        return ()
    start = max(0, min(positions) - 3)
    end = min(len(utterances), start + 8)
    start = max(0, end - 8)
    return tuple(utterances[start:end])


def _parse_semantic(value: str) -> _SemanticPayload:
    return _SemanticPayload.model_validate_json(value)


def _semantic_valid(value: str) -> bool:
    try:
        _parse_semantic(value)
    except (ValidationError, ValueError):
        return False
    return True


def _result_content(value: object) -> str:
    content = getattr(getattr(value, "response", None), "final_content", None)
    if not isinstance(content, str):
        raise ValueError("gateway result has no final content")
    return content


def verify_candidates(
    *,
    run_id: str,
    candidates: Sequence[ExtractedFactCandidate],
    utterances: Sequence[Utterance],
    gateway: VerificationGateway | None,
    budget: RunBudget,
) -> tuple[VerificationDecision, ...]:
    """Verify deterministic invariants, then resolve only semantic ambiguity."""

    utterance_by_id = {item.id: item for item in utterances}
    decisions: list[VerificationDecision] = []
    for candidate in candidates:
        findings, source, evidence_ids = _deterministic_findings(
            candidate, utterance_by_id
        )
        if findings:
            decisions.append(
                VerificationDecision(
                    candidate_id=candidate.id,
                    status="contradicted",
                    evidence_ids=evidence_ids,
                    deterministic_findings=findings,
                    semantic_finding=None,
                )
            )
            continue
        if _normalized_statement(candidate.text) in _normalized_statement(source):
            decisions.append(
                VerificationDecision(
                    candidate_id=candidate.id,
                    status="supported",
                    evidence_ids=evidence_ids,
                    deterministic_findings=("deterministic_supported",),
                    semantic_finding=None,
                )
            )
            continue

        ambiguous = ("deterministic_ambiguous",)
        if gateway is None:
            decisions.append(
                VerificationDecision(
                    candidate_id=candidate.id,
                    status="uncertain",
                    evidence_ids=evidence_ids,
                    deterministic_findings=ambiguous,
                    semantic_finding=None,
                    error_code="semantic_verification_unavailable",
                )
            )
            continue
        source_window = _window(evidence_ids, utterances)
        allowed_ids = {item.id for item in source_window}
        request = GatewayRequest(
            run_id=run_id,
            stage="verify",
            role="bounded_fact_verifier",
            profile_name="structured_off",
            messages=(
                {
                    "role": "system",
                    "content": "Classify support using only the bounded transcript window.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "candidate": candidate.model_dump(mode="json"),
                            "source_window": [
                                item.model_dump(mode="json") for item in source_window
                            ],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ),
            output_schema=_SemanticPayload.model_json_schema(),
        )
        try:
            result = gateway.call(request, budget=budget, validate=_semantic_valid)
            semantic = _parse_semantic(_result_content(result))
            if any(item not in allowed_ids for item in semantic.evidence_ids):
                raise PermissionError("semantic_scope_violation")
            status: Literal["supported", "uncertain"] = (
                "supported" if semantic.status == "supported" else "uncertain"
            )
            decisions.append(
                VerificationDecision(
                    candidate_id=candidate.id,
                    status=status,
                    evidence_ids=semantic.evidence_ids or evidence_ids,
                    deterministic_findings=ambiguous,
                    semantic_finding=semantic.status,
                )
            )
        except PermissionError:
            decisions.append(
                VerificationDecision(
                    candidate_id=candidate.id,
                    status="uncertain",
                    evidence_ids=evidence_ids,
                    deterministic_findings=ambiguous,
                    semantic_finding=None,
                    error_code="semantic_scope_violation",
                )
            )
        except Exception as exc:
            decisions.append(
                VerificationDecision(
                    candidate_id=candidate.id,
                    status="uncertain",
                    evidence_ids=evidence_ids,
                    deterministic_findings=ambiguous,
                    semantic_finding=None,
                    error_code=f"semantic_verification_failure:{type(exc).__name__}",
                )
            )
    return tuple(decisions)


def verified_candidates(
    candidates: Sequence[ExtractedFactCandidate],
    decisions: Sequence[VerificationDecision],
) -> tuple[ExtractedFactCandidate, ...]:
    """Return only explicitly supported candidates in original source order."""

    status_by_id = {item.candidate_id: item.status for item in decisions}
    return tuple(item for item in candidates if status_by_id.get(item.id) == "supported")
