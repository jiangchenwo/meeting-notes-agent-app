from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
import json
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from notes_agent_v2.domain.evidence import (
    EvidenceChunk,
    EvidenceSpan,
    ExtractedFactCandidate,
    canonical_digest,
)
from notes_agent_v2.domain.transcript import Utterance
from notes_agent_v2.runtime.budget import RunBudget
from notes_agent_v2.runtime.gateway import GatewayRequest

from .extraction_contracts import (
    CandidatePayload,
    ExtractionPayload,
    build_extraction_messages,
)


class ExtractionGateway(Protocol):
    def call(
        self,
        request: GatewayRequest,
        *,
        budget: RunBudget,
        validate: Callable[[str], bool],
    ) -> object: ...


class ChunkExtractionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    status: Literal["completed", "failed"]
    candidate_ids: tuple[str, ...]
    error_code: str | None
    artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExtractionRunResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunks: tuple[ChunkExtractionResult, ...]
    candidates: tuple[ExtractedFactCandidate, ...]
    complete: bool


class _CandidateValidationError(ValueError):
    pass


def _parse_payload(value: str) -> ExtractionPayload:
    return ExtractionPayload.model_validate_json(value)


def _valid_payload(value: str) -> bool:
    try:
        _parse_payload(value)
    except (ValidationError, ValueError):
        return False
    return True


def _candidate_identifier(
    *, chunk_digest: str, local_index: int, payload: CandidatePayload
) -> str:
    fact_digest = canonical_digest(payload.model_dump(mode="json"))
    digest = hashlib.sha256(
        f"{chunk_digest}:{local_index}:{fact_digest}".encode()
    ).hexdigest()
    return f"fc{int(digest[:12], 16) % 1_000_000:06d}"


def _canonicalize_candidate(
    payload: CandidatePayload,
    *,
    chunk: EvidenceChunk,
    utterance_by_id: dict[str, Utterance],
) -> CandidatePayload:
    source_ids = tuple(
        identifier
        for span in payload.evidence
        for identifier in span.utterance_ids
    )
    if any(identifier not in utterance_by_id for identifier in source_ids):
        raise _CandidateValidationError("unknown_utterance")
    if any(identifier not in chunk.utterance_ids for identifier in source_ids):
        raise _CandidateValidationError("utterance_outside_chunk")

    evidence = tuple(
        EvidenceSpan(
            utterance_ids=span.utterance_ids,
            quote="\n".join(
                utterance_by_id[identifier].text
                for identifier in span.utterance_ids
            ),
        )
        for span in payload.evidence
    )
    source_utterances = [utterance_by_id[identifier] for identifier in source_ids]
    source_text = "\n".join(item.quote for item in evidence).casefold()
    speaker_aliases = {
        alias.casefold(): item.speaker_id
        for item in source_utterances
        for alias in (item.speaker_id, item.speaker_name)
        if alias is not None and item.speaker_id is not None
    }
    speaker_ids = tuple(
        dict.fromkeys(
            speaker_aliases.get(speaker.casefold(), speaker)
            for speaker in payload.speaker_ids
        )
    )
    owner = payload.owner if payload.kind == "action" else None
    due_text = payload.due_text if payload.kind == "action" else None
    if owner is not None and owner.casefold() not in source_text:
        owner = None
    if due_text is not None and due_text.casefold() not in source_text:
        due_text = None
    return payload.model_copy(
        update={
            "speaker_ids": speaker_ids,
            "owner": owner,
            "due_text": due_text,
            "evidence": evidence,
        }
    )


def _validate_candidate(
    payload: CandidatePayload,
    *,
    chunk: EvidenceChunk,
    utterance_by_id: dict[str, Utterance],
) -> None:
    if not payload.text.strip():
        raise _CandidateValidationError("candidate_text_blank")
    source_ids: list[str] = []
    source_texts: list[str] = []
    for span in payload.evidence:
        if any(identifier not in utterance_by_id for identifier in span.utterance_ids):
            raise _CandidateValidationError("unknown_utterance")
        if any(identifier not in chunk.utterance_ids for identifier in span.utterance_ids):
            raise _CandidateValidationError("utterance_outside_chunk")
        source = "\n".join(
            utterance_by_id[identifier].text for identifier in span.utterance_ids
        )
        if span.quote not in source:
            raise _CandidateValidationError("quote_not_exact")
        source_ids.extend(span.utterance_ids)
        source_texts.append(source)

    source_utterances = [utterance_by_id[identifier] for identifier in source_ids]
    stated_speakers = {item.speaker_id for item in source_utterances if item.speaker_id}
    if any(speaker not in stated_speakers for speaker in payload.speaker_ids):
        raise _CandidateValidationError("speaker_not_in_evidence")
    combined_source = "\n".join(source_texts).casefold()
    if payload.owner is not None and payload.owner.casefold() not in combined_source:
        raise _CandidateValidationError("owner_not_stated")
    if payload.due_text is not None and payload.due_text.casefold() not in combined_source:
        raise _CandidateValidationError("due_not_stated")
    if (payload.owner is not None or payload.due_text is not None) and payload.kind != "action":
        raise _CandidateValidationError("owner_due_only_allowed_on_action")
    proposal_markers = ("propose", "proposal", "suggest", "could", "might", "option")
    approval_markers = ("approved", "agreed", "decided", "accepted")
    if (
        payload.status == "approved"
        and any(marker in combined_source for marker in proposal_markers)
        and not any(marker in combined_source for marker in approval_markers)
    ):
        raise _CandidateValidationError("proposal_cannot_be_approved")


def _safe_chunk_result(
    *,
    chunk_id: str,
    status: Literal["completed", "failed"],
    candidate_ids: tuple[str, ...] = (),
    error_code: str | None = None,
) -> ChunkExtractionResult:
    payload = {
        "chunk_id": chunk_id,
        "status": status,
        "candidate_ids": list(candidate_ids),
        "error_code": error_code,
    }
    return ChunkExtractionResult(**payload, artifact_digest=canonical_digest(payload))


def _extract_content(result: object) -> str:
    response = getattr(result, "response", None)
    content = getattr(response, "final_content", None)
    if not isinstance(content, str):
        raise ValueError("gateway result has no final content")
    return content


def extract_cited_facts(
    *,
    run_id: str,
    instruction: str,
    chunks: Sequence[EvidenceChunk],
    utterances: Sequence[Utterance],
    gateway: ExtractionGateway,
    budget: RunBudget,
    profile_name: str = "structured_off",
    persist_artifact: Callable[[ChunkExtractionResult], None] = lambda _item: None,
) -> ExtractionRunResult:
    """Extract every chunk independently and fail the run closed on any loss."""

    utterance_by_id = {item.id: item for item in utterances}
    outcomes: list[ChunkExtractionResult] = []
    candidates: list[ExtractedFactCandidate] = []
    used_candidate_ids: set[str] = set()
    for chunk in chunks:
        chunk_utterances = [utterance_by_id[item] for item in chunk.utterance_ids]
        request = GatewayRequest(
            run_id=run_id,
            stage="extract",
            role="atomic_fact_extractor",
            profile_name=profile_name,
            messages=build_extraction_messages(
                instruction=instruction, utterances=chunk_utterances
            ),
            output_schema=ExtractionPayload.model_json_schema(),
        )
        try:
            gateway_result = gateway.call(request, budget=budget, validate=_valid_payload)
            parsed = _parse_payload(_extract_content(gateway_result))
            chunk_candidates: list[ExtractedFactCandidate] = []
            for local_index, payload in enumerate(parsed.candidates):
                payload = _canonicalize_candidate(
                    payload, chunk=chunk, utterance_by_id=utterance_by_id
                )
                _validate_candidate(
                    payload, chunk=chunk, utterance_by_id=utterance_by_id
                )
                identifier = _candidate_identifier(
                    chunk_digest=chunk.digest,
                    local_index=local_index,
                    payload=payload,
                )
                while identifier in used_candidate_ids:
                    number = (int(identifier[2:]) + 1) % 1_000_000
                    identifier = f"fc{number:06d}"
                used_candidate_ids.add(identifier)
                chunk_candidates.append(
                    ExtractedFactCandidate(id=identifier, **payload.model_dump())
                )
            candidates.extend(chunk_candidates)
            outcome = _safe_chunk_result(
                chunk_id=chunk.id,
                status="completed",
                candidate_ids=tuple(item.id for item in chunk_candidates),
            )
        except _CandidateValidationError as exc:
            outcome = _safe_chunk_result(
                chunk_id=chunk.id, status="failed", error_code=str(exc)
            )
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            outcome = _safe_chunk_result(
                chunk_id=chunk.id,
                status="failed",
                error_code=f"parser_failure:{type(exc).__name__}",
            )
        except Exception as exc:
            outcome = _safe_chunk_result(
                chunk_id=chunk.id,
                status="failed",
                error_code=f"gateway_failure:{type(exc).__name__}",
            )
        outcomes.append(outcome)
        persist_artifact(outcome)

    complete = len(outcomes) == len(chunks) and all(
        item.status == "completed" for item in outcomes
    )
    return ExtractionRunResult(
        chunks=tuple(outcomes),
        candidates=tuple(candidates),
        complete=complete,
    )
