from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import re

from pydantic import BaseModel, ConfigDict, Field

from notes_agent_v2.domain.evidence import EvidenceChunk, canonical_digest
from notes_agent_v2.domain.transcript import Utterance
from notes_agent_v2.runtime.context import PromptTokenizer

from .extraction_contracts import ExtractionPayload, render_extraction_prompt


class ChunkPlanningError(RuntimeError):
    """Raised before model work when a complete bounded plan cannot be built."""


class EvidenceChunkPlan(BaseModel):
    """Immutable, persistence-ready provenance for a transcript chunk plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    transcript_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    utterance_ids: tuple[str, ...] = Field(min_length=1)
    tokenizer_model_key: str = Field(min_length=1)
    tokenizer_instance_id: str = Field(min_length=1)
    instruction_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_prompt_tokens: int = Field(gt=0)
    overlap_utterances: int = Field(ge=0)
    chunks: tuple[EvidenceChunk, ...] = Field(min_length=1)
    chunk_plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


_WHITESPACE = re.compile(r"\s+")


def _normalize_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("segment text must be a string")
    return _WHITESPACE.sub(" ", value).strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"segment {field} must be a string")
    normalized = _normalize_text(value)
    return normalized or None


def _optional_milliseconds(
    segment: Mapping[str, object], *, milliseconds_key: str, seconds_key: str
) -> int | None:
    if milliseconds_key in segment:
        value = segment[milliseconds_key]
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"segment {milliseconds_key} must be an integer")
        milliseconds = value
    elif seconds_key in segment:
        value = segment[seconds_key]
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"segment {seconds_key} must be numeric")
        milliseconds = round(value * 1_000)
    else:
        return None
    if milliseconds < 0:
        raise ValueError("segment timing must be nonnegative")
    return milliseconds


def normalize_transcript(
    full_text: str,
    segments: Sequence[Mapping[str, object]] | None,
) -> tuple[Utterance, ...]:
    """Normalize supplied segments or deterministically fall back to text lines."""

    normalized: list[dict[str, object]] = []
    if segments:
        previous_start: int | None = None
        previous_end: int | None = None
        for segment in segments:
            text = _normalize_text(segment.get("text"))
            if not text:
                continue
            start_ms = _optional_milliseconds(
                segment, milliseconds_key="start_ms", seconds_key="start"
            )
            end_ms = _optional_milliseconds(
                segment, milliseconds_key="end_ms", seconds_key="end"
            )
            if start_ms is not None and end_ms is not None and end_ms < start_ms:
                raise ValueError("utterance end must not precede start")
            if previous_start is not None and start_ms is not None and start_ms < previous_start:
                raise ValueError("segment start timing must be monotonic")
            if previous_end is not None and end_ms is not None and end_ms < previous_end:
                raise ValueError("segment end timing must be monotonic")
            if start_ms is not None:
                previous_start = start_ms
            if end_ms is not None:
                previous_end = end_ms
            normalized.append(
                {
                    "speaker_id": _optional_text(
                        segment.get("speaker_id"), field="speaker_id"
                    ),
                    "speaker_name": _optional_text(
                        segment.get("speaker_name", segment.get("speaker")),
                        field="speaker_name",
                    ),
                    "text": text,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }
            )
    if not normalized:
        if not isinstance(full_text, str):
            raise ValueError("full transcript text must be a string")
        normalized = [
            {
                "speaker_id": None,
                "speaker_name": None,
                "text": text,
                "start_ms": None,
                "end_ms": None,
            }
            for line in full_text.splitlines()
            if (text := _normalize_text(line))
        ]
    if not normalized:
        raise ValueError("transcript is empty")
    return tuple(
        Utterance(id=f"u{index:06d}", **item)
        for index, item in enumerate(normalized, start=1)
    )


def _render_chunk_prompt(
    utterances: Sequence[Utterance], tokenizer: PromptTokenizer, instruction: str
) -> str:
    return render_extraction_prompt(
        instruction=instruction, utterances=utterances, tokenizer=tokenizer
    )


def build_evidence_chunks(
    utterances: Sequence[Utterance],
    tokenizer: PromptTokenizer,
    max_prompt_tokens: int = 18_000,
    overlap_utterances: int = 4,
    *,
    instruction: str = "",
    is_cancelled: Callable[[], bool] = lambda: False,
) -> tuple[EvidenceChunk, ...]:
    """Greedily create exact-token chunks with deterministic utterance overlap."""

    if not utterances:
        raise ChunkPlanningError("at least one utterance is required")
    if max_prompt_tokens <= 0 or overlap_utterances < 0:
        raise ChunkPlanningError("chunk limits must be nonnegative")
    if not tokenizer.exact:
        raise ChunkPlanningError("an exact tokenizer is required")

    chunks: list[EvidenceChunk] = []
    start = 0
    while start < len(utterances):
        if is_cancelled():
            raise ChunkPlanningError("chunk planning cancelled")
        if len(chunks) >= 40:
            raise ChunkPlanningError("evidence chunk run exceeds the 40-chunk cap")

        best_end: int | None = None
        best_tokens: int | None = None
        for end in range(start + 1, len(utterances) + 1):
            rendered = _render_chunk_prompt(
                utterances[start:end], tokenizer, instruction
            )
            tokens = tokenizer.count_tokens(rendered)
            if tokens > max_prompt_tokens:
                break
            best_end = end
            best_tokens = tokens
        if best_end is None or best_tokens is None:
            raise ChunkPlanningError(
                f"utterance {utterances[start].id} cannot fit in the prompt budget"
            )

        identifiers = tuple(item.id for item in utterances[start:best_end])
        payload = {
            "utterance_ids": list(identifiers),
            "rendered_token_count": best_tokens,
        }
        chunks.append(
            EvidenceChunk(
                id=f"ec{len(chunks) + 1:06d}",
                utterance_ids=identifiers,
                rendered_token_count=best_tokens,
                digest=canonical_digest(payload),
            )
        )
        if best_end == len(utterances):
            break
        overlap = min(overlap_utterances, len(identifiers) - 1)
        next_start = best_end - overlap
        if next_start <= start:
            raise ChunkPlanningError("overlap policy prevents chunk progress")
        start = next_start
    return tuple(chunks)


def build_evidence_chunk_plan(
    utterances: Sequence[Utterance],
    tokenizer: PromptTokenizer,
    max_prompt_tokens: int = 18_000,
    overlap_utterances: int = 4,
    *,
    instruction: str = "",
    is_cancelled: Callable[[], bool] = lambda: False,
) -> EvidenceChunkPlan:
    """Return all canonical values that must be persisted before extraction."""

    chunks = build_evidence_chunks(
        utterances,
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        overlap_utterances=overlap_utterances,
        instruction=instruction,
        is_cancelled=is_cancelled,
    )
    transcript_payload = [item.model_dump(mode="json") for item in utterances]
    plan_payload = {
        "transcript_digest": canonical_digest(transcript_payload),
        "utterance_ids": [item.id for item in utterances],
        "tokenizer_model_key": tokenizer.model_key,
        "tokenizer_instance_id": tokenizer.instance_id,
        "instruction_digest": canonical_digest(instruction),
        "prompt_schema_digest": canonical_digest(
            ExtractionPayload.model_json_schema()
        ),
        "max_prompt_tokens": max_prompt_tokens,
        "overlap_utterances": overlap_utterances,
        "chunks": [item.model_dump(mode="json") for item in chunks],
    }
    return EvidenceChunkPlan(
        **plan_payload,
        chunk_plan_digest=canonical_digest(plan_payload),
    )
