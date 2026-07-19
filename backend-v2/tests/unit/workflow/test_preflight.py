from __future__ import annotations

import re

import pytest

from notes_agent_v2.workflow.preflight import (
    ChunkPlanningError,
    build_evidence_chunk_plan,
    build_evidence_chunks,
    normalize_transcript,
)


class ExactTokenizer:
    model_key = "test/model"
    instance_id = "instance-1"
    exact = True

    def render_chat(self, messages, tools=None, output_schema=None):
        del tools, output_schema
        return "\n".join(str(message["content"]) for message in messages)

    def count_tokens(self, rendered_prompt: str) -> int:
        # One invariant prompt token plus one token per rendered utterance.
        return 1 + len(
            re.findall(r'<utterance id=|"id":\s*"u', rendered_prompt)
        )


class CapturingTokenizer(ExactTokenizer):
    def __init__(self) -> None:
        self.output_schemas = []

    def render_chat(self, messages, tools=None, output_schema=None):
        del tools
        self.output_schemas.append(output_schema)
        return super().render_chat(messages, output_schema=output_schema)


def _segments(count: int) -> list[dict[str, object]]:
    return [
        {
            "text": f"  line   {index}  ",
            "speaker_id": f"s{index % 2}",
            "speaker_name": f"Speaker {index % 2}",
            "start_ms": index * 1_000,
            "end_ms": index * 1_000 + 900,
        }
        for index in range(count)
    ]


def test_normalize_segments_preserves_source_metadata_and_is_deterministic() -> None:
    first = normalize_transcript("ignored when segments are valid", _segments(3))
    second = normalize_transcript("different fallback", _segments(3))

    assert first == second
    assert tuple(item.id for item in first) == ("u000001", "u000002", "u000003")
    assert tuple(item.text for item in first) == ("line 0", "line 1", "line 2")
    assert first[1].model_dump() == {
        "id": "u000002",
        "speaker_id": "s1",
        "speaker_name": "Speaker 1",
        "text": "line 1",
        "start_ms": 1_000,
        "end_ms": 1_900,
    }


def test_normalize_falls_back_to_full_text_without_loss_and_rejects_empty() -> None:
    utterances = normalize_transcript(" First   line \n\n Second\tline ", None)
    assert [item.text for item in utterances] == ["First line", "Second line"]
    assert " ".join(item.text for item in utterances) == "First line Second line"

    with pytest.raises(ValueError, match="empty"):
        normalize_transcript(" \n\t ", None)


@pytest.mark.parametrize(
    "segments, message",
    [
        ([{"text": "a", "start_ms": -1}], "nonnegative"),
        ([{"text": "a", "start_ms": 10, "end_ms": 9}], "precede"),
        (
            [
                {"text": "a", "start_ms": 20, "end_ms": 30},
                {"text": "b", "start_ms": 10, "end_ms": 40},
            ],
            "monotonic",
        ),
    ],
)
def test_normalize_rejects_invalid_timing(
    segments: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_transcript("fallback must not hide invalid segments", segments)


def test_chunks_are_exact_token_bounded_overlapped_complete_and_repeatable() -> None:
    tokenizer = ExactTokenizer()
    utterances = normalize_transcript("unused", _segments(10))

    first = build_evidence_chunks(utterances, tokenizer, max_prompt_tokens=7)
    second = build_evidence_chunks(utterances, tokenizer, max_prompt_tokens=7)

    assert first == second
    assert all(chunk.rendered_token_count <= 7 for chunk in first)
    assert [chunk.id for chunk in first] == [
        f"ec{index:06d}" for index in range(1, len(first) + 1)
    ]
    assert set().union(*(set(chunk.utterance_ids) for chunk in first)) == {
        item.id for item in utterances
    }
    for previous, current in zip(first, first[1:], strict=False):
        assert previous.utterance_ids[-4:] == current.utterance_ids[:4]


def test_chunk_plan_binds_transcript_tokenizer_and_boundaries() -> None:
    utterances = normalize_transcript("unused", _segments(8))
    plan = build_evidence_chunk_plan(
        utterances, ExactTokenizer(), max_prompt_tokens=6
    )

    assert plan.tokenizer_model_key == "test/model"
    assert plan.tokenizer_instance_id == "instance-1"
    assert plan.transcript_digest
    assert plan.chunk_plan_digest
    assert plan.utterance_ids == tuple(item.id for item in utterances)
    assert plan.chunks == build_evidence_chunks(
        utterances, ExactTokenizer(), max_prompt_tokens=6
    )


def test_chunk_plan_counts_the_actual_extraction_schema_and_binds_instruction() -> None:
    tokenizer = CapturingTokenizer()
    utterances = normalize_transcript("A fact.", None)

    first = build_evidence_chunk_plan(
        utterances, tokenizer, max_prompt_tokens=4, instruction="Focus on facts."
    )
    second = build_evidence_chunk_plan(
        utterances, tokenizer, max_prompt_tokens=4, instruction="Focus on actions."
    )

    assert all(schema and "candidates" in schema["properties"] for schema in tokenizer.output_schemas)
    assert first.instruction_digest != second.instruction_digest
    assert first.chunk_plan_digest != second.chunk_plan_digest


def test_chunking_fails_on_inexact_tokenizer_oversized_utterance_and_run_cap() -> None:
    tokenizer = ExactTokenizer()
    tokenizer.exact = False
    utterances = normalize_transcript("one", None)
    with pytest.raises(ChunkPlanningError, match="exact tokenizer"):
        build_evidence_chunks(utterances, tokenizer, max_prompt_tokens=2)

    tokenizer.exact = True
    with pytest.raises(ChunkPlanningError, match="cannot fit"):
        build_evidence_chunks(utterances, tokenizer, max_prompt_tokens=1)

    many = normalize_transcript("unused", _segments(42))
    with pytest.raises(ChunkPlanningError, match="40"):
        build_evidence_chunks(
            many,
            tokenizer,
            max_prompt_tokens=2,
            overlap_utterances=0,
        )


def test_chunking_checks_cancellation_between_chunks() -> None:
    utterances = normalize_transcript("unused", _segments(6))
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    with pytest.raises(ChunkPlanningError, match="cancelled"):
        build_evidence_chunks(
            utterances,
            ExactTokenizer(),
            max_prompt_tokens=3,
            overlap_utterances=1,
            is_cancelled=cancelled,
        )
