from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from notes_agent_v2.domain.evidence import EvidenceChunk, canonical_digest
from notes_agent_v2.runtime.budget import RunBudget
from notes_agent_v2.workflow.extract import extract_cited_facts
from notes_agent_v2.workflow.preflight import normalize_transcript


class FakeGateway:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.requests = []

    def call(self, request, *, budget, validate):
        del budget
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert validate(response)
        return SimpleNamespace(response=SimpleNamespace(final_content=response))


def _chunk(*ids: str, number: int = 1) -> EvidenceChunk:
    payload = {"utterance_ids": list(ids), "rendered_token_count": 20}
    return EvidenceChunk(
        id=f"ec{number:06d}",
        utterance_ids=ids,
        rendered_token_count=20,
        digest=canonical_digest(payload),
    )


def _candidate(
    text: str,
    kind: str,
    status: str,
    quote: str,
    *,
    utterance_ids: list[str],
    speaker_ids: list[str] | None = None,
    owner: str | None = None,
    due_text: str | None = None,
) -> dict[str, object]:
    return {
        "text": text,
        "kind": kind,
        "status": status,
        "speaker_ids": speaker_ids or [],
        "owner": owner,
        "due_text": due_text,
        "evidence": [{"utterance_ids": utterance_ids, "quote": quote}],
    }


def test_extracts_all_atomic_fact_kinds_with_exact_citations_and_stable_ids() -> None:
    utterances = normalize_transcript(
        "unused",
        [
            {"text": "Revenue is up 12 percent.", "speaker_id": "s1"},
            {"text": "We approved the launch.", "speaker_id": "s2"},
            {"text": "Alice will ship the API by Friday.", "speaker_id": "s1"},
            {"text": "I propose a beta first.", "speaker_id": "s2"},
            {"text": "We rejected the desktop client.", "speaker_id": "s1"},
            {"text": "Correction: the total is 14, not 12.", "speaker_id": "s2"},
            {"text": "Who owns the migration?", "speaker_id": "s1"},
            {"text": "The vendor delay is a risk.", "speaker_id": "s2"},
        ],
    )
    candidates = [
        _candidate("Revenue is up 12 percent.", "fact", "asserted", "Revenue is up 12 percent.", utterance_ids=["u000001"], speaker_ids=["s1"]),
        _candidate("The launch was approved.", "decision", "approved", "We approved the launch.", utterance_ids=["u000002"], speaker_ids=["s2"]),
        _candidate("Alice will ship the API by Friday.", "action", "asserted", "Alice will ship the API by Friday.", utterance_ids=["u000003"], speaker_ids=["s1"], owner="Alice", due_text="Friday"),
        _candidate("Run a beta first.", "proposal", "proposed", "I propose a beta first.", utterance_ids=["u000004"], speaker_ids=["s2"]),
        _candidate("The desktop client was rejected.", "proposal", "rejected", "We rejected the desktop client.", utterance_ids=["u000005"], speaker_ids=["s1"]),
        _candidate("The corrected total is 14.", "correction", "asserted", "Correction: the total is 14, not 12.", utterance_ids=["u000006"], speaker_ids=["s2"]),
        _candidate("Who owns the migration?", "question", "asserted", "Who owns the migration?", utterance_ids=["u000007"], speaker_ids=["s1"]),
        _candidate("The vendor delay is a risk.", "risk", "asserted", "The vendor delay is a risk.", utterance_ids=["u000008"], speaker_ids=["s2"]),
    ]
    gateway = FakeGateway([json.dumps({"candidates": candidates})])
    chunk = _chunk(*(item.id for item in utterances))

    first = extract_cited_facts(
        run_id="r1",
        instruction="Focus on delivery commitments.",
        chunks=(chunk,),
        utterances=utterances,
        gateway=gateway,
        budget=RunBudget(),
    )
    second = extract_cited_facts(
        run_id="r1",
        instruction="Focus on delivery commitments.",
        chunks=(chunk,),
        utterances=utterances,
        gateway=FakeGateway([json.dumps({"candidates": candidates})]),
        budget=RunBudget(),
    )

    assert first.complete is True
    assert first.candidates == second.candidates
    assert len({item.id for item in first.candidates}) == 8
    assert {item.kind for item in first.candidates} == {
        "fact", "decision", "action", "proposal", "correction", "question", "risk"
    }
    assert gateway.requests[0].profile_name == "structured_off"
    assert gateway.requests[0].tools == ()
    assert "Focus on delivery commitments." in gateway.requests[0].messages[-1]["content"]
    system = gateway.requests[0].messages[0]["content"]
    assert "verbatim" in system
    assert "non-null speaker_id" in system
    assert "owner and due_text" in system


def test_no_fact_chunk_is_completed_without_candidates() -> None:
    utterances = normalize_transcript("Thanks everyone.", None)
    result = extract_cited_facts(
        run_id="r1",
        instruction="Summarize.",
        chunks=(_chunk("u000001"),),
        utterances=utterances,
        gateway=FakeGateway([json.dumps({"candidates": []})]),
        budget=RunBudget(),
    )
    assert result.complete is True
    assert result.candidates == ()


def test_extraction_profile_can_be_bounded_by_the_caller() -> None:
    utterances = normalize_transcript("Known text.", None)
    gateway = FakeGateway([json.dumps({"candidates": []})])

    extract_cited_facts(
        run_id="r1",
        instruction="Summarize.",
        chunks=(_chunk("u000001"),),
        utterances=utterances,
        gateway=gateway,
        budget=RunBudget(),
        profile_name="evaluation_structured_off",
    )

    assert gateway.requests[0].profile_name == "evaluation_structured_off"


@pytest.mark.parametrize(
    "candidate, error",
    [
        (_candidate("x", "fact", "asserted", "Known text.", utterance_ids=["u999999"]), "unknown_utterance"),
        (_candidate("x", "fact", "asserted", "Known text.", utterance_ids=["u000001"], speaker_ids=["missing"]), "speaker"),
        (_candidate("x", "proposal", "approved", "I propose option A.", utterance_ids=["u000002"], speaker_ids=["s2"]), "proposal"),
    ],
)
def test_rejects_unfounded_candidate_fields(candidate: dict[str, object], error: str) -> None:
    utterances = normalize_transcript(
        "unused",
        [
            {"text": "Known text.", "speaker_id": "s1"},
            {"text": "I propose option A.", "speaker_id": "s2"},
        ],
    )
    result = extract_cited_facts(
        run_id="r1",
        instruction="Summarize.",
        chunks=(_chunk("u000001", "u000002"),),
        utterances=utterances,
        gateway=FakeGateway([json.dumps({"candidates": [candidate]})]),
        budget=RunBudget(),
    )
    assert result.complete is False
    assert error in (result.chunks[0].error_code or "")
    assert result.candidates == ()


def test_canonicalizes_model_quote_to_exact_cited_source_span() -> None:
    utterances = normalize_transcript(
        "unused",
        [
            {"text": "Ship the API", "speaker_id": "s1"},
            {"text": "by Friday.", "speaker_id": "s1"},
        ],
    )
    candidate = _candidate(
        "Ship the API by Friday.",
        "action",
        "asserted",
        "The API should ship Friday.",
        utterance_ids=["u000001", "u000002"],
    )

    result = extract_cited_facts(
        run_id="r1",
        instruction="Summarize.",
        chunks=(_chunk("u000001", "u000002"),),
        utterances=utterances,
        gateway=FakeGateway([json.dumps({"candidates": [candidate]})]),
        budget=RunBudget(),
    )

    assert result.complete is True
    assert result.candidates[0].evidence[0].quote == "Ship the API\nby Friday."


def test_drops_optional_owner_and_due_when_cited_source_does_not_state_them() -> None:
    utterances = normalize_transcript(
        "The migration must happen.",
        None,
    )
    candidate = _candidate(
        "Migrate the service.",
        "action",
        "asserted",
        "The migration must happen.",
        utterance_ids=["u000001"],
        owner="Alice",
        due_text="Friday",
    )

    result = extract_cited_facts(
        run_id="r1",
        instruction="Summarize.",
        chunks=(_chunk("u000001"),),
        utterances=utterances,
        gateway=FakeGateway([json.dumps({"candidates": [candidate]})]),
        budget=RunBudget(),
    )

    assert result.complete is True
    assert result.candidates[0].owner is None
    assert result.candidates[0].due_text is None


def test_normalizes_cited_speaker_display_name_to_source_id() -> None:
    utterances = normalize_transcript(
        "unused",
        [{"text": "The launch is approved.", "speaker_id": "s1", "speaker_name": "Alice"}],
    )
    candidate = _candidate(
        "The launch is approved.",
        "decision",
        "approved",
        "The launch is approved.",
        utterance_ids=["u000001"],
        speaker_ids=["Alice"],
    )

    result = extract_cited_facts(
        run_id="r1",
        instruction="Summarize.",
        chunks=(_chunk("u000001"),),
        utterances=utterances,
        gateway=FakeGateway([json.dumps({"candidates": [candidate]})]),
        budget=RunBudget(),
    )

    assert result.complete is True
    assert result.candidates[0].speaker_ids == ("s1",)


def test_parser_or_timeout_failure_is_persisted_per_chunk_and_run_is_incomplete() -> None:
    utterances = normalize_transcript("First fact.\nSecond fact.", None)
    persisted = []
    gateway = FakeGateway(["not json", TimeoutError("late")])
    result = extract_cited_facts(
        run_id="r1",
        instruction="Summarize.",
        chunks=(_chunk("u000001"), _chunk("u000002", number=2)),
        utterances=utterances,
        gateway=gateway,
        budget=RunBudget(),
        persist_artifact=persisted.append,
    )

    assert result.complete is False
    assert [item.status for item in result.chunks] == ["failed", "failed"]
    assert len(persisted) == 2
    assert all(item.artifact_digest for item in persisted)
    assert result.candidates == ()


def test_successful_chunk_candidates_survive_a_later_chunk_failure() -> None:
    utterances = normalize_transcript("First fact.\nSecond fact.", None)
    valid = _candidate(
        "First fact.",
        "fact",
        "asserted",
        "First fact.",
        utterance_ids=["u000001"],
    )
    result = extract_cited_facts(
        run_id="r1",
        instruction="Summarize.",
        chunks=(_chunk("u000001"), _chunk("u000002", number=2)),
        utterances=utterances,
        gateway=FakeGateway([json.dumps({"candidates": [valid]}), "not json"]),
        budget=RunBudget(),
    )

    assert result.complete is False
    assert len(result.candidates) == 1
    assert result.chunks[0].candidate_ids == (result.candidates[0].id,)
