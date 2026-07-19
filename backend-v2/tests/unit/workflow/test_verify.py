from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from notes_agent_v2.domain.evidence import EvidenceSpan, ExtractedFactCandidate
from notes_agent_v2.runtime.budget import RunBudget
from notes_agent_v2.workflow.preflight import normalize_transcript
from notes_agent_v2.workflow.verify import verified_candidates, verify_candidates


class SemanticGateway:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.requests = []

    def call(self, request, *, budget, validate):
        del budget
        self.requests.append(request)
        content = json.dumps(self.response)
        assert validate(content)
        return SimpleNamespace(response=SimpleNamespace(final_content=content))


def _candidate(
    text: str,
    *,
    quote: str,
    ids: tuple[str, ...] = ("u000002",),
    kind: str = "fact",
    status: str = "asserted",
    speakers: tuple[str, ...] = ("s1",),
    owner: str | None = None,
    due: str | None = None,
    number: int = 1,
) -> ExtractedFactCandidate:
    return ExtractedFactCandidate(
        id=f"fc{number:06d}",
        text=text,
        kind=kind,
        status=status,
        speaker_ids=speakers,
        owner=owner,
        due_text=due,
        evidence=(EvidenceSpan(utterance_ids=ids, quote=quote),),
    )


def _utterances():
    return normalize_transcript(
        "unused",
        [
            {"text": "The meeting date is July 10.", "speaker_id": "s0"},
            {
                "text": "Alice will ship 12 units by July 18; email alice@example.com and use https://example.com/ship.",
                "speaker_id": "s1",
            },
            {"text": "We did not approve option B.", "speaker_id": "s2"},
            {"text": "Correction: the quantity is 14, not 12.", "speaker_id": "s1"},
            {"text": "The rollout creates a material delivery risk.", "speaker_id": "s1"},
        ],
    )


@pytest.mark.parametrize(
    "candidate, finding",
    [
        (_candidate("Alice will ship 13 units.", quote="Alice will ship 12 units", kind="action", owner="Alice"), "number_mismatch"),
        (_candidate("Alice ships by July 19.", quote="Alice will ship 12 units by July 18", kind="action", owner="Alice", due="July 19"), "date_mismatch"),
        (_candidate("Contact bob@example.com.", quote="email alice@example.com"), "email_mismatch"),
        (_candidate("Use https://wrong.example/.", quote="use https://example.com/ship"), "url_mismatch"),
        (_candidate("Option B was approved.", quote="We did not approve option B.", ids=("u000003",), kind="decision", status="approved", speakers=("s2",)), "status_or_negation_mismatch"),
        (_candidate("The quantity is 12.", quote="Correction: the quantity is 14, not 12.", ids=("u000004",), kind="correction"), "correction_mismatch"),
        (_candidate("Bob will ship 12 units.", quote="Alice will ship 12 units", kind="action", owner="Bob"), "owner_not_cooccurring"),
        (_candidate("Alice ships by July 10.", quote="The meeting date is July 10.", ids=("u000001",), kind="action", owner="Alice", due="July 10", speakers=("s0",)), "meeting_date_as_due"),
    ],
)
def test_deterministic_corruptions_are_contradicted(
    candidate: ExtractedFactCandidate, finding: str
) -> None:
    gateway = SemanticGateway({"status": "supported", "evidence_ids": ["u000002"]})
    decisions = verify_candidates(
        run_id="r1",
        candidates=(candidate,),
        utterances=_utterances(),
        gateway=gateway,
        budget=RunBudget(),
    )

    assert decisions[0].status == "contradicted"
    assert finding in decisions[0].deterministic_findings
    assert gateway.requests == []
    assert verified_candidates((candidate,), decisions) == ()


def test_exact_source_statement_is_supported_without_model_call() -> None:
    candidate = _candidate(
        "Alice will ship 12 units by July 18.",
        quote="Alice will ship 12 units by July 18",
        kind="action",
        owner="Alice",
        due="July 18",
    )
    gateway = SemanticGateway({"status": "uncertain", "evidence_ids": []})
    decisions = verify_candidates(
        run_id="r1",
        candidates=(candidate,),
        utterances=_utterances(),
        gateway=gateway,
        budget=RunBudget(),
    )
    assert decisions[0].status == "supported"
    assert gateway.requests == []
    assert verified_candidates((candidate,), decisions) == (candidate,)


def test_ambiguous_paraphrase_uses_only_bounded_neighbor_window() -> None:
    candidate = _candidate(
        "The delivery schedule is materially threatened.",
        quote="The rollout creates a material delivery risk.",
        ids=("u000005",),
    )
    gateway = SemanticGateway(
        {"status": "supported", "evidence_ids": ["u000005"]}
    )
    decisions = verify_candidates(
        run_id="r1",
        candidates=(candidate,),
        utterances=_utterances(),
        gateway=gateway,
        budget=RunBudget(),
    )
    assert decisions[0].status == "supported"
    assert decisions[0].semantic_finding == "supported"
    payload = json.loads(gateway.requests[0].messages[-1]["content"])
    assert len(payload["source_window"]) <= 8
    assert gateway.requests[0].profile_name == "structured_off"


def test_semantic_disagreement_becomes_uncertain_and_preserves_both_findings() -> None:
    candidate = _candidate(
        "The delivery schedule is materially threatened.",
        quote="The rollout creates a material delivery risk.",
        ids=("u000005",),
    )
    gateway = SemanticGateway(
        {"status": "contradicted", "evidence_ids": ["u000005"]}
    )
    decision = verify_candidates(
        run_id="r1",
        candidates=(candidate,),
        utterances=_utterances(),
        gateway=gateway,
        budget=RunBudget(),
    )[0]
    assert decision.status == "uncertain"
    assert "deterministic_ambiguous" in decision.deterministic_findings
    assert decision.semantic_finding == "contradicted"


def test_semantic_verifier_cannot_widen_evidence_scope() -> None:
    candidate = _candidate(
        "The delivery schedule is materially threatened.",
        quote="The rollout creates a material delivery risk.",
        ids=("u000005",),
    )
    gateway = SemanticGateway(
        {"status": "supported", "evidence_ids": ["u999999"]}
    )
    decision = verify_candidates(
        run_id="r1",
        candidates=(candidate,),
        utterances=_utterances(),
        gateway=gateway,
        budget=RunBudget(),
    )[0]
    assert decision.status == "uncertain"
    assert decision.error_code == "semantic_scope_violation"
