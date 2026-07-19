from __future__ import annotations

from notes_agent_v2.domain.evidence import (
    EvidenceSpan,
    ExtractedFactCandidate,
    validate_fact_graph,
)
from notes_agent_v2.workflow.consolidate import consolidate_candidates
from notes_agent_v2.workflow.preflight import normalize_transcript


class Groups:
    def __init__(self, *groups: tuple[str, ...]) -> None:
        self.groups = groups

    def propose(self, candidates):
        del candidates
        return self.groups


def _candidate(
    number: int,
    text: str,
    utterance_id: str,
    quote: str,
    *,
    kind: str = "fact",
    status: str = "asserted",
    speaker: str = "s1",
    owner: str | None = None,
    due: str | None = None,
) -> ExtractedFactCandidate:
    return ExtractedFactCandidate(
        id=f"fc{number:06d}",
        text=text,
        kind=kind,
        status=status,
        speaker_ids=(speaker,),
        owner=owner,
        due_text=due,
        evidence=(EvidenceSpan(utterance_ids=(utterance_id,), quote=quote),),
    )


def _utterances():
    return normalize_transcript(
        "unused",
        [
            {"text": "The budget is 12 million.", "speaker_id": "s1"},
            {"text": "The budget is 12 million.", "speaker_id": "s1"},
            {"text": "Funding totals twelve million dollars.", "speaker_id": "s1"},
            {"text": "I propose launching in May.", "speaker_id": "s1"},
            {"text": "We approved launching in May.", "speaker_id": "s1"},
            {"text": "Correction: the budget is 14 million, not 12.", "speaker_id": "s1"},
            {"text": "The budget is 13 million.", "speaker_id": "s2"},
            {"text": "Ship the API.", "speaker_id": "s1"},
            {"text": "Alice will ship the API by Friday.", "speaker_id": "s1"},
            {"text": "The mobile risk is latency.", "speaker_id": "s1"},
            {"text": "The desktop risk is latency.", "speaker_id": "s1"},
        ],
    )


def test_exact_and_semantic_duplicates_merge_without_losing_evidence() -> None:
    candidates = (
        _candidate(1, "The budget is 12 million.", "u000001", "The budget is 12 million."),
        _candidate(2, "The budget is 12 million.", "u000002", "The budget is 12 million."),
        _candidate(3, "Funding is twelve million dollars.", "u000003", "Funding totals twelve million dollars."),
    )
    result = consolidate_candidates(
        candidates,
        _utterances(),
        semantic_grouper=Groups(("fc000001", "fc000003")),
    )

    assert len(result.facts) == 1
    fact = result.facts[0]
    assert fact.id == "f000001"
    assert fact.source_candidate_ids == ("fc000001", "fc000002", "fc000003")
    assert {span.utterance_ids[0] for span in fact.evidence} == {
        "u000001", "u000002", "u000003"
    }
    assert set(result.candidate_to_fact) == {item.id for item in candidates}


def test_status_change_and_explicit_correction_preserve_history() -> None:
    candidates = (
        _candidate(1, "Launch in May.", "u000004", "I propose launching in May.", kind="proposal", status="proposed"),
        _candidate(2, "Launch in May.", "u000005", "We approved launching in May.", kind="decision", status="approved"),
        _candidate(3, "The budget is 12 million.", "u000001", "The budget is 12 million."),
        _candidate(4, "The budget is 14 million.", "u000006", "Correction: the budget is 14 million, not 12.", kind="correction"),
    )
    result = consolidate_candidates(candidates, _utterances())
    by_candidate = {
        candidate_id: result.facts_by_id[fact_id]
        for candidate_id, fact_id in result.candidate_to_fact.items()
    }

    assert by_candidate["fc000002"].supersedes_fact_ids == (
        by_candidate["fc000001"].id,
    )
    assert by_candidate["fc000004"].supersedes_fact_ids == (
        by_candidate["fc000003"].id,
    )
    assert len(result.facts) == 4


def test_conflicts_are_symmetric_and_candidates_remain_inspectable() -> None:
    candidates = (
        _candidate(1, "The budget is 12 million.", "u000001", "The budget is 12 million.", speaker="s1"),
        _candidate(2, "The budget is 13 million.", "u000007", "The budget is 13 million.", speaker="s2"),
    )
    result = consolidate_candidates(candidates, _utterances())

    assert result.facts[0].conflicts_with_fact_ids == (result.facts[1].id,)
    assert result.facts[1].conflicts_with_fact_ids == (result.facts[0].id,)
    validate_fact_graph(result.facts, _utterances())


def test_repeated_action_merges_richer_owner_and_due_fields() -> None:
    candidates = (
        _candidate(1, "Ship the API.", "u000008", "Ship the API.", kind="action"),
        _candidate(2, "Ship the API.", "u000009", "Alice will ship the API by Friday.", kind="action", owner="Alice", due="Friday"),
    )
    result = consolidate_candidates(candidates, _utterances())
    assert len(result.facts) == 1
    assert (result.facts[0].owner, result.facts[0].due_text) == ("Alice", "Friday")
    assert len(result.facts[0].evidence) == 2


def test_invalid_semantic_group_cannot_merge_unrelated_similar_phrases() -> None:
    candidates = (
        _candidate(1, "The mobile risk is latency.", "u000010", "The mobile risk is latency.", kind="risk"),
        _candidate(2, "The desktop risk is latency.", "u000011", "The desktop risk is latency.", kind="risk"),
    )
    result = consolidate_candidates(
        candidates,
        _utterances(),
        semantic_grouper=Groups(("fc000001", "fc000002")),
    )
    assert len(result.facts) == 2


def test_token_reordering_and_shared_evidence_do_not_collapse_distinct_facts() -> None:
    utterances = normalize_transcript(
        "Alice blocked Bob.\nBob blocked Alice.\nThe budget is 12 and launch is May.",
        None,
    )
    candidates = (
        _candidate(1, "Alice blocked Bob.", "u000001", "Alice blocked Bob."),
        _candidate(2, "Bob blocked Alice.", "u000002", "Bob blocked Alice."),
        _candidate(3, "The budget is 12.", "u000003", "The budget is 12 and launch is May."),
        _candidate(4, "Launch is in May.", "u000003", "The budget is 12 and launch is May."),
    )

    result = consolidate_candidates(candidates, utterances)

    assert len(result.facts) == 4
    assert len(set(result.candidate_to_fact.values())) == 4


def test_final_ids_follow_first_evidence_not_candidate_or_model_order() -> None:
    candidates = (
        _candidate(9, "The desktop risk is latency.", "u000011", "The desktop risk is latency.", kind="risk"),
        _candidate(1, "The budget is 12 million.", "u000001", "The budget is 12 million."),
    )
    result = consolidate_candidates(candidates, _utterances())
    assert [fact.text for fact in result.facts] == [
        "The budget is 12 million.",
        "The desktop risk is latency.",
    ]
    assert [fact.id for fact in result.facts] == ["f000001", "f000002"]
