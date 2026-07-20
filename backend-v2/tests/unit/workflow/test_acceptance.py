from __future__ import annotations

from notes_agent_v2.domain.document import DocumentBlock, DocumentClaim, NotesDocument
from notes_agent_v2.domain.evidence import EvidenceSpan, Fact
from notes_agent_v2.domain.quality import CriticIssue
from notes_agent_v2.workflow.acceptance import DraftCandidate, evaluate_acceptance, rank_drafts


def _fact(identifier: str, *, verification: str = "supported") -> Fact:
    text = f"Fact {identifier}"
    return Fact(
        id=identifier,
        text=text,
        kind="fact",
        status="uncertain" if verification == "uncertain" else "asserted",
        speaker_ids=("s1",),
        owner=None,
        due_text=None,
        confidence=1,
        verification=verification,
        evidence=(EvidenceSpan(utterance_ids=("u000001",), quote=text),),
        source_candidate_ids=("fc000001",),
        supersedes_fact_ids=(),
        conflicts_with_fact_ids=(),
    )


def _document(*fact_ids: str, version: int = 1) -> NotesDocument:
    return NotesDocument(
        id=f"d{version:06d}",
        run_id="r000001",
        version=version,
        parent_id=None if version == 1 else f"d{version - 1:06d}",
        title="Notes",
        blocks=(
            DocumentBlock(
                id="b000001",
                capability="overview",
                title="Overview",
                claims=tuple(
                    DocumentClaim(
                        id=f"c{index:06d}",
                        text=f"Fact {identifier}",
                        fact_ids=(identifier,),
                        project_context_citations=(),
                    )
                    for index, identifier in enumerate(fact_ids, start=1)
                ),
                structured_items=(),
            ),
        ),
    )


def _issue(
    *,
    critic: str = "claim",
    severity: str = "critical",
    category: str = "contradiction",
    identifier: int = 1,
) -> CriticIssue:
    return CriticIssue(
        id=f"i{identifier:06d}",
        critic=critic,
        severity=severity,
        category=category,
        block_id=None if critic == "system" else "b000001",
        claim_id=None,
        fact_ids=() if critic == "system" else ("f000001",),
        message="Issue",
        confidence=None,
    )


def test_acceptance_order_routes_critic_failure_to_review_before_rejection() -> None:
    facts = (_fact("f000001"), _fact("f000002"))
    report = evaluate_acceptance(
        document=_document("f000001"),
        facts=facts,
        mandatory_fact_ids=("f000002",),
        issues=(
            _issue(critic="system", category="critic_failure"),
            _issue(identifier=2),
        ),
    )
    assert report.disposition == "review_required"
    assert report.critic_failure_count == 1
    assert report.mandatory_coverage == 0


def test_critical_missing_or_unsupported_content_is_rejected() -> None:
    supported = _fact("f000001")
    uncertain = _fact("f000002", verification="uncertain")
    critical = evaluate_acceptance(
        document=_document("f000001"),
        facts=(supported,),
        mandatory_fact_ids=(),
        issues=(_issue(),),
    )
    missing = evaluate_acceptance(
        document=_document("f000001"),
        facts=(supported, uncertain),
        mandatory_fact_ids=(uncertain.id,),
        issues=(),
    )
    unsupported = evaluate_acceptance(
        document=_document("f000002"),
        facts=(supported, uncertain),
        mandatory_fact_ids=(),
        issues=(),
    )
    assert [item.disposition for item in (critical, missing, unsupported)] == [
        "rejected",
        "rejected",
        "rejected",
    ]
    assert unsupported.unsupported_claim_count == 1


def test_warning_policy_accepts_up_to_five_unless_constraint_is_violated() -> None:
    fact = _fact("f000001")
    warnings = tuple(
        _issue(severity="warning", category=f"warning_{index}", identifier=index)
        for index in range(1, 6)
    )
    accepted = evaluate_acceptance(
        document=_document(fact.id),
        facts=(fact,),
        mandatory_fact_ids=(fact.id,),
        issues=warnings,
    )
    review = evaluate_acceptance(
        document=_document(fact.id),
        facts=(fact,),
        mandatory_fact_ids=(fact.id,),
        issues=warnings,
        explicit_constraint_violated=True,
    )
    assert accepted.disposition == "accepted"
    assert review.disposition == "review_required"


def test_draft_ranking_is_lexicographic_and_factual_before_fluency() -> None:
    facts = (_fact("f000001"), _fact("f000002"))
    clean = evaluate_acceptance(
        document=_document("f000001", "f000002", version=2),
        facts=facts,
        mandatory_fact_ids=("f000001",),
        issues=(),
        revision_count=1,
    )
    warning = evaluate_acceptance(
        document=_document("f000001", "f000002"),
        facts=facts,
        mandatory_fact_ids=("f000001",),
        issues=(_issue(severity="warning"),),
    )
    critical = evaluate_acceptance(
        document=_document("f000001", "f000002"),
        facts=facts,
        mandatory_fact_ids=("f000001",),
        issues=(_issue(),),
    )
    ranked = rank_drafts(
        (
            DraftCandidate(document=_document("f000001", "f000002"), report=critical),
            DraftCandidate(document=_document("f000001", "f000002"), report=warning),
            DraftCandidate(document=_document("f000001", "f000002", version=2), report=clean),
        )
    )
    assert [candidate.report for candidate in ranked] == [clean, warning, critical]
