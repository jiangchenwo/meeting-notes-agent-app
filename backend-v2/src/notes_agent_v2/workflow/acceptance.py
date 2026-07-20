from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from notes_agent_v2.domain.document import NotesDocument
from notes_agent_v2.domain.evidence import Fact
from notes_agent_v2.domain.quality import CriticIssue, QualityReport


class DraftCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document: NotesDocument
    report: QualityReport


def evaluate_acceptance(
    *,
    document: NotesDocument,
    facts: Sequence[Fact],
    mandatory_fact_ids: Sequence[str],
    issues: Sequence[CriticIssue],
    revision_count: int = 0,
    explicit_constraint_violated: bool = False,
) -> QualityReport:
    fact_by_id = {item.id: item for item in facts}
    outputs = [
        output
        for block in document.blocks
        for output in (*block.claims, *block.structured_items)
    ]
    covered = {
        identifier for output in outputs for identifier in output.fact_ids
    }
    linked = sum(bool(output.fact_ids) for output in outputs)
    evidence_link_rate = linked / len(outputs) if outputs else 0
    supported_ids = {
        identifier
        for identifier, fact in fact_by_id.items()
        if fact.verification == "supported"
    }
    unsupported_claim_count = sum(
        any(identifier not in supported_ids for identifier in output.fact_ids)
        for output in outputs
    )
    mandatory = tuple(dict.fromkeys(mandatory_fact_ids))
    mandatory_coverage = (
        sum(identifier in covered and identifier in supported_ids for identifier in mandatory)
        / len(mandatory)
        if mandatory
        else 1
    )
    total_coverage = (
        len(covered & supported_ids) / len(supported_ids) if supported_ids else 1
    )
    issue_tuple = tuple(issues)
    critic_failure_count = sum(
        item.critic == "system" and item.category == "critic_failure"
        for item in issue_tuple
    )
    warning_count = sum(item.severity == "warning" for item in issue_tuple)
    critical_count = sum(item.severity == "critical" for item in issue_tuple)
    if critic_failure_count:
        disposition = "review_required"
    elif (
        critical_count
        or mandatory_coverage < 1
        or unsupported_claim_count
        or evidence_link_rate < 1
    ):
        disposition = "rejected"
    elif explicit_constraint_violated or warning_count > 5:
        disposition = "review_required"
    else:
        disposition = "accepted"
    return QualityReport(
        disposition=disposition,
        issues=issue_tuple,
        mandatory_coverage=mandatory_coverage,
        total_coverage=total_coverage,
        evidence_link_rate=evidence_link_rate,
        unsupported_claim_count=unsupported_claim_count,
        critic_failure_count=critic_failure_count,
        warning_count=warning_count,
        revision_count=revision_count,
    )


def rank_drafts(candidates: Sequence[DraftCandidate]) -> tuple[DraftCandidate, ...]:
    def key(candidate: DraftCandidate) -> tuple[float | int, ...]:
        report = candidate.report
        other_critical = sum(
            issue.severity == "critical"
            and not (issue.critic == "system" and issue.category == "critic_failure")
            for issue in report.issues
        )
        return (
            report.critic_failure_count,
            other_critical,
            1 - report.mandatory_coverage,
            report.unsupported_claim_count,
            report.warning_count,
            -report.mandatory_coverage,
            -report.total_coverage,
            report.revision_count,
        )

    return tuple(sorted(candidates, key=key))
