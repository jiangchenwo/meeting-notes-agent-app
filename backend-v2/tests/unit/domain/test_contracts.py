from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from notes_agent_v2.domain.document import (
    DocumentBlock,
    DocumentClaim,
    NotesDocument,
    StructuredItem,
    validate_document_integrity,
)
from notes_agent_v2.domain.evidence import (
    EvidenceChunk,
    EvidenceSpan,
    ExtractedFactCandidate,
    Fact,
    ProjectContextCitation,
    ProjectContextRecord,
    canonical_digest,
    validate_fact_graph,
)
from notes_agent_v2.domain.planning import CapabilityPlan, GenerationBrief, PlannedBlock
from notes_agent_v2.domain.quality import CriticIssue, QualityReport
from notes_agent_v2.domain.run import RunStatus, StageName, validate_run_transition
from notes_agent_v2.domain.transcript import Transcript, Utterance


def utterances() -> tuple[Utterance, ...]:
    return (
        Utterance(id="u000001", speaker_id="s1", speaker_name="Ava", text="Ava proposed launch on Friday.", start_ms=0, end_ms=1000),
        Utterance(id="u000002", speaker_id="s2", speaker_name="Bo", text="Bo approved the Friday launch.", start_ms=1001, end_ms=2000),
    )


def fact(**updates: object) -> Fact:
    values: dict[str, object] = {
        "id": "f000001",
        "text": "The Friday launch was approved.",
        "kind": "decision",
        "status": "approved",
        "speaker_ids": ("s2",),
        "owner": None,
        "due_text": None,
        "confidence": 0.95,
        "verification": "supported",
        "evidence": (EvidenceSpan(utterance_ids=("u000002",), quote="approved the Friday launch"),),
        "source_candidate_ids": ("fc000001",),
        "supersedes_fact_ids": (),
        "conflicts_with_fact_ids": (),
    }
    values.update(updates)
    return Fact(**values)


def test_transcript_contract_is_frozen_strict_and_monotonic() -> None:
    transcript = Transcript(id="t000001", note_id="n000001", utterances=utterances())
    assert transcript.utterances[1].id == "u000002"
    with pytest.raises(ValidationError):
        Utterance(id="u1", text="bad", unexpected=True)
    with pytest.raises(ValidationError):
        Transcript(id="t000001", note_id="n000001", utterances=(utterances()[1], utterances()[0]))
    with pytest.raises(ValidationError):
        transcript.id = "t000002"


def test_evidence_contracts_validate_digests_quotes_and_fact_graph() -> None:
    source = utterances()
    chunk = EvidenceChunk(
        id="ec000001",
        utterance_ids=("u000001", "u000002"),
        rendered_token_count=19,
        digest=canonical_digest({"utterance_ids": ["u000001", "u000002"], "rendered_token_count": 19}),
    )
    assert chunk.rendered_token_count == 19
    with pytest.raises(ValidationError, match="unique"):
        EvidenceChunk(
            id="ec000001", utterance_ids=("u000001", "u000001"), rendered_token_count=19,
            digest=canonical_digest({"utterance_ids": ["u000001", "u000001"], "rendered_token_count": 19}),
        )
    validate_fact_graph((fact(),), source)

    with pytest.raises(ValueError, match="evidence quote"):
        validate_fact_graph((fact(evidence=(EvidenceSpan(utterance_ids=("u000002",), quote="never said"),)),), source)
    with pytest.raises(ValidationError):
        fact(status="uncertain", verification="supported")
    with pytest.raises(ValidationError):
        fact(kind="decision", status="approved", owner="Ava", due_text=None)

    left = fact(id="f000001", conflicts_with_fact_ids=("f000002",))
    right = fact(id="f000002", conflicts_with_fact_ids=())
    with pytest.raises(ValueError, match="symmetric"):
        validate_fact_graph((left, right), source)
    cycle_a = fact(id="f000001", supersedes_fact_ids=("f000002",))
    cycle_b = fact(id="f000002", supersedes_fact_ids=("f000001",))
    with pytest.raises(ValueError, match="acyclic"):
        validate_fact_graph((cycle_a, cycle_b), source)
    with pytest.raises(ValueError, match="monotonically"):
        validate_fact_graph((fact(id="f000002"),), source)


def test_candidate_and_project_context_are_bounded_and_approved() -> None:
    candidate = ExtractedFactCandidate(
        id="fc000001", text="The Friday launch was approved.", kind="decision", status="approved",
        speaker_ids=("s2",), owner=None, due_text=None,
        evidence=(EvidenceSpan(utterance_ids=("u000002",), quote="approved the Friday launch"),),
    )
    assert candidate.evidence
    content = "The project codename is Juniper."
    record = ProjectContextRecord(
        id="pc000001", note_id="n000001", title="Project glossary", content=content,
        digest=canonical_digest(content), approved_at=datetime(2026, 7, 17, tzinfo=UTC),
    )
    citation = ProjectContextCitation(record_id=record.id, quote="codename is Juniper")
    assert citation.quote in record.content
    with pytest.raises(ValidationError):
        ProjectContextRecord(
            id="pc000001", note_id="n000001", title="x", content=content,
            digest="0" * 64, approved_at=datetime(2026, 7, 17),
        )


def test_planning_requires_closed_capabilities_and_required_fact_coverage() -> None:
    brief = GenerationBrief(
        instruction="Write concise notes for the engineering team.", audience="engineering",
        desired_depth="brief", constraints=(), requested_emphasis=("decisions",),
        forbidden_content=(), uncertainties=(),
    )
    assert brief.audience == "engineering"
    plan = CapabilityPlan(
        capabilities=("overview", "decisions"),
        blocks=(
            PlannedBlock(id="b000001", capability="overview", title="Overview", purpose="Summarize", fact_ids=("f000001",), project_context_ids=(), required=True),
            PlannedBlock(id="b000002", capability="decisions", title="Decisions", purpose="Record decisions", fact_ids=("f000002",), project_context_ids=(), required=True),
        ),
        required_fact_ids=("f000001", "f000002"),
    )
    assert len(plan.blocks) == 2
    with pytest.raises(ValidationError, match="required fact"):
        CapabilityPlan(capabilities=("overview",), blocks=(plan.blocks[0],), required_fact_ids=("f000002",))
    with pytest.raises(ValidationError):
        PlannedBlock(id="b000003", capability="finance", title="x", purpose="x", fact_ids=(), project_context_ids=(), required=False)


def test_document_integrity_rejects_context_only_structured_truth() -> None:
    context = ProjectContextRecord(
        id="pc000001", note_id="n000001", title="Glossary", content="The codename is Juniper.",
        digest=canonical_digest("The codename is Juniper."), approved_at=datetime(2026, 7, 17, tzinfo=UTC),
    )
    claim = DocumentClaim(id="c000001", text="The Friday launch was approved.", fact_ids=("f000001",), project_context_citations=())
    item = StructuredItem(id="si000001", kind="decision", text=claim.text, fact_ids=("f000001",), status="approved", owner=None, due_text=None)
    document = NotesDocument(
        id="d000001", run_id="r000001", version=1, parent_id=None, title="Notes",
        blocks=(DocumentBlock(id="b000001", capability="decisions", title="Decisions", claims=(claim,), structured_items=(item,)),),
    )
    validate_document_integrity(document, facts=(fact(),), project_context=(context,), note_id="n000001")

    unsupported = document.model_copy(update={"blocks": (
        document.blocks[0].model_copy(update={"structured_items": (item.model_copy(update={"fact_ids": ()}),)}),
    )})
    with pytest.raises(ValueError, match="structured item"):
        validate_document_integrity(unsupported, facts=(fact(),), project_context=(context,), note_id="n000001")
    with pytest.raises(ValidationError):
        DocumentClaim(id="c000002", text="", fact_ids=(), project_context_citations=())


def test_quality_contract_enforces_counts_targets_and_disposition() -> None:
    issue = CriticIssue(
        id="i000001", critic="claim", severity="critical", category="unsupported",
        block_id="b000001", claim_id="c000001", fact_ids=("f000001",),
        message="The claim is unsupported.", confidence=0.9,
    )
    report = QualityReport(
        disposition="rejected", issues=(issue,), mandatory_coverage=1, total_coverage=1,
        evidence_link_rate=0.5, unsupported_claim_count=1, critic_failure_count=0,
        warning_count=0, revision_count=0,
    )
    assert report.disposition == "rejected"
    with pytest.raises(ValidationError, match="accepted"):
        QualityReport(
            disposition="accepted", issues=(issue,), mandatory_coverage=1, total_coverage=1,
            evidence_link_rate=1, unsupported_claim_count=0, critic_failure_count=0,
            warning_count=0, revision_count=0,
        )
    with pytest.raises(ValidationError):
        CriticIssue.model_validate({**issue.model_dump(), "confidence": 1.1})


def test_run_stage_and_status_literals_reject_arbitrary_transitions() -> None:
    assert StageName("preflight") is StageName.preflight
    assert RunStatus("running") is RunStatus.running
    validate_run_transition(RunStatus.queued, RunStatus.running)
    validate_run_transition(RunStatus.running, RunStatus.review_required)
    with pytest.raises(ValueError, match="run transition"):
        validate_run_transition(RunStatus.completed, RunStatus.running)
    with pytest.raises(ValueError):
        StageName("supervisor")


def test_contracts_expose_no_routing_fields_or_literals() -> None:
    forbidden = {"domain", "domain_id", "template", "workflow_config", "WorkflowSpec"}
    classes = (
        Utterance, EvidenceSpan, EvidenceChunk, ProjectContextRecord,
        ExtractedFactCandidate, Fact, GenerationBrief, PlannedBlock,
        CapabilityPlan, DocumentClaim, StructuredItem, DocumentBlock,
        NotesDocument, CriticIssue, QualityReport,
    )
    assert all(not (set(model.model_fields) & forbidden) for model in classes)


def test_contract_modules_have_no_framework_or_database_imports() -> None:
    root = Path(__file__).resolve().parents[3] / "src/notes_agent_v2/domain"
    source = "\n".join(path.read_text() for path in root.glob("*.py"))
    assert "fastapi" not in source.lower()
    assert "sqlalchemy" not in source.lower()
