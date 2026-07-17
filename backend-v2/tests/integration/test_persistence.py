from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import inspect
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError

from notes_agent_v2.domain.document import DocumentBlock, DocumentClaim, NotesDocument, StructuredItem
from notes_agent_v2.domain.evidence import EvidenceSpan, Fact, ProjectContextRecord, canonical_digest
from notes_agent_v2.domain.transcript import Transcript, Utterance
from notes_agent_v2.domain.quality import CriticIssue, QualityReport
from notes_agent_v2.persistence.database import Database, upgrade_database
from notes_agent_v2.persistence.repositories import (
    ImmutableRecordError,
    PersistenceScopeError,
    Repositories,
    SafeModelCallRecord,
)


def transcript() -> Transcript:
    return Transcript(
        id="t000001", note_id="n000001",
        utterances=(Utterance(id="u000001", speaker_id="s1", speaker_name="Ava", text="The launch was approved.", start_ms=0, end_ms=1000),),
    )


def supported_fact(identifier: str = "f000001") -> Fact:
    return Fact(
        id=identifier, text="The launch was approved.", kind="decision", status="approved",
        speaker_ids=("s1",), owner=None, due_text=None, confidence=1,
        verification="supported",
        evidence=(EvidenceSpan(utterance_ids=("u000001",), quote="launch was approved"),),
        source_candidate_ids=("fc000001",), supersedes_fact_ids=(), conflicts_with_fact_ids=(),
    )


def document(run_id: str = "r000001", fact_id: str = "f000001") -> NotesDocument:
    claim = DocumentClaim(id="c000001", text="The launch was approved.", fact_ids=(fact_id,), project_context_citations=())
    item = StructuredItem(id="si000001", kind="decision", text=claim.text, fact_ids=(fact_id,), status="approved", owner=None, due_text=None)
    return NotesDocument(
        id="d000001", run_id=run_id, version=1, parent_id=None, title="Notes",
        blocks=(DocumentBlock(id="b000001", capability="decisions", title="Decisions", claims=(claim,), structured_items=(item,)),),
    )


@pytest.fixture
def repositories(tmp_path: Path) -> Repositories:
    url = f"sqlite:///{tmp_path / 'notes.db'}"
    upgrade_database(url)
    return Repositories(Database(url))


def test_repositories_round_trip_contracts_and_snapshot_context(repositories: Repositories) -> None:
    repositories.notes.create("n000001", "Launch meeting")
    repositories.transcripts.put(transcript())
    content = "The project codename is Juniper."
    context = ProjectContextRecord(
        id="pc000001", note_id="n000001", title="Glossary", content=content,
        digest=canonical_digest(content), approved_at=datetime(2026, 7, 17, tzinfo=UTC),
    )
    repositories.context.put(context)
    repositories.runs.create(
        "r000001", note_id="n000001", transcript_id="t000001", instruction="Write meeting notes.",
        project_context_ids=("pc000001",), idempotency_key="request-1",
    )
    repositories.facts.put_many("r000001", (supported_fact(),))
    repositories.documents.put(document())

    assert repositories.transcripts.get("t000001") == transcript()
    assert repositories.facts.list("r000001") == (supported_fact(),)
    assert repositories.documents.get("d000001") == document()
    before = repositories.runs.get("r000001").project_context_snapshot
    repositories.context.tombstone("pc000001")
    assert repositories.runs.get("r000001").project_context_snapshot == before
    assert before[0].content == content


def test_repositories_reject_cross_run_and_cross_note_trust_widening(repositories: Repositories) -> None:
    repositories.notes.create("n000001", "One")
    repositories.notes.create("n000002", "Two")
    repositories.transcripts.put(transcript())
    repositories.runs.create("r000001", note_id="n000001", transcript_id="t000001", instruction="Notes", project_context_ids=(), idempotency_key="one")
    repositories.runs.create("r000002", note_id="n000001", transcript_id="t000001", instruction="Notes", project_context_ids=(), idempotency_key="two")
    repositories.facts.put_many("r000001", (supported_fact(),))
    with pytest.raises(PersistenceScopeError, match="same run"):
        repositories.documents.put(document(run_id="r000002"))

    content = "Private context for another note."
    repositories.context.put(ProjectContextRecord(
        id="pc000001", note_id="n000002", title="Other", content=content,
        digest=canonical_digest(content), approved_at=datetime(2026, 7, 17, tzinfo=UTC),
    ))
    with pytest.raises(PersistenceScopeError, match="same note"):
        repositories.runs.create("r000003", note_id="n000001", transcript_id="t000001", instruction="Notes", project_context_ids=("pc000001",), idempotency_key="three")


def test_run_rejects_duplicate_project_context_snapshots(repositories: Repositories) -> None:
    repositories.notes.create("n000001", "One")
    repositories.transcripts.put(transcript())
    content = "Approved context."
    repositories.context.put(ProjectContextRecord(
        id="pc000001", note_id="n000001", title="Context", content=content,
        digest=canonical_digest(content), approved_at=datetime(2026, 7, 17, tzinfo=UTC),
    ))
    with pytest.raises(ValueError, match="unique"):
        repositories.runs.create(
            "r000001", note_id="n000001", transcript_id="t000001", instruction="Notes",
            project_context_ids=("pc000001", "pc000001"), idempotency_key="one",
        )


def test_immutable_records_and_transaction_rollback(repositories: Repositories) -> None:
    repositories.notes.create("n000001", "One")
    repositories.transcripts.put(transcript())
    repositories.runs.create("r000001", note_id="n000001", transcript_id="t000001", instruction="Notes", project_context_ids=(), idempotency_key="one")
    repositories.facts.put_many("r000001", (supported_fact(),))
    repositories.documents.put(document())
    with pytest.raises(ImmutableRecordError):
        repositories.documents.put(document().model_copy(update={"title": "Changed"}))

    with pytest.raises(RuntimeError, match="injected"):
        repositories.store_stage_and_event(
            run_id="r000001", stage="extract", version=1, artifact_type="facts",
            payload={"count": 1}, input_digest="1" * 64, output_digest="2" * 64,
            event_id="e000001", fail_after_artifact=True,
        )
    assert repositories.artifacts.list("r000001") == ()
    assert repositories.events.list("r000001") == ()


def test_model_call_records_are_safe_and_immutable(repositories: Repositories) -> None:
    repositories.notes.create("n000001", "One")
    repositories.transcripts.put(transcript())
    repositories.runs.create("r000001", note_id="n000001", transcript_id="t000001", instruction="Notes", project_context_ids=(), idempotency_key="one")
    record = SafeModelCallRecord(
        id="mc000001", run_id="r000001", stage="extract", runtime_fingerprint="1" * 64,
        profile_fingerprint="2" * 64, prompt_fingerprint="3" * 64, schema_fingerprint="4" * 64,
        input_tokens=100, output_tokens=20, latency_ms=250, status="completed",
        trace_id="trace-1", audit_id="audit-1",
    )
    repositories.model_calls.put(record)
    assert repositories.model_calls.get(record.id) == record
    columns = {item["name"] for item in inspect(repositories.database.engine).get_columns("model_call_records")}
    forbidden = {"prompt", "transcript", "output", "tool_arguments", "secret", "authorization", "reasoning"}
    assert not columns & forbidden
    with pytest.raises(ImmutableRecordError):
        repositories.model_calls.put(record.model_copy(update={"latency_ms": 251}))


def test_critic_and_quality_contracts_round_trip_and_are_database_immutable(repositories: Repositories) -> None:
    repositories.notes.create("n000001", "One")
    repositories.transcripts.put(transcript())
    repositories.runs.create("r000001", note_id="n000001", transcript_id="t000001", instruction="Notes", project_context_ids=(), idempotency_key="one")
    repositories.facts.put_many("r000001", (supported_fact(),))
    repositories.documents.put(document())
    issue = CriticIssue(
        id="i000001", critic="claim", severity="critical", category="unsupported",
        block_id="b000001", claim_id="c000001", fact_ids=("f000001",),
        message="Unsupported claim.", confidence=1,
    )
    report = QualityReport(
        disposition="rejected", issues=(issue,), mandatory_coverage=1, total_coverage=1,
        evidence_link_rate=0, unsupported_claim_count=1, critic_failure_count=0,
        warning_count=0, revision_count=0,
    )
    repositories.critic_issues.put_many("r000001", "d000001", (issue,))
    repositories.quality.put("r000001", "d000001", report)
    assert repositories.critic_issues.list("d000001") == (issue,)
    assert repositories.quality.get("d000001") == report

    with pytest.raises(DatabaseError):
        with repositories.database.session() as session:
            session.execute(text("UPDATE quality_reports SET payload_json = '{}' WHERE document_id = 'd000001'"))
