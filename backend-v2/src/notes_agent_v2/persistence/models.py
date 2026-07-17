from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class NoteRow(Base):
    __tablename__ = "notes"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)


class TranscriptRow(Base):
    __tablename__ = "transcripts"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    note_id: Mapped[str] = mapped_column(ForeignKey("notes.id"), nullable=False)
    digest: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


class UtteranceRow(Base):
    __tablename__ = "utterances"
    transcript_id: Mapped[str] = mapped_column(ForeignKey("transcripts.id"), primary_key=True)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (UniqueConstraint("transcript_id", "ordinal"),)


class ProjectContextRow(Base):
    __tablename__ = "project_context_records"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    note_id: Mapped[str] = mapped_column(ForeignKey("notes.id"), nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    digest: Mapped[str] = mapped_column(String(64), nullable=False)
    approved_at: Mapped[str] = mapped_column(String, nullable=False)
    tombstoned_at: Mapped[str | None] = mapped_column(String)


class PromptPresetRow(Base):
    __tablename__ = "prompt_presets"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    instruction: Mapped[str] = mapped_column(Text, nullable=False)
    tags_json: Mapped[str] = mapped_column(Text, nullable=False)
    tombstoned_at: Mapped[str | None] = mapped_column(String)


class GenerationRunRow(Base):
    __tablename__ = "generation_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    note_id: Mapped[str] = mapped_column(ForeignKey("notes.id"), nullable=False)
    transcript_id: Mapped[str] = mapped_column(ForeignKey("transcripts.id"), nullable=False)
    preset_id: Mapped[str | None] = mapped_column(ForeignKey("prompt_presets.id"))
    instruction_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    project_context_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued")
    idempotency_key: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    runtime_fingerprint: Mapped[str | None] = mapped_column(String(64))
    profile_fingerprint: Mapped[str | None] = mapped_column(String(64))
    prompt_fingerprint: Mapped[str | None] = mapped_column(String(64))
    schema_fingerprint: Mapped[str | None] = mapped_column(String(64))
    lease_owner: Mapped[str | None] = mapped_column(String)
    lease_token: Mapped[str | None] = mapped_column(String)
    lease_expires_at: Mapped[str | None] = mapped_column(String)
    cancellation_requested: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class StageArtifactRow(Base):
    __tablename__ = "stage_artifacts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("generation_runs.id"), nullable=False)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    artifact_type: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    input_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    output_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    dependency_digests_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    __table_args__ = (UniqueConstraint("run_id", "stage", "artifact_type", "version"),)


class FactRow(Base):
    __tablename__ = "facts"
    run_id: Mapped[str] = mapped_column(ForeignKey("generation_runs.id"), primary_key=True)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


class DocumentRow(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("generation_runs.id"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"))
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (UniqueConstraint("run_id", "version"),)


class CriticIssueRow(Base):
    __tablename__ = "critic_issues"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("generation_runs.id"), nullable=False)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


class QualityReportRow(Base):
    __tablename__ = "quality_reports"
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("generation_runs.id"), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


class RunEventRow(Base):
    __tablename__ = "run_events"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("generation_runs.id"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String)
    __table_args__ = (UniqueConstraint("run_id", "sequence"),)


class ModelCallRow(Base):
    __tablename__ = "model_call_records"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("generation_runs.id"), nullable=False)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    runtime_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    trace_id: Mapped[str] = mapped_column(String, nullable=False)
    audit_id: Mapped[str] = mapped_column(String, nullable=False)
