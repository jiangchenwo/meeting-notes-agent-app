from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from notes_agent_v2.domain.document import NotesDocument, validate_document_integrity
from notes_agent_v2.domain.evidence import Fact, ProjectContextRecord, canonical_digest, validate_fact_graph
from notes_agent_v2.domain.run import StageName
from notes_agent_v2.domain.transcript import Transcript
from notes_agent_v2.domain.quality import CriticIssue, QualityReport

from .database import Database
from .models import (
    DocumentRow,
    CriticIssueRow,
    FactRow,
    GenerationRunRow,
    ModelCallRow,
    NoteRow,
    ProjectContextRow,
    PromptPresetRow,
    QualityReportRow,
    RunEventRow,
    StageArtifactRow,
    TranscriptRow,
    UtteranceRow,
)


class PersistenceScopeError(RuntimeError):
    pass


class ImmutableRecordError(RuntimeError):
    pass


class RunSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    note_id: str
    transcript_id: str
    instruction: str
    preset_id: str | None
    status: str
    project_context_snapshot: tuple[ProjectContextRecord, ...]


class StageArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run_id: str
    stage: StageName
    version: int
    artifact_type: str
    payload: dict[str, object]
    input_digest: str
    output_digest: str


class RunEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    run_id: str
    sequence: int
    stage: StageName
    status: str
    error_code: str | None = None


class SafeModelCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    run_id: str
    stage: StageName
    runtime_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    status: Literal["completed", "failed", "cancelled"]
    trace_id: str = Field(min_length=1)
    audit_id: str = Field(min_length=1)


class PromptPreset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(pattern=r"^p[0-9]{6}$")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    tags: tuple[str, ...]


class NoteRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, identifier: str, title: str) -> None:
        with self.database.session() as session:
            session.add(NoteRow(id=identifier, title=title))


class TranscriptRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def put(self, transcript: Transcript) -> None:
        payload = transcript.model_dump_json()
        with self.database.session() as session:
            session.add(TranscriptRow(id=transcript.id, note_id=transcript.note_id, digest=canonical_digest(transcript.model_dump(mode="json")), payload_json=payload))
            session.add_all(
                UtteranceRow(transcript_id=transcript.id, id=item.id, ordinal=index, payload_json=item.model_dump_json())
                for index, item in enumerate(transcript.utterances)
            )

    def get(self, identifier: str) -> Transcript:
        with self.database.session() as session:
            row = session.get(TranscriptRow, identifier)
            if row is None:
                raise KeyError(identifier)
            return Transcript.model_validate_json(row.payload_json)


class ProjectContextRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def put(self, record: ProjectContextRecord) -> None:
        with self.database.session() as session:
            session.add(ProjectContextRow(
                id=record.id,
                note_id=record.note_id,
                title=record.title,
                content=record.content,
                digest=record.digest,
                approved_at=record.approved_at.isoformat(),
                tombstoned_at=None,
            ))

    def tombstone(self, identifier: str) -> None:
        with self.database.session() as session:
            row = session.get(ProjectContextRow, identifier)
            if row is None:
                raise KeyError(identifier)
            row.tombstoned_at = datetime.now(UTC).isoformat()


class RunRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(
        self,
        identifier: str,
        *,
        note_id: str,
        transcript_id: str,
        instruction: str,
        project_context_ids: tuple[str, ...],
        idempotency_key: str,
        preset_id: str | None = None,
    ) -> RunSnapshot:
        if not instruction.strip():
            raise ValueError("run instruction must not be blank")
        if len(project_context_ids) != len(set(project_context_ids)):
            raise ValueError("project context snapshot IDs must be unique")
        with self.database.session() as session:
            transcript = session.get(TranscriptRow, transcript_id)
            if transcript is None or transcript.note_id != note_id:
                raise PersistenceScopeError("transcript must belong to the same note")
            records: list[ProjectContextRecord] = []
            for context_id in project_context_ids:
                row = session.get(ProjectContextRow, context_id)
                if row is None or row.note_id != note_id:
                    raise PersistenceScopeError("project context must belong to the same note")
                if row.tombstoned_at is not None:
                    raise PersistenceScopeError("tombstoned project context cannot enter a new run")
                records.append(ProjectContextRecord(
                    id=row.id,
                    note_id=row.note_id,
                    title=row.title,
                    content=row.content,
                    digest=row.digest,
                    approved_at=datetime.fromisoformat(row.approved_at),
                ))
            session.add(GenerationRunRow(
                id=identifier,
                note_id=note_id,
                transcript_id=transcript_id,
                preset_id=preset_id,
                instruction_snapshot=instruction,
                project_context_snapshot_json=json.dumps([item.model_dump(mode="json") for item in records], sort_keys=True, separators=(",", ":")),
                status="queued",
                idempotency_key=idempotency_key,
                cancellation_requested=0,
            ))
        return self.get(identifier)

    def get(self, identifier: str) -> RunSnapshot:
        with self.database.session() as session:
            row = session.get(GenerationRunRow, identifier)
            if row is None:
                raise KeyError(identifier)
            context = tuple(ProjectContextRecord.model_validate(item) for item in json.loads(row.project_context_snapshot_json))
            return RunSnapshot(
                id=row.id,
                note_id=row.note_id,
                transcript_id=row.transcript_id,
                instruction=row.instruction_snapshot,
                preset_id=row.preset_id,
                status=row.status,
                project_context_snapshot=context,
            )

    def create_from_preset(
        self,
        identifier: str,
        *,
        note_id: str,
        transcript_id: str,
        preset_id: str,
        project_context_ids: tuple[str, ...],
        idempotency_key: str,
    ) -> RunSnapshot:
        with self.database.session() as session:
            preset = session.get(PromptPresetRow, preset_id)
            if preset is None or preset.tombstoned_at is not None:
                raise KeyError(preset_id)
            instruction = preset.instruction
        return self.create(
            identifier,
            note_id=note_id,
            transcript_id=transcript_id,
            instruction=instruction,
            project_context_ids=project_context_ids,
            idempotency_key=idempotency_key,
            preset_id=preset_id,
        )


class FactRepository:
    def __init__(self, database: Database, transcripts: TranscriptRepository) -> None:
        self.database = database
        self.transcripts = transcripts

    def put_many(self, run_id: str, facts: tuple[Fact, ...]) -> None:
        with self.database.session() as session:
            run = session.get(GenerationRunRow, run_id)
            if run is None:
                raise KeyError(run_id)
            transcript = Transcript.model_validate_json(session.get(TranscriptRow, run.transcript_id).payload_json)
            validate_fact_graph(facts, transcript.utterances)
            session.add_all(FactRow(run_id=run_id, id=item.id, payload_json=item.model_dump_json()) for item in facts)

    def list(self, run_id: str) -> tuple[Fact, ...]:
        with self.database.session() as session:
            rows = session.scalars(select(FactRow).where(FactRow.run_id == run_id).order_by(FactRow.id)).all()
            return tuple(Fact.model_validate_json(row.payload_json) for row in rows)


class DocumentRepository:
    def __init__(self, database: Database, facts: FactRepository, runs: RunRepository) -> None:
        self.database = database
        self.facts = facts
        self.runs = runs

    def put(self, document: NotesDocument) -> None:
        with self.database.session() as session:
            existing = session.get(DocumentRow, document.id)
            if existing is not None:
                if existing.payload_json == document.model_dump_json():
                    return
                raise ImmutableRecordError("document record is immutable")
        run = self.runs.get(document.run_id)
        facts = self.facts.list(document.run_id)
        referenced = {identifier for block in document.blocks for claim in block.claims for identifier in claim.fact_ids}
        referenced.update(identifier for block in document.blocks for item in block.structured_items for identifier in item.fact_ids)
        if not referenced <= {item.id for item in facts}:
            raise PersistenceScopeError("document facts must belong to the same run")
        validate_document_integrity(document, facts=facts, project_context=run.project_context_snapshot, note_id=run.note_id)
        with self.database.session() as session:
            session.add(DocumentRow(
                id=document.id,
                run_id=document.run_id,
                version=document.version,
                parent_id=document.parent_id,
                payload_json=document.model_dump_json(),
            ))

    def get(self, identifier: str) -> NotesDocument:
        with self.database.session() as session:
            row = session.get(DocumentRow, identifier)
            if row is None:
                raise KeyError(identifier)
            return NotesDocument.model_validate_json(row.payload_json)


class ArtifactRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def list(self, run_id: str) -> tuple[StageArtifact, ...]:
        with self.database.session() as session:
            rows = session.scalars(select(StageArtifactRow).where(StageArtifactRow.run_id == run_id).order_by(StageArtifactRow.id)).all()
            return tuple(StageArtifact(
                run_id=row.run_id,
                stage=row.stage,
                version=row.version,
                artifact_type=row.artifact_type,
                payload=json.loads(row.payload_json),
                input_digest=row.input_digest,
                output_digest=row.output_digest,
            ) for row in rows)


class EventRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def list(self, run_id: str) -> tuple[RunEvent, ...]:
        with self.database.session() as session:
            rows = session.scalars(select(RunEventRow).where(RunEventRow.run_id == run_id).order_by(RunEventRow.sequence)).all()
            return tuple(RunEvent(id=row.id, run_id=row.run_id, sequence=row.sequence, stage=row.stage, status=row.status, error_code=row.error_code) for row in rows)


class ModelCallRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def put(self, record: SafeModelCallRecord) -> None:
        with self.database.session() as session:
            existing = session.get(ModelCallRow, record.id)
            if existing is not None:
                if self._from_row(existing) == record:
                    return
                raise ImmutableRecordError("model call record is immutable")
            session.add(ModelCallRow(**record.model_dump(mode="json")))

    def get(self, identifier: str) -> SafeModelCallRecord:
        with self.database.session() as session:
            row = session.get(ModelCallRow, identifier)
            if row is None:
                raise KeyError(identifier)
            return self._from_row(row)

    @staticmethod
    def _from_row(row: ModelCallRow) -> SafeModelCallRecord:
        return SafeModelCallRecord.model_validate({column.name: getattr(row, column.name) for column in ModelCallRow.__table__.columns})


class PresetRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _validate_values(name: str, description: str, instruction: str, tags: tuple[str, ...]) -> None:
        if not name.strip() or not description.strip() or not instruction.strip():
            raise ValueError("preset name, description, and instruction must not be blank")
        if any(not tag.strip() for tag in tags) or len(tags) != len(set(tags)):
            raise ValueError("preset tags must be non-blank and unique")

    def create(self, *, name: str, description: str, instruction: str, tags: tuple[str, ...]) -> PromptPreset:
        self._validate_values(name, description, instruction, tags)
        with self.database.session() as session:
            identifiers = session.scalars(select(PromptPresetRow.id).order_by(PromptPresetRow.id.desc()).limit(1)).all()
            number = int(identifiers[0][1:]) + 1 if identifiers else 1
            identifier = f"p{number:06d}"
            session.add(PromptPresetRow(
                id=identifier,
                name=name,
                description=description,
                instruction=instruction,
                tags_json=json.dumps(list(tags), sort_keys=True, separators=(",", ":")),
                tombstoned_at=None,
            ))
        return self.get(identifier)

    def get(self, identifier: str) -> PromptPreset:
        with self.database.session() as session:
            row = session.get(PromptPresetRow, identifier)
            if row is None or row.tombstoned_at is not None:
                raise KeyError(identifier)
            return self._from_row(row)

    def list(self, *, tag: str | None = None) -> tuple[PromptPreset, ...]:
        with self.database.session() as session:
            rows = session.scalars(select(PromptPresetRow).where(PromptPresetRow.tombstoned_at.is_(None)).order_by(PromptPresetRow.id)).all()
            presets = tuple(self._from_row(row) for row in rows)
            return tuple(item for item in presets if tag is None or tag in item.tags)

    def update(
        self,
        identifier: str,
        *,
        name: str | None = None,
        description: str | None = None,
        instruction: str | None = None,
        tags: tuple[str, ...] | None = None,
    ) -> PromptPreset:
        with self.database.session() as session:
            row = session.get(PromptPresetRow, identifier)
            if row is None or row.tombstoned_at is not None:
                raise KeyError(identifier)
            next_name = row.name if name is None else name
            next_description = row.description if description is None else description
            next_instruction = row.instruction if instruction is None else instruction
            next_tags = tuple(json.loads(row.tags_json)) if tags is None else tags
            self._validate_values(next_name, next_description, next_instruction, next_tags)
            row.name = next_name
            row.description = next_description
            row.instruction = next_instruction
            row.tags_json = json.dumps(list(next_tags), sort_keys=True, separators=(",", ":"))
        return self.get(identifier)

    def tombstone(self, identifier: str) -> None:
        with self.database.session() as session:
            row = session.get(PromptPresetRow, identifier)
            if row is None or row.tombstoned_at is not None:
                raise KeyError(identifier)
            row.tombstoned_at = datetime.now(UTC).isoformat()

    @staticmethod
    def _from_row(row: PromptPresetRow) -> PromptPreset:
        return PromptPreset(
            id=row.id,
            name=row.name,
            description=row.description,
            instruction=row.instruction,
            tags=tuple(json.loads(row.tags_json)),
        )


class CriticIssueRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def put_many(self, run_id: str, document_id: str, issues: tuple[CriticIssue, ...]) -> None:
        if len({item.id for item in issues}) != len(issues):
            raise ValueError("critic issue IDs must be unique")
        with self.database.session() as session:
            document = session.get(DocumentRow, document_id)
            if document is None or document.run_id != run_id:
                raise PersistenceScopeError("critic issues and document must belong to the same run")
            session.add_all(CriticIssueRow(
                id=item.id,
                run_id=run_id,
                document_id=document_id,
                payload_json=item.model_dump_json(),
            ) for item in issues)

    def list(self, document_id: str) -> tuple[CriticIssue, ...]:
        with self.database.session() as session:
            rows = session.scalars(select(CriticIssueRow).where(CriticIssueRow.document_id == document_id).order_by(CriticIssueRow.id)).all()
            return tuple(CriticIssue.model_validate_json(row.payload_json) for row in rows)


class QualityReportRepository:
    def __init__(self, database: Database, critic_issues: CriticIssueRepository) -> None:
        self.database = database
        self.critic_issues = critic_issues

    def put(self, run_id: str, document_id: str, report: QualityReport) -> None:
        with self.database.session() as session:
            document = session.get(DocumentRow, document_id)
            if document is None or document.run_id != run_id:
                raise PersistenceScopeError("quality report and document must belong to the same run")
        if self.critic_issues.list(document_id) != report.issues:
            raise PersistenceScopeError("quality report issues must match the persisted document issues")
        with self.database.session() as session:
            session.add(QualityReportRow(
                document_id=document_id,
                run_id=run_id,
                payload_json=report.model_dump_json(),
            ))

    def get(self, document_id: str) -> QualityReport:
        with self.database.session() as session:
            row = session.get(QualityReportRow, document_id)
            if row is None:
                raise KeyError(document_id)
            return QualityReport.model_validate_json(row.payload_json)


class Repositories:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.notes = NoteRepository(database)
        self.transcripts = TranscriptRepository(database)
        self.context = ProjectContextRepository(database)
        self.runs = RunRepository(database)
        self.facts = FactRepository(database, self.transcripts)
        self.documents = DocumentRepository(database, self.facts, self.runs)
        self.artifacts = ArtifactRepository(database)
        self.events = EventRepository(database)
        self.model_calls = ModelCallRepository(database)
        self.presets = PresetRepository(database)
        self.critic_issues = CriticIssueRepository(database)
        self.quality = QualityReportRepository(database, self.critic_issues)

    def store_stage_and_event(
        self,
        *,
        run_id: str,
        stage: StageName,
        version: int,
        artifact_type: str,
        payload: dict[str, object],
        input_digest: str,
        output_digest: str,
        event_id: str,
        fail_after_artifact: bool = False,
    ) -> None:
        with self.database.session() as session:
            session.add(StageArtifactRow(
                run_id=run_id,
                stage=stage,
                version=version,
                artifact_type=artifact_type,
                payload_json=json.dumps(payload, sort_keys=True, separators=(",", ":")),
                input_digest=input_digest,
                output_digest=output_digest,
                dependency_digests_json="[]",
            ))
            session.flush()
            if fail_after_artifact:
                raise RuntimeError("injected repository failure")
            sequence = session.scalar(select(RunEventRow.sequence).where(RunEventRow.run_id == run_id).order_by(RunEventRow.sequence.desc()).limit(1))
            session.add(RunEventRow(
                id=event_id,
                run_id=run_id,
                sequence=(sequence or 0) + 1,
                stage=stage,
                status="completed",
                error_code=None,
            ))
