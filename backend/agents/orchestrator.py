"""DB wrapper around the DB-free pipeline.

`run_workflow(note_id)` is the FastAPI BackgroundTask entry point: it loads the
note + transcription, builds NoteDeps and the WorkflowSpec, runs
`pipeline.run_pipeline` with a _DBObserver that persists WorkflowRun /
WorkflowStepResult rows as steps execute, then writes the Summary.
"""
import datetime
import json
import logging

import lm_config
from database import SessionLocal
from models import NoteBlock, Summary, Transcription, WorkflowRun, WorkflowStepResult
from telemetry import workflow_span
from transcript_format import build_speaker_transcript

from .context import NoteDeps
from .pipeline import PipelineObserver, run_pipeline
from .workflow_spec import select_workflow

logger = logging.getLogger("agents.orchestrator")

DEFAULT_TEMPLATE_PROMPT = "Summarize the meeting: key decisions, action items, and blockers."


def build_note_deps(note: NoteBlock, cfg: dict) -> NoteDeps:
    """Compose the per-note agent dependencies. Shared with the prompt-preview
    endpoint so the preview matches what the pipeline actually sends."""
    return NoteDeps(
        note_id=note.id,
        domain_name=note.domain.name if note.domain else "Project",
        template_name=note.template.name if note.template else "Default",
        template_prompt=(
            note.template.prompt_template
            if note.template and note.template.prompt_template
            else DEFAULT_TEMPLATE_PROMPT
        ),
        project_system_prompt=note.project.custom_system_prompt or "" if note.project else "",
        project_knowledge_base=note.project.knowledge_base or "" if note.project else "",
        global_system_prompt=cfg.get("global_system_prompt") or "",
    )


class _DBObserver(PipelineObserver):
    """Persists pipeline progress so the polling endpoints see live state."""

    def __init__(self, db, run: WorkflowRun):
        self.db = db
        self.run = run

    def phase(self, phase: str) -> None:
        self.run.status = phase
        if phase == "assembling":
            self.run.current_step = "assembling"
        self.db.commit()

    def step_start(self, step_name: str, attempt: int, current_step: str) -> WorkflowStepResult:
        rec = WorkflowStepResult(
            run_id=self.run.id, step_name=step_name, status="running", attempt=attempt
        )
        self.db.add(rec)
        self.run.current_step = current_step
        self.db.commit()
        return rec

    def step_done(
        self,
        token: WorkflowStepResult,
        *,
        duration_ms: int,
        result: dict,
        critique_score: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        model_name: str | None = None,
    ) -> None:
        token.status = "done"
        token.duration_ms = duration_ms
        token.result_json = json.dumps(result)
        token.critique_score = critique_score
        token.input_tokens = input_tokens
        token.output_tokens = output_tokens
        token.model_name = model_name
        if input_tokens:
            self.run.total_input_tokens = (self.run.total_input_tokens or 0) + input_tokens
        if output_tokens:
            self.run.total_output_tokens = (self.run.total_output_tokens or 0) + output_tokens
        if model_name:
            self.run.model_name = model_name
        self.db.commit()

    def step_error(self, token: WorkflowStepResult, *, duration_ms: int, error: str) -> None:
        token.status = "error"
        token.duration_ms = duration_ms
        token.result_json = json.dumps({"error": error})
        self.db.commit()


def run_workflow(note_id: int) -> None:
    db = SessionLocal()
    run: WorkflowRun | None = None
    try:
        note = db.get(NoteBlock, note_id)
        if not note:
            return

        transcription = db.query(Transcription).filter_by(note_block_id=note_id).first()
        if not transcription or not transcription.full_text:
            note.status = "error"
            db.commit()
            return

        run = WorkflowRun(note_block_id=note_id, status="planning")
        db.add(run)
        note.status = "summarizing"
        note.updated_at = datetime.datetime.utcnow()
        db.commit()

        cfg = lm_config.load()
        deps = build_note_deps(note, cfg)
        domain_name = deps.domain_name
        spec = select_workflow(
            domain_name, note.template.workflow_config if note.template else None
        )
        run.workflow_plan_json = spec.model_dump_json()
        db.commit()

        transcript = build_speaker_transcript(
            transcription.full_text, transcription.segments_json
        )
        logger.info(
            "workflow run %d for note %d: domain=%s steps=%s transcript=%d chars",
            run.id, note_id, domain_name, spec.step_names, len(transcript),
        )

        with workflow_span(note_id) as trace_id:
            run.trace_id = trace_id
            result = run_pipeline(
                transcript=transcript,
                spec=spec,
                deps=deps,
                cfg=cfg,
                observer=_DBObserver(db, run),
            )

        raw_sections = {
            **result.results,
            "_schema_checks": result.schema_checks,
            "_risk_classification": result.risk_classification,
        }

        now = datetime.datetime.utcnow()
        model = result.model_name or cfg.get("model", "")
        existing = db.query(Summary).filter_by(note_block_id=note_id).first()
        if existing:
            existing.summary_text = result.summary_text
            existing.action_items_json = json.dumps(result.action_items)
            existing.suggestions_text = result.suggestions_text
            existing.llm_model_used = model
            existing.generated_at = now
            existing.workflow_run_id = run.id
            existing.confidence_score = result.confidence_score
            existing.raw_sections_json = json.dumps(raw_sections)
        else:
            db.add(Summary(
                note_block_id=note_id,
                summary_text=result.summary_text,
                action_items_json=json.dumps(result.action_items),
                suggestions_text=result.suggestions_text,
                llm_model_used=model,
                generated_at=now,
                workflow_run_id=run.id,
                confidence_score=result.confidence_score,
                raw_sections_json=json.dumps(raw_sections),
            ))

        run.total_input_tokens = result.input_tokens
        run.total_output_tokens = result.output_tokens
        run.model_name = result.model_name or None
        note.status = "done"
        note.updated_at = now
        run.status = "done"
        run.finished_at = now
        db.commit()
        logger.info(
            "workflow run %d done: confidence=%s tokens=%d in / %d out",
            run.id, result.confidence_score, result.input_tokens, result.output_tokens,
        )

    except Exception as exc:
        logger.exception("workflow for note %d failed", note_id)
        db.rollback()
        try:
            note = db.get(NoteBlock, note_id)
            if note:
                note.status = "error"
                note.updated_at = datetime.datetime.utcnow()
            if run:
                run.status = "error"
                run.error_message = str(exc)
                run.finished_at = datetime.datetime.utcnow()
            db.commit()
        except Exception:
            pass
    finally:
        db.close()
