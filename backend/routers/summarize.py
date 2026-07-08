"""Summary read/edit endpoints + the legacy generate alias.

POST /summarize is kept for API compatibility but now delegates to the agentic
workflow (agents.orchestrator.run_workflow) — the old single-call LLM path is
gone. Per-step progress is available from the /workflow-run endpoints.
"""
import json
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional
from database import get_db
from models import NoteBlock, Transcription, Summary
from transcript_format import build_speaker_transcript
from agents import run_workflow
from agents.definitions import INSTRUCTION_BUILDERS, build_user_prompt
from agents.orchestrator import build_note_deps
from agents.pipeline import _truncate_transcript
import lm_config

router = APIRouter(prefix="/api/notes", tags=["summarization"])


@router.post("/{note_id}/summarize", status_code=202)
def start_summarization(
    note_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    if note.status in ("summarizing", "transcribing"):
        raise HTTPException(409, "Note is already being processed")
    t = db.query(Transcription).filter_by(note_block_id=note_id).first()
    if not t or not t.full_text:
        raise HTTPException(400, "Note has no transcription")
    background_tasks.add_task(run_workflow, note_id)
    return {"status": "started"}


@router.get("/{note_id}/summary")
def get_summary(note_id: int, db: Session = Depends(get_db)):
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    s = db.query(Summary).filter_by(note_block_id=note_id).first()
    if not s:
        return {
            "note_id": note_id,
            "summary_text": None,
            "action_items": [],
            "suggestions_text": None,
            "llm_model_used": None,
            "generated_at": None,
        }
    return {
        "note_id": note_id,
        "summary_text": s.summary_text,
        "action_items": json.loads(s.action_items_json or "[]"),
        "suggestions_text": s.suggestions_text,
        "llm_model_used": s.llm_model_used,
        "generated_at": s.generated_at.isoformat() if s.generated_at else None,
    }


class SummaryUpdate(BaseModel):
    summary_text: Optional[str] = None
    action_items: Optional[list[dict]] = None
    suggestions_text: Optional[str] = None


@router.patch("/{note_id}/summary")
def update_summary(note_id: int, body: SummaryUpdate, db: Session = Depends(get_db)):
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    s = db.query(Summary).filter_by(note_block_id=note_id).first()
    if not s:
        raise HTTPException(404, "No summary found for this note")
    if body.summary_text is not None:
        s.summary_text = body.summary_text
    if body.action_items is not None:
        s.action_items_json = json.dumps(body.action_items)
    if body.suggestions_text is not None:
        s.suggestions_text = body.suggestions_text
    db.commit()
    return {
        "note_id": note_id,
        "summary_text": s.summary_text,
        "action_items": json.loads(s.action_items_json or "[]"),
        "suggestions_text": s.suggestions_text,
        "llm_model_used": s.llm_model_used,
        "generated_at": s.generated_at.isoformat() if s.generated_at else None,
    }


@router.get("/{note_id}/prompt-preview")
def get_prompt_preview(note_id: int, db: Session = Depends(get_db)):
    """Preview of the Summarizer step's composed prompt — built from the same
    instruction functions the pipeline uses, so it matches the real run."""
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    t = db.query(Transcription).filter_by(note_block_id=note_id).first()
    if not t:
        raise HTTPException(400, "No transcription available")
    cfg = lm_config.load()
    deps = build_note_deps(note, cfg)
    transcript = _truncate_transcript(
        build_speaker_transcript(t.full_text, t.segments_json), cfg
    )
    return {
        "system": INSTRUCTION_BUILDERS["Summarizer"](deps),
        "user": build_user_prompt("Summarizer", deps, transcript),
    }
