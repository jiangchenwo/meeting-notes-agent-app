import datetime
import json
import os
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from typing import Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import get_db, SessionLocal
from models import NoteBlock, Transcription
from asr_client import transcribe_via_asr
import asr_config

router = APIRouter(prefix="/api/notes", tags=["transcription"])


def _run_transcription(note_id: int, diarize: bool) -> None:
    db = SessionLocal()
    try:
        note = db.get(NoteBlock, note_id)
        if not note or not note.audio_file_path:
            return

        note.status = "transcribing"
        note.updated_at = datetime.datetime.utcnow()
        db.commit()

        try:
            with open(note.audio_file_path, "rb") as f:
                audio_bytes = f.read()
            data = transcribe_via_asr(
                audio_bytes,
                os.path.basename(note.audio_file_path),
                diarize=diarize,
                base_url=asr_config.load()["base_url"],
            )

            segments = data.get("segments", [])
            full_text = data.get("full_text", "") or ""
            duration_ms = data.get("duration_ms")

            existing = db.query(Transcription).filter_by(note_block_id=note_id).first()
            if existing:
                existing.full_text = full_text
                existing.segments_json = json.dumps(segments)
                existing.model_used = data.get("model_used")
                existing.language = data.get("language")
                existing.diarized = bool(data.get("diarized"))
            else:
                db.add(Transcription(
                    note_block_id=note_id,
                    full_text=full_text,
                    segments_json=json.dumps(segments),
                    model_used=data.get("model_used"),
                    language=data.get("language"),
                    diarized=bool(data.get("diarized")),
                ))

            if note.audio_duration_ms is None and duration_ms is not None:
                note.audio_duration_ms = duration_ms

            note.status = "transcribed"
            note.updated_at = datetime.datetime.utcnow()
            db.commit()

        except Exception:
            note.status = "error"
            note.updated_at = datetime.datetime.utcnow()
            db.commit()

    finally:
        db.close()


@router.post("/{note_id}/transcribe", status_code=202)
def start_transcription(
    note_id: int,
    background_tasks: BackgroundTasks,
    diarize: bool = False,
    db: Session = Depends(get_db),
):
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    if not note.audio_file_path:
        raise HTTPException(400, "Note has no audio file")
    if note.status == "transcribing":
        raise HTTPException(409, "Already transcribing")
    background_tasks.add_task(_run_transcription, note_id, diarize)
    return {"status": "started"}


@router.get("/{note_id}/transcription")
def get_transcription(note_id: int, db: Session = Depends(get_db)):
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    t = db.query(Transcription).filter_by(note_block_id=note_id).first()
    if not t:
        return {"note_id": note_id, "full_text": None, "segments": [], "model_used": None, "language": None, "diarized": False}
    return {
        "note_id": note_id,
        "full_text": t.full_text,
        "segments": json.loads(t.segments_json or "[]"),
        "model_used": t.model_used,
        "language": t.language,
        "diarized": bool(t.diarized),
    }


class TranscriptionUpdate(BaseModel):
    full_text: Optional[str] = None
    segments: Optional[list[dict]] = None


@router.patch("/{note_id}/transcription")
def update_transcription(note_id: int, body: TranscriptionUpdate, db: Session = Depends(get_db)):
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    t = db.query(Transcription).filter_by(note_block_id=note_id).first()
    if not t:
        raise HTTPException(404, "No transcription found for this note")
    if body.segments is not None:
        t.segments_json = json.dumps(body.segments)
        t.full_text = " ".join(s.get("text", "").strip() for s in body.segments)
    elif body.full_text is not None:
        t.full_text = body.full_text
    db.commit()
    return {
        "note_id": note_id,
        "full_text": t.full_text,
        "segments": json.loads(t.segments_json or "[]"),
        "model_used": t.model_used,
        "language": t.language,
    }
