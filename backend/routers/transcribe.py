import datetime
import json
import os
import subprocess
import tempfile
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from typing import Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import get_db, SessionLocal
from models import NoteBlock, Transcription
import whisper_config as wcfg

router = APIRouter(prefix="/api/notes", tags=["transcription"])

WHISPER_BINARY_PATH = os.getenv("WHISPER_BINARY_PATH", "")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "")


def _find_binary() -> str:
    if WHISPER_BINARY_PATH:
        candidate = os.path.join(WHISPER_BINARY_PATH, "whisper-cli")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        f"whisper-cli not found in WHISPER_BINARY_PATH={WHISPER_BINARY_PATH!r}. "
        "Set WHISPER_BINARY_PATH in backend/.env"
    )


def _find_model() -> str:
    if WHISPER_MODEL_PATH and os.path.isfile(WHISPER_MODEL_PATH):
        return WHISPER_MODEL_PATH
    if WHISPER_BINARY_PATH:
        # whisper.cpp layout: build/bin/ → ../../models/
        whisper_root = os.path.abspath(os.path.join(WHISPER_BINARY_PATH, "..", ".."))
        models_dir = os.path.join(whisper_root, "models")
        for name in [f"ggml-{WHISPER_MODEL}.bin", f"ggml-{WHISPER_MODEL}.en.bin"]:
            path = os.path.join(models_dir, name)
            if os.path.isfile(path):
                return path
    raise FileNotFoundError(
        f"Model '{WHISPER_MODEL}' not found. Set WHISPER_MODEL_PATH to the .bin file."
    )


def _run_transcription(note_id: int) -> None:
    db = SessionLocal()
    try:
        note = db.get(NoteBlock, note_id)
        if not note or not note.audio_file_path:
            return

        note.status = "transcribing"
        note.updated_at = datetime.datetime.utcnow()
        db.commit()

        try:
            whisper_bin = _find_binary()
            model_path = _find_model()

            with tempfile.TemporaryDirectory() as tmpdir:
                output_prefix = os.path.join(tmpdir, "out")
                cmd = [
                    whisper_bin,
                    "-m", model_path,
                    "-f", note.audio_file_path,
                    "--output-json",
                    "-of", output_prefix,
                    "--no-prints",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                output_json = output_prefix + ".json"
                if result.returncode != 0 or not os.path.isfile(output_json):
                    note.status = "error"
                    db.commit()
                    return

                with open(output_json) as f:
                    data = json.load(f)

                raw_segs = data.get("transcription", [])
                full_text = " ".join(s.get("text", "").strip() for s in raw_segs).strip()
                segments = [
                    {
                        "start": s.get("offsets", {}).get("from", 0) / 1000.0,
                        "end": s.get("offsets", {}).get("to", 0) / 1000.0,
                        "text": s.get("text", "").strip(),
                    }
                    for s in raw_segs
                    if s.get("text", "").strip()
                ]

                duration_ms = int(segments[-1]["end"] * 1000) if segments else None
                existing = db.query(Transcription).filter_by(note_block_id=note_id).first()
                if existing:
                    existing.full_text = full_text
                    existing.segments_json = json.dumps(segments)
                    existing.model_used = WHISPER_MODEL
                    existing.duration_ms = duration_ms
                else:
                    db.add(Transcription(
                        note_block_id=note_id,
                        full_text=full_text,
                        segments_json=json.dumps(segments),
                        model_used=WHISPER_MODEL,
                        duration_ms=duration_ms,
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
    db: Session = Depends(get_db),
):
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    if not note.audio_file_path:
        raise HTTPException(400, "Note has no audio file")
    if note.status == "transcribing":
        raise HTTPException(409, "Already transcribing")
    background_tasks.add_task(_run_transcription, note_id)
    return {"status": "started"}


@router.get("/{note_id}/transcription")
def get_transcription(note_id: int, db: Session = Depends(get_db)):
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    t = db.query(Transcription).filter_by(note_block_id=note_id).first()
    if not t:
        return {"note_id": note_id, "full_text": None, "segments": [], "model_used": None, "language": None}
    return {
        "note_id": note_id,
        "full_text": t.full_text,
        "segments": json.loads(t.segments_json or "[]"),
        "model_used": t.model_used,
        "language": t.language,
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
