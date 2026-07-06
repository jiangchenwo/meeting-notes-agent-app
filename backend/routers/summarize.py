import datetime
import json
import re
import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional
from database import get_db, SessionLocal
from models import NoteBlock, Transcription, Summary
from transcript_format import build_speaker_transcript
import lm_config

router = APIRouter(prefix="/api/notes", tags=["summarization"])


def _build_prompt(note: NoteBlock, transcription: Transcription, cfg: dict) -> tuple[str, str]:
    domain_name = note.domain.name if note.domain else "Project"

    global_prompt = cfg.get("global_system_prompt") or lm_config.DEFAULT_SYSTEM_PROMPT
    system_parts = [global_prompt]
    if note.project and note.project.custom_system_prompt:
        system_parts.append(note.project.custom_system_prompt)
    if note.project and note.project.knowledge_base:
        system_parts.append(f"Context about this project/team:\n{note.project.knowledge_base}")
    system_parts.append(
        'Respond ONLY with valid JSON in this exact shape — no prose outside the JSON block:\n'
        '{"summary": "...", "action_items": [{"task": "...", "owner": "...", "deadline": "..."}], "suggestions": "..."}\n\n'
        'Important rules for content quality:\n'
        '- "summary": follow the Task instructions precisely and in full. Be thorough — do not abbreviate or truncate.\n'
        '- "action_items": extract every concrete next step with owner and deadline when mentioned; return [] if none.\n'
        '- "suggestions": provide substantive recommendations relevant to the domain and task. Be specific.\n'
        '- Use Markdown inside string values (bold, bullets, headings) as instructed by the system prompt.'
    )

    template_prompt = (
        note.template.prompt_template
        if note.template and note.template.prompt_template
        else "Summarize the meeting: key decisions, action items, and blockers."
    )

    transcript = build_speaker_transcript(transcription.full_text, transcription.segments_json)
    max_chars = max(500, (cfg.get("max_tokens", 4096) - 1000) * 4)
    truncated = len(transcript) > max_chars
    if truncated:
        transcript = transcript[:max_chars]

    user_parts = [
        f"Domain: {domain_name}",
        f"Task: {template_prompt}",
        "",
        "Transcript:",
        transcript,
    ]
    if truncated:
        user_parts.append("\n[Transcript was truncated due to context length limits]")

    return "\n\n".join(system_parts), "\n".join(user_parts)


def _run_summarization(note_id: int) -> None:
    db = SessionLocal()
    try:
        note = db.get(NoteBlock, note_id)
        if not note:
            return

        transcription = db.query(Transcription).filter_by(note_block_id=note_id).first()
        if not transcription or not transcription.full_text:
            note.status = "error"
            db.commit()
            return

        note.status = "summarizing"
        note.updated_at = datetime.datetime.utcnow()
        db.commit()

        try:
            cfg = lm_config.load()
            base_url = cfg["base_url"].rstrip("/")
            model = cfg.get("model") or ""

            system_msg, user_msg = _build_prompt(note, transcription, cfg)

            max_response_tokens = cfg.get("max_response_tokens", 2048)
            payload: dict = {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.3,
                "max_tokens": max_response_tokens,
            }
            if model:
                payload["model"] = model

            with httpx.Client(timeout=180) as client:
                resp = client.post(f"{base_url}/chat/completions", json=payload)
                resp.raise_for_status()

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            llm_model = data.get("model", model)

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"```(?:json)?\s*([\s\S]+?)```", content)
                if match:
                    parsed = json.loads(match.group(1))
                else:
                    parsed = {"summary": content, "action_items": [], "suggestions": ""}

            def _str(v, default="") -> str:
                if isinstance(v, list):
                    return "\n".join(str(i) for i in v)
                return str(v) if v is not None else default

            def _items(v) -> list:
                if isinstance(v, list):
                    return [i if isinstance(i, dict) else {"task": str(i), "owner": "", "deadline": ""} for i in v]
                return []

            summary_text = _str(parsed.get("summary"))
            action_items_json = json.dumps(_items(parsed.get("action_items", [])))
            suggestions_text = _str(parsed.get("suggestions"))

            existing = db.query(Summary).filter_by(note_block_id=note_id).first()
            if existing:
                existing.summary_text = summary_text
                existing.action_items_json = action_items_json
                existing.suggestions_text = suggestions_text
                existing.llm_model_used = llm_model
                existing.generated_at = datetime.datetime.utcnow()
            else:
                db.add(Summary(
                    note_block_id=note_id,
                    summary_text=summary_text,
                    action_items_json=action_items_json,
                    suggestions_text=suggestions_text,
                    llm_model_used=llm_model,
                ))

            note.status = "done"
            note.updated_at = datetime.datetime.utcnow()
            db.commit()

        except Exception:
            db.rollback()
            note = db.get(NoteBlock, note_id)
            if note:
                note.status = "error"
                note.updated_at = datetime.datetime.utcnow()
                db.commit()

    finally:
        db.close()


@router.post("/{note_id}/summarize", status_code=202)
def start_summarization(
    note_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    if note.status == "summarizing":
        raise HTTPException(409, "Already summarizing")
    t = db.query(Transcription).filter_by(note_block_id=note_id).first()
    if not t or not t.full_text:
        raise HTTPException(400, "Note has no transcription")
    background_tasks.add_task(_run_summarization, note_id)
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
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    t = db.query(Transcription).filter_by(note_block_id=note_id).first()
    if not t:
        raise HTTPException(400, "No transcription available")
    cfg = lm_config.load()
    system_msg, user_msg = _build_prompt(note, t, cfg)
    return {"system": system_msg, "user": user_msg}
