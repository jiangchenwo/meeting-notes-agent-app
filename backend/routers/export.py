import json
import re
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session
from database import get_db
from models import NoteBlock, Transcription, Summary

router = APIRouter(prefix="/api/notes", tags=["export"])


def _strip_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    return text


def _build_markdown(note: NoteBlock, transcription, summary) -> str:
    lines = [f"# {note.display_name}", ""]
    lines.append(f"**Date:** {note.created_at.strftime('%Y-%m-%d %H:%M')}")
    if note.project_name:
        lines.append(f"**Project:** {note.project_name}")
    if note.domain_name:
        lines.append(f"**Domain:** {note.domain_name}")
    if note.template_name:
        lines.append(f"**Template:** {note.template_name}")
    lines.append(f"**Status:** {note.status}")
    lines.append("")

    if summary and summary.summary_text:
        lines += ["## Summary", "", summary.summary_text, ""]

    if summary and summary.action_items_json:
        try:
            items = json.loads(summary.action_items_json)
        except Exception:
            items = []
        if items:
            lines += ["## Action Items", ""]
            for item in items:
                task = item.get("task", "")
                meta = " · ".join(filter(None, [item.get("owner", ""), item.get("deadline", "")]))
                lines.append(f"- [ ] {task}" + (f" _{meta}_" if meta else ""))
            lines.append("")

    if summary and summary.suggestions_text:
        lines += ["## Suggestions", "", summary.suggestions_text, ""]

    if transcription and transcription.full_text:
        lines += ["## Transcript", "", transcription.full_text, ""]

    return "\n".join(lines)


def _build_text(note: NoteBlock, transcription, summary) -> str:
    title = note.display_name
    lines = [title, "=" * len(title), ""]
    lines.append(f"Date: {note.created_at.strftime('%Y-%m-%d %H:%M')}")
    if note.project_name:
        lines.append(f"Project: {note.project_name}")
    if note.domain_name:
        lines.append(f"Domain: {note.domain_name}")
    lines.append("")

    if summary and summary.summary_text:
        lines += ["SUMMARY", "-" * 7, _strip_markdown(summary.summary_text), ""]

    if summary and summary.action_items_json:
        try:
            items = json.loads(summary.action_items_json)
        except Exception:
            items = []
        if items:
            lines += ["ACTION ITEMS", "-" * 12, ""]
            for item in items:
                task = item.get("task", "")
                meta = ", ".join(filter(None, [item.get("owner", ""), item.get("deadline", "")]))
                lines.append(f"[ ] {task}" + (f" ({meta})" if meta else ""))
            lines.append("")

    if summary and summary.suggestions_text:
        lines += ["SUGGESTIONS", "-" * 11, _strip_markdown(summary.suggestions_text), ""]

    if transcription and transcription.full_text:
        lines += ["TRANSCRIPT", "-" * 10, transcription.full_text, ""]

    return "\n".join(lines)


@router.get("/{note_id}/export")
def export_note(
    note_id: int,
    format: str = Query("markdown", pattern="^(markdown|text)$"),
    db: Session = Depends(get_db),
):
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")

    transcription = db.query(Transcription).filter_by(note_block_id=note_id).first()
    summary = db.query(Summary).filter_by(note_block_id=note_id).first()

    safe_name = "".join(c for c in note.display_name if c.isalnum() or c in " -_").strip()[:50] or "note"

    if format == "markdown":
        content = _build_markdown(note, transcription, summary)
        filename = f"{safe_name}.md"
        media_type = "text/markdown"
    else:
        content = _build_text(note, transcription, summary)
        filename = f"{safe_name}.txt"
        media_type = "text/plain"

    return Response(
        content=content.encode("utf-8"),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
