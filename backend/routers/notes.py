import datetime
import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, or_
from sqlalchemy.orm import Session, joinedload
from database import get_db
from models import NoteBlock, Transcription, Summary
from schemas import NoteBlockResponse, NoteBlockUpdate

router = APIRouter(prefix="/api/notes", tags=["notes"])


def _load(db: Session):
    return db.query(NoteBlock).options(
        joinedload(NoteBlock.project),
        joinedload(NoteBlock.domain),
        joinedload(NoteBlock.template),
    )


@router.get("", response_model=list[NoteBlockResponse])
def list_notes(project_id: Optional[int] = None, db: Session = Depends(get_db)):
    q = _load(db)
    if project_id is not None:
        q = q.filter(NoteBlock.project_id == project_id)
    return q.order_by(NoteBlock.sort_order.asc(), NoteBlock.created_at.desc()).all()


@router.get("/search", response_model=list[NoteBlockResponse])
def search_notes(
    q: str = "",
    project_id: Optional[int] = None,
    domain_id: Optional[int] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = _load(db)

    if project_id is not None:
        query = query.filter(NoteBlock.project_id == project_id)
    if domain_id is not None:
        query = query.filter(NoteBlock.domain_id == domain_id)
    if status:
        query = query.filter(NoteBlock.status == status)

    if q.strip():
        pattern = f"%{q.lower()}%"
        query = (
            query
            .outerjoin(Transcription, Transcription.note_block_id == NoteBlock.id)
            .outerjoin(Summary, Summary.note_block_id == NoteBlock.id)
            .filter(
                or_(
                    func.lower(NoteBlock.display_name).like(pattern),
                    func.lower(Transcription.full_text).like(pattern),
                    func.lower(Summary.summary_text).like(pattern),
                )
            )
        )

    return query.order_by(NoteBlock.sort_order.asc(), NoteBlock.created_at.desc()).all()


class BulkDeleteBody(BaseModel):
    ids: list[int]


class BulkUpdateBody(BaseModel):
    ids: list[int]
    project_id: Optional[int] = None


@router.post("/bulk-delete")
def bulk_delete(body: BulkDeleteBody, db: Session = Depends(get_db)):
    notes = db.query(NoteBlock).filter(NoteBlock.id.in_(body.ids)).all()
    for note in notes:
        if note.audio_file_path:
            try:
                os.remove(note.audio_file_path)
            except FileNotFoundError:
                pass
        db.delete(note)
    db.commit()
    return {"ok": True, "deleted": len(notes)}


@router.patch("/bulk-update")
def bulk_update(body: BulkUpdateBody, db: Session = Depends(get_db)):
    notes = db.query(NoteBlock).filter(NoteBlock.id.in_(body.ids)).all()
    for note in notes:
        if "project_id" in body.model_dump(exclude_unset=True):
            note.project_id = body.project_id
        note.updated_at = datetime.datetime.utcnow()
    db.commit()
    return {"ok": True, "updated": len(notes)}


@router.get("/{note_id}", response_model=NoteBlockResponse)
def get_note(note_id: int, db: Session = Depends(get_db)):
    note = _load(db).filter(NoteBlock.id == note_id).first()
    if not note:
        raise HTTPException(404, "Note not found")
    return note


@router.patch("/{note_id}", response_model=NoteBlockResponse)
def update_note(note_id: int, body: NoteBlockUpdate, db: Session = Depends(get_db)):
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(note, field, value)
    note.updated_at = datetime.datetime.utcnow()
    db.commit()
    return _load(db).filter(NoteBlock.id == note_id).first()


@router.delete("/{note_id}")
def delete_note(note_id: int, db: Session = Depends(get_db)):
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    if note.audio_file_path:
        try:
            os.remove(note.audio_file_path)
        except FileNotFoundError:
            pass
    db.delete(note)
    db.commit()
    return {"ok": True}
