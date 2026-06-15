import datetime
import os
import uuid
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session, joinedload
import aiofiles
from audio_utils import probe_duration_ms
from database import get_db
from models import NoteBlock
from schemas import NoteBlockResponse

router = APIRouter(prefix="/api", tags=["uploads"])

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(_ROOT, "uploads"))
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".webm"}


@router.post("/upload", response_model=NoteBlockResponse, status_code=201)
async def upload_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = file.filename or "untitled"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")

    size = 0
    try:
        async with aiofiles.open(dest_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(413, "File too large (max 2 GB)")
                await out.write(chunk)
    except HTTPException:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise

    display_name = os.path.splitext(filename)[0]
    now = datetime.datetime.utcnow()
    note = NoteBlock(
        display_name=display_name,
        audio_file_path=dest_path,
        audio_file_name=filename,
        audio_file_size=size,
        audio_duration_ms=probe_duration_ms(dest_path),
        status="pending",
        created_at=now,
        updated_at=now,
    )
    db.add(note)
    db.commit()
    db.refresh(note)

    return (
        db.query(NoteBlock)
        .options(joinedload(NoteBlock.project), joinedload(NoteBlock.domain), joinedload(NoteBlock.template))
        .filter(NoteBlock.id == note.id)
        .first()
    )
