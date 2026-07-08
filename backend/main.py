import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
import storage
from audio_utils import probe_duration_ms
from database import engine, SessionLocal
from models import Base, NoteBlock
from routers import domains, notes, projects, uploads
from routers import transcribe, settings, summarize, export, workflow
from seed import seed

# Create the storage subdirs and migrate any legacy flat files into them
# (db/, uploads/, config/) before the database is opened below.
storage.ensure_and_migrate()

Base.metadata.create_all(bind=engine)

# Add columns introduced after initial schema creation
with engine.connect() as _conn:
    for _stmt in [
        "ALTER TABLE domains ADD COLUMN color VARCHAR",
        "ALTER TABLE domains ADD COLUMN sort_order INTEGER DEFAULT 0",
        "ALTER TABLE note_blocks ADD COLUMN color VARCHAR",
        "ALTER TABLE note_blocks ADD COLUMN sort_order INTEGER DEFAULT 0",
        "ALTER TABLE note_blocks ADD COLUMN audio_duration_ms INTEGER",
        "ALTER TABLE templates ADD COLUMN workflow_config TEXT",
        "ALTER TABLE templates ADD COLUMN description TEXT DEFAULT ''",
        "ALTER TABLE summaries ADD COLUMN workflow_run_id INTEGER",
        "ALTER TABLE summaries ADD COLUMN confidence_score REAL",
        "ALTER TABLE summaries ADD COLUMN raw_sections_json TEXT",
        "ALTER TABLE projects ADD COLUMN color VARCHAR",
        "ALTER TABLE projects ADD COLUMN icon VARCHAR",
        "ALTER TABLE workflow_runs ADD COLUMN total_input_tokens INTEGER",
        "ALTER TABLE workflow_runs ADD COLUMN total_output_tokens INTEGER",
        "ALTER TABLE workflow_runs ADD COLUMN model_name VARCHAR",
        "ALTER TABLE workflow_runs ADD COLUMN trace_id VARCHAR",
        "ALTER TABLE workflow_step_results ADD COLUMN input_tokens INTEGER",
        "ALTER TABLE workflow_step_results ADD COLUMN output_tokens INTEGER",
        "ALTER TABLE workflow_step_results ADD COLUMN model_name VARCHAR",
    ]:
        try:
            _conn.execute(text(_stmt))
            _conn.commit()
        except Exception:
            pass  # column already exists

    # Backfill duration for existing notes from their transcriptions
    try:
        _conn.execute(text(
            "UPDATE note_blocks SET audio_duration_ms = ("
            "SELECT duration_ms FROM transcriptions "
            "WHERE transcriptions.note_block_id = note_blocks.id) "
            "WHERE audio_duration_ms IS NULL"
        ))
        _conn.commit()
    except Exception:
        pass  # transcriptions has no duration_ms column; duration is probed below

# Probe duration for any remaining notes whose audio file exists but has no duration
_db = SessionLocal()
try:
    for _note in _db.query(NoteBlock).filter(
        NoteBlock.audio_duration_ms.is_(None),
        NoteBlock.audio_file_path.isnot(None),
    ).all():
        if os.path.isfile(_note.audio_file_path):
            _dur = probe_duration_ms(_note.audio_file_path)
            if _dur is not None:
                _note.audio_duration_ms = _dur
    _db.commit()
finally:
    _db.close()

seed()

# Reset notes stuck in transient background-task states from a previous crashed run
with engine.connect() as _conn:
    _conn.execute(text("UPDATE note_blocks SET status='transcribed' WHERE status='summarizing'"))
    _conn.execute(text("UPDATE note_blocks SET status='pending' WHERE status='transcribing'"))
    _conn.execute(text(
        "UPDATE workflow_runs SET status='error', error_message='Process interrupted' "
        "WHERE status NOT IN ('done', 'error')"
    ))
    _conn.commit()

import telemetry

telemetry.configure_telemetry()

app = FastAPI(title="Meeting Notes Agent API", version="1.0.0")


@app.on_event("shutdown")
def _flush_telemetry():
    telemetry.shutdown_telemetry()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(uploads.router)
app.include_router(notes.router)
app.include_router(transcribe.router)
app.include_router(export.router)
app.include_router(projects.router)
app.include_router(domains.router)
app.include_router(settings.router)
app.include_router(summarize.router)
app.include_router(workflow.router)

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_UPLOADS_DIR = os.getenv("UPLOAD_DIR", os.path.join(_BACKEND_DIR, "uploads"))
os.makedirs(_UPLOADS_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=_UPLOADS_DIR), name="uploads")


@app.get("/api/health")
def health():
    return {"status": "ok"}
