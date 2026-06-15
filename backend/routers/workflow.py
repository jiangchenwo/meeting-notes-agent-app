import json
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import NoteBlock, Transcription, WorkflowRun, WorkflowStepResult
from agents import run_workflow

router = APIRouter(prefix="/api/notes", tags=["workflow"])


@router.post("/{note_id}/run-workflow", status_code=202)
def start_workflow(
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
        raise HTTPException(400, "Note has no transcription yet")
    background_tasks.add_task(run_workflow, note_id)
    return {"status": "started"}


@router.get("/{note_id}/workflow-run")
def get_workflow_run(note_id: int, db: Session = Depends(get_db)):
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    run = (
        db.query(WorkflowRun)
        .filter_by(note_block_id=note_id)
        .order_by(WorkflowRun.id.desc())
        .first()
    )
    if not run:
        return {"run": None}
    return {
        "run": {
            "id": run.id,
            "status": run.status,
            "current_step": run.current_step,
            "workflow_plan": json.loads(run.workflow_plan_json) if run.workflow_plan_json else None,
            "error_message": run.error_message,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        }
    }


@router.get("/{note_id}/workflow-run/steps")
def get_workflow_steps(note_id: int, db: Session = Depends(get_db)):
    note = db.get(NoteBlock, note_id)
    if not note:
        raise HTTPException(404, "Note not found")
    run = (
        db.query(WorkflowRun)
        .filter_by(note_block_id=note_id)
        .order_by(WorkflowRun.id.desc())
        .first()
    )
    if not run:
        return {"run_id": None, "steps": []}
    steps = (
        db.query(WorkflowStepResult)
        .filter_by(run_id=run.id)
        .order_by(WorkflowStepResult.id)
        .all()
    )
    return {
        "run_id": run.id,
        "steps": [
            {
                "id": s.id,
                "step_name": s.step_name,
                "status": s.status,
                "duration_ms": s.duration_ms,
                "critique_score": s.critique_score,
                "attempt": s.attempt,
                "result": json.loads(s.result_json) if s.result_json else None,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in steps
        ],
    }
