import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload
from database import get_db
from models import Project, NoteBlock, ProjectSpeaker
from schemas import (
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    ProjectSpeakerCreate,
    ProjectSpeakerResponse,
    ProjectSpeakerUpdate,
)

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _load(db: Session):
    return db.query(Project).options(
        selectinload(Project.note_blocks).selectinload(NoteBlock.domain)
    )


@router.get("", response_model=list[ProjectResponse])
def list_projects(db: Session = Depends(get_db)):
    return _load(db).order_by(Project.name).all()


@router.post("", response_model=ProjectResponse, status_code=201)
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    project = Project(**body.model_dump())
    db.add(project)
    db.commit()
    db.refresh(project)
    db.query(Project).options(selectinload(Project.note_blocks)).filter(Project.id == project.id).first()
    return project


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int, db: Session = Depends(get_db)):
    p = _load(db).filter(Project.id == project_id).first()
    if not p:
        raise HTTPException(404, "Project not found")
    return p


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(project_id: int, body: ProjectUpdate, db: Session = Depends(get_db)):
    p = db.get(Project, project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(p, field, value)
    p.updated_at = datetime.datetime.utcnow()
    db.commit()
    return _load(db).filter(Project.id == project_id).first()


@router.delete("/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    p = db.get(Project, project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    db.delete(p)
    db.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Per-project speaker roster — shared vocabulary for consistent speaker labels
# ---------------------------------------------------------------------------
def _require_project(project_id: int, db: Session) -> Project:
    p = db.get(Project, project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    return p


@router.get("/{project_id}/speakers", response_model=list[ProjectSpeakerResponse])
def list_project_speakers(project_id: int, db: Session = Depends(get_db)):
    _require_project(project_id, db)
    return (
        db.query(ProjectSpeaker)
        .filter(ProjectSpeaker.project_id == project_id)
        .order_by(ProjectSpeaker.name)
        .all()
    )


@router.post("/{project_id}/speakers", response_model=ProjectSpeakerResponse, status_code=201)
def create_project_speaker(
    project_id: int, body: ProjectSpeakerCreate, db: Session = Depends(get_db)
):
    _require_project(project_id, db)
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Speaker name cannot be empty")
    existing = (
        db.query(ProjectSpeaker)
        .filter(
            ProjectSpeaker.project_id == project_id,
            func.lower(ProjectSpeaker.name) == name.lower(),
        )
        .first()
    )
    if existing:
        return existing
    speaker = ProjectSpeaker(project_id=project_id, name=name, color=body.color)
    db.add(speaker)
    db.commit()
    db.refresh(speaker)
    return speaker


@router.patch("/{project_id}/speakers/{speaker_id}", response_model=ProjectSpeakerResponse)
def update_project_speaker(
    project_id: int,
    speaker_id: int,
    body: ProjectSpeakerUpdate,
    db: Session = Depends(get_db),
):
    speaker = db.get(ProjectSpeaker, speaker_id)
    if not speaker or speaker.project_id != project_id:
        raise HTTPException(404, "Speaker not found")
    data = body.model_dump(exclude_unset=True)
    if "name" in data:
        name = (data["name"] or "").strip()
        if not name:
            raise HTTPException(422, "Speaker name cannot be empty")
        data["name"] = name
    for field, value in data.items():
        setattr(speaker, field, value)
    db.commit()
    db.refresh(speaker)
    return speaker


@router.delete("/{project_id}/speakers/{speaker_id}")
def delete_project_speaker(project_id: int, speaker_id: int, db: Session = Depends(get_db)):
    speaker = db.get(ProjectSpeaker, speaker_id)
    if not speaker or speaker.project_id != project_id:
        raise HTTPException(404, "Speaker not found")
    db.delete(speaker)
    db.commit()
    return {"ok": True}
