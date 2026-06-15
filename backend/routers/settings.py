import json
from datetime import datetime, timezone
from typing import Optional
import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import get_db
from models import Domain, Template, Project
from seed import sync_builtin_domains_and_templates
import lm_config
import whisper_config

router = APIRouter(prefix="/api/settings", tags=["settings"])

@router.get("/backup")
def backup(db: Session = Depends(get_db)):
    domains = db.query(Domain).order_by(Domain.sort_order, Domain.name).all()
    templates = db.query(Template).order_by(Template.name).all()
    projects = db.query(Project).order_by(Project.name).all()
    domain_id_to_name = {d.id: d.name for d in domains}

    data = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "domains": [
            {"name": d.name, "description": d.description, "color": d.color, "sort_order": d.sort_order}
            for d in domains
        ],
        "templates": [
            {
                "name": t.name,
                "domain_name": domain_id_to_name.get(t.domain_id),
                "prompt_template": t.prompt_template,
                "output_sections": json.loads(t.output_sections) if t.output_sections else [],
            }
            for t in templates
        ],
        "projects": [
            {
                "name": p.name,
                "description": p.description,
                "custom_system_prompt": p.custom_system_prompt,
                "knowledge_base": p.knowledge_base,
            }
            for p in projects
        ],
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": f"attachment; filename=meeting-notes-backup-{stamp}.json"},
    )


@router.post("/restore-defaults")
def restore_defaults(db: Session = Depends(get_db)):
    restored = sync_builtin_domains_and_templates(db)
    db.commit()
    return {"ok": True, "restored": restored}


class LMConfigUpdate(BaseModel):
    base_url: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    max_response_tokens: Optional[int] = None
    global_system_prompt: Optional[str] = None


@router.get("/llm")
def get_llm_config():
    return lm_config.load()


@router.put("/llm")
def update_llm_config(payload: LMConfigUpdate):
    cfg = lm_config.load()
    if payload.base_url is not None:
        cfg["base_url"] = payload.base_url.strip()
    if payload.model is not None:
        cfg["model"] = payload.model.strip()
    if payload.max_tokens is not None:
        cfg["max_tokens"] = max(512, min(128000, payload.max_tokens))
    if payload.max_response_tokens is not None:
        cfg["max_response_tokens"] = max(256, min(32000, payload.max_response_tokens))
    if payload.global_system_prompt is not None:
        cfg["global_system_prompt"] = payload.global_system_prompt
    lm_config.save(cfg)
    return cfg


@router.get("/lm-studio/status")
def lm_studio_status():
    cfg = lm_config.load()
    base_url = cfg["base_url"].rstrip("/")
    try:
        with httpx.Client(timeout=3) as client:
            resp = client.get(f"{base_url}/models")
            if resp.status_code == 200:
                models = [m.get("id", "") for m in resp.json().get("data", [])]
                return {"connected": True, "models": models}
    except Exception:
        pass
    return {"connected": False, "models": []}


class WhisperConfigUpdate(BaseModel):
    binary_path: Optional[str] = None
    model: Optional[str] = None
    model_path: Optional[str] = None


@router.get("/whisper")
def get_whisper_config():
    cfg = whisper_config.load()
    return {**cfg, "available_models": whisper_config.WHISPER_MODELS}


@router.put("/whisper")
def update_whisper_config(payload: WhisperConfigUpdate):
    cfg = whisper_config.load()
    if payload.binary_path is not None:
        cfg["binary_path"] = payload.binary_path.strip()
    if payload.model is not None:
        cfg["model"] = payload.model.strip()
    if payload.model_path is not None:
        cfg["model_path"] = payload.model_path.strip()
    whisper_config.save(cfg)
    return {**cfg, "available_models": whisper_config.WHISPER_MODELS}
