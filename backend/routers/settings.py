import json
from datetime import datetime, timezone
from typing import Optional
import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import get_db
from models import Domain, Template, Project
from seed import sync_builtin_domains_and_templates
import lm_config
import asr_config
import telemetry
import telemetry_config

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
    output_mode: Optional[str] = None


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
    if payload.output_mode is not None:
        if payload.output_mode not in ("native", "prompted"):
            raise HTTPException(422, "output_mode must be 'native' or 'prompted'")
        cfg["output_mode"] = payload.output_mode
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


class TelemetryConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    endpoint: Optional[str] = None
    capture_content: Optional[bool] = None


@router.get("/telemetry")
def get_telemetry_config():
    return telemetry_config.load()


@router.put("/telemetry")
def update_telemetry_config(payload: TelemetryConfigUpdate):
    cfg = telemetry_config.load()
    if payload.enabled is not None:
        cfg["enabled"] = payload.enabled
    if payload.endpoint is not None:
        cfg["endpoint"] = payload.endpoint.strip()
    if payload.capture_content is not None:
        cfg["capture_content"] = payload.capture_content
    telemetry_config.save(cfg)
    # Re-apply immediately — no backend restart needed.
    active = telemetry.configure_telemetry()
    return {**cfg, "active": active}


class ASRConfigUpdate(BaseModel):
    base_url: Optional[str] = None


@router.get("/asr")
def get_asr_config():
    return asr_config.load()


@router.put("/asr")
def update_asr_config(payload: ASRConfigUpdate):
    cfg = asr_config.load()
    if payload.base_url is not None:
        cfg["base_url"] = payload.base_url.strip()
    asr_config.save(cfg)
    return cfg


@router.get("/asr/status")
def asr_status():
    configured_url = asr_config.load()["base_url"]
    base_url = configured_url.rstrip("/")
    try:
        with httpx.Client(timeout=3) as client:
            resp = client.get(f"{base_url}/health")
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "base_url": configured_url,
                    "connected": True,
                    "models_loaded": bool(data.get("models_loaded", False)),
                }
    except Exception:
        pass
    return {"base_url": configured_url, "connected": False, "models_loaded": False}
