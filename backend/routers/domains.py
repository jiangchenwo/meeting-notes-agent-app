import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import Domain, Template
from schemas import (
    DomainCreate, DomainUpdate, DomainResponse,
    TemplateCreate, TemplateUpdate, TemplateResponse,
)

router = APIRouter(prefix="/api", tags=["domains"])


# ── Domains ──────────────────────────────────────────────────────────────────

@router.get("/domains", response_model=list[DomainResponse])
def list_domains(db: Session = Depends(get_db)):
    return db.query(Domain).order_by(Domain.sort_order, Domain.name).all()


@router.post("/domains", response_model=DomainResponse, status_code=201)
def create_domain(body: DomainCreate, db: Session = Depends(get_db)):
    domain = Domain(name=body.name, description=body.description, is_builtin=False)
    db.add(domain)
    db.commit()
    db.refresh(domain)
    return domain


@router.patch("/domains/{domain_id}", response_model=DomainResponse)
def update_domain(domain_id: int, body: DomainUpdate, db: Session = Depends(get_db)):
    d = db.get(Domain, domain_id)
    if not d:
        raise HTTPException(404, "Domain not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(d, field, value)
    db.commit()
    db.refresh(d)
    return d


@router.delete("/domains/{domain_id}")
def delete_domain(domain_id: int, db: Session = Depends(get_db)):
    d = db.get(Domain, domain_id)
    if not d:
        raise HTTPException(404, "Domain not found")
    db.delete(d)
    db.commit()
    return {"ok": True}


# ── Templates ─────────────────────────────────────────────────────────────────

@router.get("/templates", response_model=list[TemplateResponse])
def list_templates(domain_id: Optional[int] = None, db: Session = Depends(get_db)):
    q = db.query(Template)
    if domain_id is not None:
        q = q.filter(Template.domain_id == domain_id)
    return q.order_by(Template.name).all()


@router.get("/templates/{template_id}", response_model=TemplateResponse)
def get_template(template_id: int, db: Session = Depends(get_db)):
    t = db.get(Template, template_id)
    if not t:
        raise HTTPException(404, "Template not found")
    return t


@router.post("/templates", response_model=TemplateResponse, status_code=201)
def create_template(body: TemplateCreate, db: Session = Depends(get_db)):
    t = Template(
        name=body.name,
        description=body.description,
        domain_id=body.domain_id,
        prompt_template=body.prompt_template,
        output_sections=json.dumps(body.output_sections),
        is_builtin=False,
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@router.patch("/templates/{template_id}", response_model=TemplateResponse)
def update_template(template_id: int, body: TemplateUpdate, db: Session = Depends(get_db)):
    t = db.get(Template, template_id)
    if not t:
        raise HTTPException(404, "Template not found")
    data = body.model_dump(exclude_unset=True)
    if "output_sections" in data:
        data["output_sections"] = json.dumps(data["output_sections"])
    for field, value in data.items():
        setattr(t, field, value)
    db.commit()
    db.refresh(t)
    return t


@router.delete("/templates/{template_id}")
def delete_template(template_id: int, db: Session = Depends(get_db)):
    t = db.get(Template, template_id)
    if not t:
        raise HTTPException(404, "Template not found")
    db.delete(t)
    db.commit()
    return {"ok": True}
