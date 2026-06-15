import datetime
import json
from typing import Optional
from pydantic import BaseModel, field_validator


class DomainResponse(BaseModel):
    id: int
    name: str
    description: str
    is_builtin: bool
    color: Optional[str] = None
    sort_order: int = 0
    model_config = {"from_attributes": True}


class DomainCreate(BaseModel):
    name: str
    description: str = ""


class DomainUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    sort_order: Optional[int] = None


class TemplateResponse(BaseModel):
    id: int
    name: str
    description: str = ""
    domain_id: Optional[int]
    prompt_template: str
    output_sections: list[str]
    is_builtin: bool
    model_config = {"from_attributes": True}

    @field_validator("output_sections", mode="before")
    @classmethod
    def parse_sections(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return ["summary", "action_items"]
        return v or ["summary", "action_items"]


class TemplateCreate(BaseModel):
    name: str
    description: str = ""
    domain_id: Optional[int] = None
    prompt_template: str = ""
    output_sections: list[str] = ["summary", "action_items"]


class TemplateUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    domain_id: Optional[int] = None
    prompt_template: Optional[str] = None
    output_sections: Optional[list[str]] = None


class NoteBlockResponse(BaseModel):
    id: int
    display_name: str
    audio_file_name: Optional[str]
    audio_file_size: Optional[int]
    audio_url: Optional[str]
    project_id: Optional[int]
    project_name: Optional[str]
    project_color: Optional[str] = None
    audio_duration_ms: Optional[int] = None
    domain_id: Optional[int]
    domain_name: Optional[str]
    template_id: Optional[int]
    template_name: Optional[str]
    status: str
    color: Optional[str] = None
    sort_order: int = 0
    created_at: datetime.datetime
    updated_at: datetime.datetime
    model_config = {"from_attributes": True}


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    note_id: int
    full_text: Optional[str]
    segments: list[TranscriptionSegment]
    model_used: Optional[str]
    language: Optional[str]


class NoteBlockUpdate(BaseModel):
    display_name: Optional[str] = None
    project_id: Optional[int] = None
    domain_id: Optional[int] = None
    template_id: Optional[int] = None
    color: Optional[str] = None
    sort_order: Optional[int] = None


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: str
    custom_system_prompt: str
    knowledge_base: str
    color: Optional[str] = None
    icon: Optional[str] = None
    note_count: int
    top_domains: list[str]
    total_size: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    model_config = {"from_attributes": True}


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    custom_system_prompt: Optional[str] = None
    knowledge_base: Optional[str] = None
    color: Optional[str] = None
    icon: Optional[str] = None
