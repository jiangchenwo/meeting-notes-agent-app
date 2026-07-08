import datetime
import os
from typing import Optional
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship, backref
from database import Base


class Domain(Base):
    __tablename__ = "domains"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, default="")
    is_builtin = Column(Boolean, default=True)
    color = Column(String, nullable=True)
    sort_order = Column(Integer, default=0)


class Template(Base):
    __tablename__ = "templates"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    domain_id = Column(Integer, ForeignKey("domains.id"), nullable=True)
    prompt_template = Column(Text, default="")
    output_sections = Column(Text, default='["summary","action_items"]')
    is_builtin = Column(Boolean, default=True)
    workflow_config = Column(Text, nullable=True)
    domain = relationship("Domain", backref="templates")


class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    custom_system_prompt = Column(Text, default="")
    knowledge_base = Column(Text, default="")
    color = Column(String, nullable=True)
    icon = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

    @property
    def note_count(self) -> int:
        return len(self.note_blocks)

    @property
    def top_domains(self) -> list[str]:
        from collections import Counter
        names = [nb.domain.name for nb in self.note_blocks if nb.domain]
        return [name for name, _ in Counter(names).most_common(3)]

    @property
    def total_size(self) -> int:
        return sum(nb.audio_file_size or 0 for nb in self.note_blocks)


class ProjectSpeaker(Base):
    __tablename__ = "project_speakers"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    color = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    project = relationship(
        "Project",
        backref=backref("speakers", cascade="all, delete-orphan"),
    )


class NoteBlock(Base):
    __tablename__ = "note_blocks"
    id = Column(Integer, primary_key=True, index=True)
    display_name = Column(String, nullable=False)
    audio_file_path = Column(String, nullable=True)
    audio_file_name = Column(String, nullable=True)
    audio_file_size = Column(Integer, nullable=True)
    audio_duration_ms = Column(Integer, nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    domain_id = Column(Integer, ForeignKey("domains.id"), nullable=True)
    template_id = Column(Integer, ForeignKey("templates.id"), nullable=True)
    status = Column(String, default="pending")
    color = Column(String, nullable=True)
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

    project = relationship("Project", backref="note_blocks")
    domain = relationship("Domain")
    template = relationship("Template")

    @property
    def project_name(self) -> Optional[str]:
        return self.project.name if self.project else None

    @property
    def project_color(self) -> Optional[str]:
        return self.project.color if self.project else None

    @property
    def domain_name(self) -> Optional[str]:
        return self.domain.name if self.domain else None

    @property
    def template_name(self) -> Optional[str]:
        return self.template.name if self.template else None

    @property
    def audio_url(self) -> Optional[str]:
        if self.audio_file_path:
            return f"/uploads/{os.path.basename(self.audio_file_path)}"
        return None


class Transcription(Base):
    __tablename__ = "transcriptions"
    id = Column(Integer, primary_key=True, index=True)
    note_block_id = Column(Integer, ForeignKey("note_blocks.id"), unique=True)
    full_text = Column(Text, default="")
    segments_json = Column(Text, default="[]")
    language = Column(String, nullable=True)
    model_used = Column(String, nullable=True)
    diarized = Column(Boolean, default=False)
    note_block = relationship("NoteBlock", backref="transcription")


class Summary(Base):
    __tablename__ = "summaries"
    id = Column(Integer, primary_key=True, index=True)
    note_block_id = Column(Integer, ForeignKey("note_blocks.id"), unique=True)
    summary_text = Column(Text, default="")
    action_items_json = Column(Text, default="[]")
    suggestions_text = Column(Text, default="")
    llm_model_used = Column(String, nullable=True)
    generated_at = Column(DateTime, default=datetime.datetime.utcnow)
    workflow_run_id = Column(Integer, ForeignKey("workflow_runs.id"), nullable=True)
    confidence_score = Column(Float, nullable=True)
    raw_sections_json = Column(Text, nullable=True)
    note_block = relationship("NoteBlock", backref="summary")


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"
    id = Column(Integer, primary_key=True, index=True)
    note_block_id = Column(Integer, ForeignKey("note_blocks.id"))
    status = Column(String, default="queued")
    current_step = Column(String, nullable=True)
    workflow_plan_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    total_input_tokens = Column(Integer, nullable=True)
    total_output_tokens = Column(Integer, nullable=True)
    model_name = Column(String, nullable=True)
    trace_id = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    note_block = relationship("NoteBlock", backref="workflow_runs")
    steps = relationship("WorkflowStepResult", backref="run", order_by="WorkflowStepResult.id")


class WorkflowStepResult(Base):
    __tablename__ = "workflow_step_results"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("workflow_runs.id"))
    step_name = Column(String)
    status = Column(String, default="pending")
    duration_ms = Column(Integer, nullable=True)
    result_json = Column(Text, nullable=True)
    critique_score = Column(Float, nullable=True)
    attempt = Column(Integer, default=1)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    model_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
