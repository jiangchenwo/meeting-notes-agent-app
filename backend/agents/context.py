"""Shared per-run dependencies handed to every agent via pydantic-ai's deps mechanism."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NoteDeps:
    note_id: int
    domain_name: str
    template_name: str
    template_prompt: str
    project_system_prompt: str = ""
    project_knowledge_base: str = ""
    global_system_prompt: str = ""
    # Outputs of already-completed steps, keyed by agent name (plain dicts).
    prior_results: dict[str, Any] = field(default_factory=dict)
