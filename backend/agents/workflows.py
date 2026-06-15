import json


DOMAIN_WORKFLOWS: dict[str, dict] = {
    # Priority 0: baseline for any conversation that doesn't fit a specific domain.
    "General": {
        "steps": ["Summarizer", "ActionItemExtractor"],
        "critique_steps": ["Summarizer"],
        "critique_threshold": 8,
        "max_retries": 2,
    },
    "Education": {
        "steps": ["Summarizer", "LectureAgent", "ActionItemExtractor"],
        "critique_steps": ["Summarizer"],
        "critique_threshold": 8,
        "max_retries": 2,
    },
    "Healthcare": {
        "steps": ["Summarizer", "ActionItemExtractor"],
        "critique_steps": ["Summarizer"],
        "critique_threshold": 8,
        "max_retries": 2,
    },
    "Interview": {
        "steps": ["Summarizer", "InterviewAgent"],
        "critique_steps": ["InterviewAgent"],
        "critique_threshold": 8,
        "max_retries": 2,
    },
    "Project": {
        "steps": ["Summarizer", "ActionItemExtractor", "DecisionLogger"],
        "critique_steps": ["Summarizer"],
        "critique_threshold": 8,
        "max_retries": 2,
    },
}

# Mirrors General — used as fallback for domains not in DOMAIN_WORKFLOWS.
_DEFAULT_WORKFLOW: dict = {
    "steps": ["Summarizer", "ActionItemExtractor", "DecisionLogger"],
    "critique_steps": ["Summarizer"],
    "critique_threshold": 8,
    "max_retries": 2,
}


def select_workflow(domain_name: str, template_workflow_config: str | None) -> dict:
    if template_workflow_config:
        try:
            override = json.loads(template_workflow_config)
            if "steps" in override:
                return {**_DEFAULT_WORKFLOW, **override}
        except Exception:
            pass
    return DOMAIN_WORKFLOWS.get(domain_name, _DEFAULT_WORKFLOW)
