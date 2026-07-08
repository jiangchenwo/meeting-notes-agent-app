"""WorkflowSpec validation + select_workflow parity with the legacy engine."""
import json

import pytest
from pydantic import ValidationError

from agents.workflow_spec import (
    DEFAULT_WORKFLOW,
    DOMAIN_WORKFLOWS,
    WorkflowSpec,
    select_workflow,
)


def test_legacy_string_steps_are_coerced():
    spec = WorkflowSpec.model_validate({"steps": ["Summarizer", "ActionItemExtractor"]})
    assert spec.step_names == ["Summarizer", "ActionItemExtractor"]
    assert spec.steps[0].prompt_override is None


def test_defaults_match_legacy_default_workflow():
    spec = WorkflowSpec.model_validate({"steps": ["Summarizer"]})
    assert spec.critique_threshold == 8.0
    assert spec.max_retries == 2


def test_unknown_agent_rejected():
    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate({"steps": ["Summarizer", "NotAnAgent"]})


def test_critique_step_must_be_in_steps():
    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate(
            {"steps": ["Summarizer"], "critique_steps": ["DecisionLogger"]}
        )


def test_bounds_enforced():
    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate({"steps": ["Summarizer"], "max_retries": 4})
    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate({"steps": ["Summarizer"], "critique_threshold": 11})
    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate({"steps": []})


def test_all_builtin_domains_have_workflows():
    assert set(DOMAIN_WORKFLOWS) == {"General", "Education", "Healthcare", "Interview", "Project"}
    for spec in DOMAIN_WORKFLOWS.values():
        assert spec.step_names[0] == "Summarizer"
        assert spec.critique_steps  # every built-in plan quality-checks something


def test_domain_workflow_plans():
    plans = {
        name: (spec.step_names, spec.critique_steps)
        for name, spec in DOMAIN_WORKFLOWS.items()
    }
    assert plans == {
        "General": (
            ["Summarizer", "ActionItemExtractor", "DecisionLogger"], ["Summarizer"],
        ),
        "Education": (
            ["Summarizer", "LectureAgent", "ActionItemExtractor"],
            ["Summarizer", "LectureAgent"],
        ),
        "Healthcare": (
            ["Summarizer", "ActionItemExtractor"],
            ["Summarizer", "ActionItemExtractor"],
        ),
        "Interview": (
            ["Summarizer", "InterviewAgent"], ["Summarizer", "InterviewAgent"],
        ),
        "Project": (
            ["Summarizer", "ActionItemExtractor", "DecisionLogger"], ["Summarizer"],
        ),
    }


def test_select_workflow_domain_lookup_and_fallback():
    assert select_workflow("Interview", None) is DOMAIN_WORKFLOWS["Interview"]
    assert select_workflow("Nonexistent Domain", None) is DEFAULT_WORKFLOW


def test_select_workflow_valid_override_merges_defaults():
    # Legacy semantics: override fields win, unspecified fields come from the default.
    override = json.dumps({"steps": ["Summarizer"], "critique_steps": [], "max_retries": 0})
    spec = select_workflow("General", override)
    assert spec.step_names == ["Summarizer"]
    assert spec.max_retries == 0
    assert spec.critique_threshold == 8.0  # inherited from default


def test_select_workflow_lenient_on_bad_config():
    # Invalid JSON, missing steps, and invalid specs all fall back (legacy read semantics).
    assert select_workflow("General", "{not json") is DOMAIN_WORKFLOWS["General"]
    assert select_workflow("General", json.dumps({"critique_threshold": 5})) is DOMAIN_WORKFLOWS["General"]
    assert select_workflow("General", json.dumps({"steps": ["Bogus"]})) is DOMAIN_WORKFLOWS["General"]
