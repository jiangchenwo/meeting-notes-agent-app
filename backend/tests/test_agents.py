"""Agent definition tests: prompt composition and structured output via TestModel."""
from pydantic_ai import capture_run_messages
from pydantic_ai.models.test import TestModel

from agents.context import NoteDeps
from agents.definitions import (
    AGENT_REGISTRY,
    INSTRUCTION_BUILDERS,
    build_critic_user_prompt,
    build_user_prompt,
)
from agents.outputs import AGENT_OUTPUT_TYPES, SummaryOutput


def make_deps(**overrides) -> NoteDeps:
    defaults = dict(
        note_id=1,
        domain_name="Project",
        template_name="Default",
        template_prompt="Summarize the meeting.",
        project_system_prompt="",
        project_knowledge_base="",
        global_system_prompt="",
    )
    defaults.update(overrides)
    return NoteDeps(**defaults)


def test_registry_matches_output_types():
    assert set(AGENT_REGISTRY) == set(AGENT_OUTPUT_TYPES) == set(INSTRUCTION_BUILDERS)


def test_all_agents_produce_valid_structured_output():
    deps = make_deps()
    for name, agent in AGENT_REGISTRY.items():
        result = agent.run_sync(
            build_user_prompt(name, deps, "Alice: hello"),
            deps=deps,
            model=TestModel(),
            output_type=AGENT_OUTPUT_TYPES[name],
        )
        assert isinstance(result.output, AGENT_OUTPUT_TYPES[name])
        # model_dump keys are the persisted result_json contract
        assert set(result.output.model_dump()) == set(
            AGENT_OUTPUT_TYPES[name].model_fields
        ) | ({"score"} if hasattr(result.output, "score") else set())


def test_summarizer_instructions_include_project_and_global_prompts():
    deps = make_deps(
        project_system_prompt="Always mention the sprint number.",
        global_system_prompt="You are a professional meeting notes assistant.",
    )
    with capture_run_messages() as messages:
        AGENT_REGISTRY["Summarizer"].run_sync(
            build_user_prompt("Summarizer", deps, "Alice: hi"),
            deps=deps,
            model=TestModel(),
            output_type=SummaryOutput,
        )
    request = messages[0]
    instructions = request.instructions
    assert "Always mention the sprint number." in instructions
    assert "You are a professional meeting notes assistant." in instructions
    assert "focused meeting summarizer" in instructions


def test_summarizer_user_prompt_includes_kb_snippet():
    deps = make_deps(
        project_knowledge_base="The project codename is Falcon.\nUnrelated line about lunch.",
        template_prompt="Summarize the project meeting",
    )
    user = build_user_prompt("Summarizer", deps, "Bob: status update")
    assert "Relevant project context:" in user
    assert "codename is Falcon" in user
    assert "Transcript:\nBob: status update" in user


def test_summarizer_retry_prompt_carries_previous_attempt_and_notes():
    deps = make_deps()
    user = build_user_prompt(
        "Summarizer", deps, "Bob: hi",
        previous_attempt="Old draft " * 1000,
        quality_notes=["Add the budget decision"],
    )
    assert "This is a revision. Previous draft" in user
    assert "Quality review notes to address:\n- Add the budget decision" in user
    # previous attempt is capped at REVISION_DRAFT_CHARS
    from agents.definitions import REVISION_DRAFT_CHARS

    assert len(user) < REVISION_DRAFT_CHARS + 500


def test_action_extractor_retry_prompt_carries_previous_attempt():
    deps = make_deps()
    user = build_user_prompt(
        "ActionItemExtractor", deps, "Bob: hi",
        previous_attempt='[{"task": "old"}]',
        quality_notes=["Missing the report deadline"],
    )
    assert "This is a revision" in user
    assert '[{"task": "old"}]' in user
    assert "Quality review notes to address:\n- Missing the report deadline" in user


def test_action_extractor_gets_summary_hint_from_prior_results():
    deps = make_deps(prior_results={"Summarizer": {"summary": "S" * 900}})
    user = build_user_prompt("ActionItemExtractor", deps, "Bob: I'll do it")
    assert "Summary (for context" in user
    assert "S" * 500 in user and "S" * 501 not in user  # capped at 500 chars


def test_critic_user_prompt_shape():
    user = build_critic_user_prompt("Summarizer", "The summary.", "The transcript.")
    assert user.startswith("Section: Summarizer")
    assert "Content to review:\nThe summary." in user
    assert "Transcript (ground truth):\nThe transcript." in user
