"""Pipeline behavior tests: scripted FunctionModel runs, no real LLM.

The pipeline builds its model from cfg internally, so tests use
Agent.override(model=FunctionModel(...)) on every registered agent (plus the
critic) to intercept all calls. Responders dispatch on the instructions text.
"""
import json
from contextlib import ExitStack, contextmanager

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from agents.context import NoteDeps
from agents.definitions import AGENT_REGISTRY, critic
from agents.outputs import FALLBACK_CRITIQUE
from agents.pipeline import PipelineError, PipelineObserver, run_pipeline
from agents.workflow_spec import DOMAIN_WORKFLOWS, WorkflowSpec

# Instruction-text markers identifying which agent a model request came from.
MARKERS = {
    "focused meeting summarizer": "Summarizer",
    "extract concrete action items": "ActionItemExtractor",
    "extract explicit decisions": "DecisionLogger",
    "job interview transcripts": "InterviewAgent",
    "educational content": "LectureAgent",
    "strict quality reviewer": "Critic",
}

DIMS_9 = {"coverage": 4, "accuracy": 3, "specificity": 1, "structure": 1}
DIMS_5 = {"coverage": 2, "accuracy": 1, "specificity": 1, "structure": 1}


def which_agent(messages) -> str:
    instructions = messages[-1].instructions or ""
    for marker, name in MARKERS.items():
        if marker in instructions:
            return name
    raise AssertionError(f"unrecognized instructions: {instructions[:120]}")


def user_text(messages) -> str:
    return "\n".join(
        str(p.content) for p in messages[-1].parts if p.part_kind == "user-prompt"
    )


def text(payload: dict) -> ModelResponse:
    return ModelResponse(parts=[TextPart(json.dumps(payload))])


def plain(s: str) -> ModelResponse:
    """Plain-text response — the Summarizer runs without a schema wrap."""
    return ModelResponse(parts=[TextPart(s)])


def make_deps(**overrides) -> NoteDeps:
    defaults = dict(
        note_id=1,
        domain_name="General",
        template_name="Default",
        template_prompt="Summarize the meeting.",
    )
    defaults.update(overrides)
    return NoteDeps(**defaults)


@contextmanager
def scripted_model(responder):
    fm = FunctionModel(responder)
    with ExitStack() as stack:
        for agent in AGENT_REGISTRY.values():
            stack.enter_context(agent.override(model=fm))
        stack.enter_context(critic.override(model=fm))
        yield


class RecordingObserver(PipelineObserver):
    def __init__(self):
        self.phases: list[str] = []
        self.steps: list[dict] = []

    def phase(self, phase):
        self.phases.append(phase)

    def step_start(self, step_name, attempt, current_step):
        rec = {"name": step_name, "attempt": attempt, "current_step": current_step, "status": "running"}
        self.steps.append(rec)
        return rec

    def step_done(self, token, *, duration_ms, result, critique_score=None,
                  input_tokens=None, output_tokens=None, model_name=None):
        token.update(
            status="done", result=result, critique_score=critique_score,
            input_tokens=input_tokens, output_tokens=output_tokens,
        )

    def step_error(self, token, *, duration_ms, error):
        token.update(status="error", error=error)

    def names(self) -> list[str]:
        return [s["name"] for s in self.steps]


def happy_responder(messages, info):
    agent = which_agent(messages)
    if agent == "Summarizer":
        return plain("## Summary\nBudget of $50k was approved.")
    if agent == "ActionItemExtractor":
        return text({"action_items": [
            {"task": "Send the report", "owner": "Alice", "deadline": None, "priority": "high"}
        ]})
    if agent == "DecisionLogger":
        return text({"decisions": [
            {"decision": "Approve the budget", "rationale": "On track", "made_by": "group"}
        ]})
    if agent == "Critic":
        return text({"dimensions": DIMS_9, "issues": []})
    raise AssertionError(f"unexpected agent {agent}")


TRANSCRIPT = "Alice: We approved the $50k budget.\nBob: I will send the report."


def test_happy_path_general_workflow(cfg):
    obs = RecordingObserver()
    with scripted_model(happy_responder):
        result = run_pipeline(
            transcript=TRANSCRIPT,
            spec=DOMAIN_WORKFLOWS["General"],
            deps=make_deps(),
            cfg=cfg,
            observer=obs,
        )

    assert set(result.results) == {"Summarizer", "ActionItemExtractor", "DecisionLogger"}
    assert "Budget of $50k" in result.summary_text
    assert result.action_items[0]["task"] == "Send the report"
    assert result.critiques["Summarizer"]["score"] == 9.0
    assert result.confidence_score == 9.0
    assert result.input_tokens > 0 and result.output_tokens > 0
    assert result.model_name  # backfilled from the model response
    # Assembly is results-driven: decisions render for any domain that ran DecisionLogger
    assert "## Decisions Made" in result.suggestions_text

    # Persisted prompt capture
    prompt = result.results["Summarizer"]["_prompt"]
    assert "focused meeting summarizer" in prompt["system"]
    assert TRANSCRIPT in prompt["user"]

    # Observer saw the full sequence with legacy-compatible naming
    assert obs.phases == ["extracting", "critiquing", "assembling"]
    assert obs.names() == [
        "Summarizer", "ActionItemExtractor", "DecisionLogger", "Critic:Summarizer",
    ]
    assert [s["current_step"] for s in obs.steps] == [
        "extracting:summarizer", "extracting:actionitemextractor",
        "extracting:decisionlogger", "critiquing:summarizer",
    ]
    assert all(s["status"] == "done" for s in obs.steps)
    assert all(s["input_tokens"] > 0 for s in obs.steps)
    assert obs.steps[-1]["critique_score"] == 9.0

    # Advisory verifiers ran
    assert isinstance(result.schema_checks, dict)
    assert isinstance(result.risk_classification, dict)


def test_critique_retry_improves_summary(cfg):
    critic_calls = 0
    retry_prompts: list[str] = []

    def responder(messages, info):
        nonlocal critic_calls
        agent = which_agent(messages)
        if agent == "Critic":
            critic_calls += 1
            if critic_calls == 1:
                return text({"dimensions": DIMS_5, "issues": ["Add the budget decision"]})
            return text({"dimensions": DIMS_9, "issues": []})
        if agent == "Summarizer":
            user = user_text(messages)
            if "This is a revision" in user:
                retry_prompts.append(user)
                return plain("## Revised\nBudget of $50k approved by the group.")
            return plain("## Draft\nA meeting happened.")
        if agent == "ActionItemExtractor":
            return text({"action_items": []})
        if agent == "DecisionLogger":
            return text({"decisions": []})
        raise AssertionError(agent)

    obs = RecordingObserver()
    with scripted_model(responder):
        result = run_pipeline(
            transcript=TRANSCRIPT,
            spec=DOMAIN_WORKFLOWS["General"],
            deps=make_deps(),
            cfg=cfg,
            observer=obs,
        )

    # Retry ran as attempt 2 with feedback in the prompt
    assert [(s["name"], s["attempt"]) for s in obs.steps] == [
        ("Summarizer", 1), ("ActionItemExtractor", 1), ("DecisionLogger", 1),
        ("Critic:Summarizer", 1), ("Summarizer", 2), ("Critic:Summarizer", 1),
    ]
    assert len(retry_prompts) == 1
    assert "Quality review notes to address:\n- Add the budget decision" in retry_prompts[0]
    assert "A meeting happened." in retry_prompts[0]  # previous draft carried over

    assert "Revised" in result.summary_text
    assert result.critiques["Summarizer"]["score"] == 9.0
    assert result.confidence_score == 9.0


def test_retry_stops_at_max_retries(cfg):
    def responder(messages, info):
        agent = which_agent(messages)
        if agent == "Critic":
            return text({"dimensions": DIMS_5, "issues": ["Still bad"]})
        return plain("## Draft")

    spec = WorkflowSpec(steps=["Summarizer"], critique_steps=["Summarizer"], max_retries=1)
    obs = RecordingObserver()
    with scripted_model(responder):
        result = run_pipeline(
            transcript=TRANSCRIPT, spec=spec, deps=make_deps(), cfg=cfg, observer=obs
        )

    # attempt 1 + exactly one retry, then give up with the low score
    assert [(s["name"], s["attempt"]) for s in obs.steps] == [
        ("Summarizer", 1), ("Critic:Summarizer", 1), ("Summarizer", 2), ("Critic:Summarizer", 1),
    ]
    assert result.confidence_score == 5.0


def test_retry_keeps_best_scoring_attempt(cfg):
    """A retry that scores worse than an earlier draft must not replace it."""
    dims_7 = {"coverage": 3, "accuracy": 2, "specificity": 1, "structure": 1}
    dims_6 = {"coverage": 2, "accuracy": 2, "specificity": 1, "structure": 1}
    critic_calls = 0

    def responder(messages, info):
        nonlocal critic_calls
        agent = which_agent(messages)
        if agent == "Critic":
            critic_calls += 1
            return text({
                "dimensions": dims_7 if critic_calls == 1 else dims_6,
                "issues": ["More detail"],
            })
        if agent == "Summarizer":
            if "This is a revision" in user_text(messages):
                return plain("## Worse\nShorter.")
            return plain("## Best draft\nBudget of $50k approved by the group.")
        raise AssertionError(agent)

    spec = WorkflowSpec(steps=["Summarizer"], critique_steps=["Summarizer"], max_retries=1)
    with scripted_model(responder):
        result = run_pipeline(transcript=TRANSCRIPT, spec=spec, deps=make_deps(), cfg=cfg)

    assert "Best draft" in result.summary_text
    assert result.critiques["Summarizer"]["score"] == 7.0
    assert result.confidence_score == 7.0


def test_short_summary_of_long_transcript_gets_length_note(cfg):
    """A degenerately short summary of a long meeting adds an explicit length
    issue to the retry feedback, even when the critic's issue list is empty."""
    long_transcript = "Alice: point.\n" * 860  # ~12k chars, below the chunking limit
    retry_prompts: list[str] = []

    def responder(messages, info):
        agent = which_agent(messages)
        if agent == "Critic":
            return text({"dimensions": DIMS_5, "issues": []})
        if agent == "Summarizer":
            user = user_text(messages)
            if "This is a revision" in user:
                retry_prompts.append(user)
            return plain("## Too short")
        raise AssertionError(agent)

    spec = WorkflowSpec(steps=["Summarizer"], critique_steps=["Summarizer"], max_retries=1)
    with scripted_model(responder):
        run_pipeline(transcript=long_transcript, spec=spec, deps=make_deps(), cfg=cfg)

    assert len(retry_prompts) == 1
    assert "far too short" in retry_prompts[0]


def test_chunked_map_reduce_for_long_transcript(cfg):
    cfg["max_tokens"] = 1000  # max_chars = 800
    transcript = "\n\n".join(
        f"Paragraph {i}: " + "discussion point. " * 12 for i in range(10)
    )
    assert len(transcript) > 800

    def responder(messages, info):
        agent = which_agent(messages)
        assert agent == "Summarizer"
        return plain("Partial recap of this segment.")

    spec = WorkflowSpec(steps=["Summarizer"], critique_steps=[])
    obs = RecordingObserver()
    with scripted_model(responder):
        result = run_pipeline(
            transcript=transcript, spec=spec, deps=make_deps(), cfg=cfg, observer=obs
        )

    chunk_steps = [s for s in obs.steps if s["name"].startswith("Summarizer[chunk ")]
    assert len(chunk_steps) >= 2
    assert chunk_steps[0]["current_step"] == f"chunking:1/{len(chunk_steps)}"
    assert obs.phases[0] == "chunking"
    assert "extracting" in obs.phases
    # The reduce digest is then summarized by the normal step
    assert obs.steps[-1]["name"] == "Summarizer"
    assert result.summary_text == "Partial recap of this segment."


def test_split_transcript_terminates_and_covers_input():
    # Regression: legacy splitter looped forever when overlap >= progress per
    # chunk (hidden because the legacy chunked path was unreachable).
    from agents.pipeline import _split_transcript

    transcript = "\n\n".join(f"Paragraph {i}: " + "word " * 40 for i in range(20))
    chunks = _split_transcript(transcript, 800)
    assert len(chunks) >= 2
    assert all(len(c) <= 800 for c in chunks)
    for i in range(20):
        assert any(f"Paragraph {i}:" in c for c in chunks)


def test_all_steps_failing_raises_pipeline_error(cfg):
    def responder(messages, info):
        raise RuntimeError("connection refused")

    obs = RecordingObserver()
    with scripted_model(responder):
        with pytest.raises(PipelineError):
            run_pipeline(
                transcript=TRANSCRIPT,
                spec=DOMAIN_WORKFLOWS["General"],
                deps=make_deps(),
                cfg=cfg,
                observer=obs,
            )

    assert all(s["status"] == "error" for s in obs.steps)
    assert "connection refused" in obs.steps[0]["error"]


def test_unparseable_critique_falls_back_to_score_5(cfg):
    def responder(messages, info):
        agent = which_agent(messages)
        if agent == "Critic":
            return ModelResponse(parts=[TextPart("this is not json at all")])
        return plain("## Fine summary")

    spec = WorkflowSpec(steps=["Summarizer"], critique_steps=["Summarizer"], max_retries=0)
    obs = RecordingObserver()
    with scripted_model(responder):
        result = run_pipeline(
            transcript=TRANSCRIPT, spec=spec, deps=make_deps(), cfg=cfg, observer=obs
        )

    assert result.critiques["Summarizer"] == FALLBACK_CRITIQUE
    assert result.confidence_score == 5.0
    # Recorded as done (advisory), not as an error — legacy semantics
    critic_step = obs.steps[-1]
    assert critic_step["name"] == "Critic:Summarizer"
    assert critic_step["status"] == "done"
    assert critic_step["critique_score"] == 5.0
    # max_retries=0 → no revision pass
    assert obs.names() == ["Summarizer", "Critic:Summarizer"]


def test_summarizer_runs_plain_text_never_schema_wrapped(cfg):
    """The Summarizer generates markdown directly — no JSON grammar (LM Studio
    grammar-constrained generation degenerates on long inputs)."""
    def responder(messages, info):
        agent = which_agent(messages)
        assert agent == "Summarizer"
        return plain("Just a plain prose summary.")

    spec = WorkflowSpec(steps=["Summarizer"], critique_steps=[])
    with scripted_model(responder):
        result = run_pipeline(
            transcript=TRANSCRIPT, spec=spec, deps=make_deps(), cfg=cfg
        )

    assert result.summary_text == "Just a plain prose summary."


def test_prompted_output_mode(cfg):
    cfg["output_mode"] = "prompted"
    with scripted_model(happy_responder):
        result = run_pipeline(
            transcript=TRANSCRIPT,
            spec=DOMAIN_WORKFLOWS["General"],
            deps=make_deps(),
            cfg=cfg,
        )
    assert "Budget of $50k" in result.summary_text
    assert result.critiques["Summarizer"]["score"] == 9.0


def test_prompt_override_replaces_template_prompt(cfg):
    seen_prompts: dict[str, str] = {}

    def responder(messages, info):
        agent = which_agent(messages)
        seen_prompts.setdefault(agent, user_text(messages))
        if agent == "Summarizer":
            return plain("S")
        return text({"decisions": []})

    spec = WorkflowSpec.model_validate({
        "steps": [
            {"agent": "Summarizer", "prompt_override": "Focus only on budget talk."},
            {"agent": "DecisionLogger"},
        ],
    })
    with scripted_model(responder):
        run_pipeline(transcript=TRANSCRIPT, spec=spec, deps=make_deps(), cfg=cfg)

    assert "Focus only on budget talk." in seen_prompts["Summarizer"]
    assert "Summarize the meeting." not in seen_prompts["Summarizer"]


def test_suggestions_assembled_for_project_domain(cfg):
    with scripted_model(happy_responder):
        result = run_pipeline(
            transcript=TRANSCRIPT,
            spec=DOMAIN_WORKFLOWS["Project"],
            deps=make_deps(domain_name="Project"),
            cfg=cfg,
        )
    assert "## Decisions Made" in result.suggestions_text
    assert "**Approve the budget**" in result.suggestions_text
    assert "*(rationale: On track)*" in result.suggestions_text
