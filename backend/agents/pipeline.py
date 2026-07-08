"""DB-free workflow pipeline.

`run_pipeline` executes a WorkflowSpec over a transcript and returns a
PipelineResult. It never touches the database — persistence happens through a
PipelineObserver implemented by the orchestrator (and skipped entirely by the
eval harness). Execution is strictly serial: LM Studio loads one model at a
time.
"""
import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings

from .context import NoteDeps
from .definitions import (
    AGENT_REGISTRY,
    INSTRUCTION_BUILDERS,
    build_critic_user_prompt,
    build_user_prompt,
    critic,
    critic_instructions,
)
from .llm import build_model, build_model_settings, wrap_output
from .outputs import AGENT_OUTPUT_TYPES, CritiqueOutput, FALLBACK_CRITIQUE
from .verifiers import RiskClassifier, SchemaVerifier
from .workflow_spec import WorkflowSpec

logger = logging.getLogger("agents.pipeline")

_SCHEMA_VERIFIER = SchemaVerifier()
_RISK_CLASSIFIER = RiskClassifier()


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot produce any output at all."""


class PipelineObserver:
    """No-op persistence hooks; the orchestrator overrides these to write
    WorkflowRun / WorkflowStepResult rows. `step_start` may return a token
    (e.g. the DB row) that is passed back to step_done / step_error."""

    def phase(self, phase: str) -> None:
        pass

    def step_start(self, step_name: str, attempt: int, current_step: str) -> Any:
        return None

    def step_done(
        self,
        token: Any,
        *,
        duration_ms: int,
        result: dict,
        critique_score: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        model_name: str | None = None,
    ) -> None:
        pass

    def step_error(self, token: Any, *, duration_ms: int, error: str) -> None:
        pass


@dataclass
class _UsageTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    model_name: str = ""

    def add(self, usage, response_model_name: str | None = None) -> tuple[int, int]:
        in_tok = usage.input_tokens or 0
        out_tok = usage.output_tokens or 0
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        if response_model_name:
            self.model_name = response_model_name
        return in_tok, out_tok


@dataclass
class PipelineResult:
    results: dict[str, dict]
    critiques: dict[str, dict]
    summary_text: str
    action_items: list
    suggestions_text: str
    confidence_score: float | None
    schema_checks: dict
    risk_classification: dict
    input_tokens: int
    output_tokens: int
    model_name: str


# ---------------------------------------------------------------------------
# Transcript sizing
# ---------------------------------------------------------------------------

def _max_chars(cfg: dict) -> int:
    return max(500, (cfg.get("max_tokens", 4096) - 800) * 4)


def _truncate_transcript(transcript: str, cfg: dict) -> str:
    limit = _max_chars(cfg)
    if len(transcript) > limit:
        return transcript[:limit] + "\n\n[Transcript truncated due to context limit]"
    return transcript


def _split_transcript(transcript: str, max_chars: int, overlap: int = 500) -> list[str]:
    """Split at paragraph/sentence boundaries; overlap to avoid cutting mid-context."""
    target = max(500, int(max_chars * 0.8))
    # Overlap must stay well under the chunk size or `start` stops advancing.
    overlap = min(overlap, target // 4)
    chunks: list[str] = []
    start = 0
    while start < len(transcript):
        end = start + target
        if end >= len(transcript):
            chunks.append(transcript[start:])
            break
        boundary = transcript.rfind("\n\n", start, end)
        if boundary == -1:
            boundary = transcript.rfind(". ", start, end)
        if boundary != -1 and boundary > start:
            end = boundary + 1
        chunks.append(transcript[start:end])
        # Guarantee forward progress even when the boundary snap pulled `end`
        # back inside the overlap window.
        start = end - overlap if end - overlap > start else end
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Single agent / critic executions
# ---------------------------------------------------------------------------

def _response_model_name(run_result) -> str | None:
    try:
        return run_result.response.model_name
    except Exception:
        return None


def _execute_step(
    agent_name: str,
    deps: NoteDeps,
    transcript: str,
    cfg: dict,
    model: OpenAIChatModel,
    settings: ModelSettings,
    observer: PipelineObserver,
    totals: _UsageTotals,
    *,
    attempt: int = 1,
    previous_attempt: str = "",
    quality_notes: list[str] | None = None,
    step_label: str | None = None,
    current_step: str | None = None,
) -> dict:
    label = step_label or agent_name
    token = observer.step_start(label, attempt, current_step or f"extracting:{agent_name.lower()}")
    t0 = time.time()
    try:
        user = build_user_prompt(agent_name, deps, transcript, previous_attempt, quality_notes)
        agent = AGENT_REGISTRY[agent_name]
        if agent_name == "Summarizer":
            # Plain text, never schema-wrapped: the summary is one markdown
            # string, so a JSON grammar buys nothing — and LM Studio's
            # grammar-constrained generation degenerates into whitespace loops
            # on long inputs (~1 tok/s and truncated drafts, where the same
            # prompt as plain text runs ~20 tok/s).
            run_result = agent.run_sync(user, deps=deps, model=model, model_settings=settings)
            output = {"summary": str(run_result.output)}
        else:
            run_result = agent.run_sync(
                user,
                deps=deps,
                model=model,
                model_settings=settings,
                output_type=wrap_output(AGENT_OUTPUT_TYPES[agent_name], cfg),
            )
            output = run_result.output.model_dump()
        in_tok, out_tok = totals.add(run_result.usage, _response_model_name(run_result))
        output["_prompt"] = {"system": INSTRUCTION_BUILDERS[agent_name](deps), "user": user}
        duration_ms = int((time.time() - t0) * 1000)
        logger.info("step %s done in %dms (%d in / %d out tokens)", label, duration_ms, in_tok, out_tok)
        observer.step_done(
            token,
            duration_ms=duration_ms,
            result=output,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model_name=totals.model_name or None,
        )
        return output
    except Exception as exc:
        duration_ms = int((time.time() - t0) * 1000)
        logger.warning("step %s failed after %dms: %s", label, duration_ms, exc)
        observer.step_error(token, duration_ms=duration_ms, error=str(exc))
        raise


def _retry_quality_notes(name: str, critique: dict, content: str, transcript: str) -> list[str]:
    """Feedback for a retry: the critic's issues, plus a plain-spoken length
    warning when a long meeting got a degenerately short summary — the most
    common failure mode on small local models, and one the critic's issue list
    rarely names explicitly."""
    notes = list(critique.get("issues") or [])
    if (
        name == "Summarizer"
        and len(transcript) > 10_000
        and len(content) < min(1500, len(transcript) // 25)
    ):
        notes.append(
            f"The draft is only ~{len(content)} characters for a ~{len(transcript)}-character "
            "transcript — far too short. Expand every section with the specific details discussed."
        )
    return notes


def _critique_content(result: dict) -> str:
    """Pick the content of a step output for the Critic to review."""
    if "summary" in result:
        return result["summary"]
    if "decisions" in result:
        return json.dumps(result["decisions"], indent=2)
    # Everything else (action items, interview assessment, lecture extraction)
    # is reviewed as a whole — flags and follow-ups matter as much as any
    # single field.
    return json.dumps({k: v for k, v in result.items() if not k.startswith("_")}, indent=2)[:2000]


def _execute_critique(
    agent_name: str,
    content: str,
    deps: NoteDeps,
    transcript: str,
    cfg: dict,
    model: OpenAIChatModel,
    settings: ModelSettings,
    observer: PipelineObserver,
    totals: _UsageTotals,
) -> dict:
    token = observer.step_start(f"Critic:{agent_name}", 1, f"critiquing:{agent_name.lower()}")
    t0 = time.time()
    user = build_critic_user_prompt(agent_name, content, transcript)
    try:
        run_result = critic.run_sync(
            user,
            deps=deps,
            model=model,
            model_settings=settings,
            output_type=wrap_output(CritiqueOutput, cfg),
        )
        critique = run_result.output.model_dump()
        in_tok, out_tok = totals.add(run_result.usage, _response_model_name(run_result))
        critique["_prompt"] = {"system": critic_instructions(), "user": user}
        duration_ms = int((time.time() - t0) * 1000)
        logger.info("critique of %s scored %.1f in %dms", agent_name, critique["score"], duration_ms)
        observer.step_done(
            token,
            duration_ms=duration_ms,
            result=critique,
            critique_score=critique["score"],
            input_tokens=in_tok,
            output_tokens=out_tok,
            model_name=totals.model_name or None,
        )
        return critique
    except UnexpectedModelBehavior:
        # Unparseable critique: advisory fallback score, recorded as done (legacy semantics).
        critique = dict(FALLBACK_CRITIQUE)
        duration_ms = int((time.time() - t0) * 1000)
        logger.warning("critique of %s unparseable; using fallback score %.1f", agent_name, critique["score"])
        observer.step_done(
            token, duration_ms=duration_ms, result=critique, critique_score=critique["score"]
        )
        return critique
    except Exception as exc:
        duration_ms = int((time.time() - t0) * 1000)
        logger.warning("critique of %s failed after %dms: %s", agent_name, duration_ms, exc)
        observer.step_error(token, duration_ms=duration_ms, error=str(exc))
        # Neutral critique so execution continues (legacy semantics).
        return {"score": 7.0, "issues": []}


# ---------------------------------------------------------------------------
# Chunked map-reduce for over-length transcripts
# ---------------------------------------------------------------------------

def _chunked_summarize(
    transcript: str,
    deps: NoteDeps,
    cfg: dict,
    model: OpenAIChatModel,
    settings: ModelSettings,
    observer: PipelineObserver,
    totals: _UsageTotals,
) -> str:
    """Run Summarizer per chunk; the concatenated partial summaries become a
    condensed substitute transcript for the downstream steps."""
    chunks = _split_transcript(transcript, _max_chars(cfg))
    n = len(chunks)
    partials: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_deps = dataclasses.replace(
            deps,
            template_prompt=deps.template_prompt + f"\n[Segment {i + 1} of {n} — full synthesis follows]",
        )
        try:
            output = _execute_step(
                "Summarizer", chunk_deps, chunk, cfg, model, settings, observer, totals,
                step_label=f"Summarizer[chunk {i + 1}/{n}]",
                current_step=f"chunking:{i + 1}/{n}",
            )
            partials.append(output.get("summary", ""))
        except Exception:
            logger.exception("chunk %d/%d failed; continuing", i + 1, n)
    return "\n\n---\n\n".join(f"[Segment {i + 1}]\n{s}" for i, s in enumerate(partials) if s)


# ---------------------------------------------------------------------------
# Assembly: build suggestions_text from domain-specific agent outputs
# ---------------------------------------------------------------------------

def _assemble_suggestions(results: dict) -> str:
    """Markdown sections from whichever domain agents actually ran.

    Driven by the results, not the domain name, so custom template workflows
    (workflow_config) that mix agents across domains still assemble output.
    """
    parts: list[str] = []

    decisions = results.get("DecisionLogger", {}).get("decisions", [])
    if decisions:
        lines = "\n".join(
            "- **" + d.get("decision", "") + "**"
            + (f" *(rationale: {d['rationale']})*" if d.get("rationale") else "")
            for d in decisions
        )
        parts.append(f"## Decisions Made\n{lines}")

    r = results.get("InterviewAgent", {})
    if r.get("red_flags"):
        parts.append("## Red Flags\n" + "\n".join(f"- {f}" for f in r["red_flags"]))
    if r.get("green_flags"):
        parts.append("## Green Flags\n" + "\n".join(f"- {f}" for f in r["green_flags"]))
    if r.get("candidate_highlights"):
        parts.append("## Candidate Highlights\n" + "\n".join(f"- {h}" for h in r["candidate_highlights"]))
    if r.get("suggested_followups"):
        parts.append("## Suggested Follow-up Questions\n" + "\n".join(f"- {q}" for q in r["suggested_followups"]))

    r = results.get("LectureAgent", {})
    if r.get("key_concepts"):
        lines = "\n".join(
            f"- **{c.get('concept', '')}**: {c.get('definition', '')}"
            for c in r["key_concepts"]
        )
        parts.append(f"## Key Concepts\n{lines}")
    if r.get("learning_objectives"):
        parts.append("## Learning Objectives\n" + "\n".join(f"- {o}" for o in r["learning_objectives"]))
    if r.get("quiz_questions"):
        lines = "\n".join(
            f"- **Q:** {q.get('question', '')}  \n  **A:** {q.get('answer', '')}"
            for q in r["quiz_questions"]
        )
        parts.append(f"## Quiz Questions\n{lines}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    *,
    transcript: str,
    spec: WorkflowSpec,
    deps: NoteDeps,
    cfg: dict,
    observer: PipelineObserver | None = None,
) -> PipelineResult:
    observer = observer or PipelineObserver()

    model = build_model(cfg)
    settings = build_model_settings(cfg)
    totals = _UsageTotals(model_name=cfg.get("model") or "")

    # cfg["max_tokens"] is the model's context budget. Transcripts that exceed
    # it go through chunked map-reduce instead of being silently truncated.
    # (The legacy engine inflated max_tokens to fit the transcript, which made
    # its chunked path unreachable and could overflow the model's context.)
    if len(transcript) > _max_chars(cfg):
        observer.phase("chunking")
        logger.info("transcript of %d chars exceeds window; chunking", len(transcript))
        transcript = _chunked_summarize(transcript, deps, cfg, model, settings, observer, totals)
    transcript = _truncate_transcript(transcript, cfg)

    results: dict[str, dict] = {}
    critiques: dict[str, dict] = {}
    deps = dataclasses.replace(deps, prior_results=results)
    step_deps: dict[str, NoteDeps] = {}

    # --- Extraction phase (serial — LM Studio is single-model) ---
    observer.phase("extracting")
    for step in spec.steps:
        name = step.agent
        step_deps[name] = (
            deps if step.prompt_override is None
            else dataclasses.replace(deps, template_prompt=step.prompt_override)
        )
        try:
            results[name] = _execute_step(
                name, step_deps[name], transcript, cfg, model, settings, observer, totals
            )
        except Exception:
            results[name] = {}

    if results and all(not r for r in results.values()):
        raise PipelineError("all workflow steps failed — is the LLM endpoint reachable?")

    # --- Critique + retry phase ---
    observer.phase("critiquing")
    for name in spec.critique_steps:
        if not results.get(name):
            continue
        content = _critique_content(results[name])
        critique = _execute_critique(
            name, content, deps, transcript, cfg, model, settings, observer, totals
        )
        critiques[name] = critique
        best_result, best_critique = results[name], critique

        attempt = 2
        while critique["score"] < spec.critique_threshold and attempt <= spec.max_retries + 1:
            try:
                result = _execute_step(
                    name, step_deps[name], transcript, cfg, model, settings, observer, totals,
                    attempt=attempt,
                    previous_attempt=content,
                    quality_notes=_retry_quality_notes(name, critique, content, transcript),
                )
                results[name] = result
                content = _critique_content(result)
                critique = _execute_critique(
                    name, content, deps, transcript, cfg, model, settings, observer, totals
                )
                critiques[name] = critique
                if critique["score"] > best_critique["score"]:
                    best_result, best_critique = result, critique
            except Exception:
                break
            attempt += 1

        # Retries that never met the threshold must not replace a
        # better-scoring earlier draft with a worse final one.
        results[name] = best_result
        critiques[name] = best_critique

    # --- Assembly phase (pure Python, no LLM call) ---
    observer.phase("assembling")
    summary_text = results.get("Summarizer", {}).get("summary", "")
    action_items = results.get("ActionItemExtractor", {}).get("action_items", [])
    suggestions_text = _assemble_suggestions(results)

    scores = [c["score"] for c in critiques.values() if "score" in c]
    confidence_score = sum(scores) / len(scores) if scores else None

    # --- Verification phase (non-LLM, advisory) ---
    schema_checks = _SCHEMA_VERIFIER.verify_all(results)
    risk_classification = _RISK_CLASSIFIER.classify(
        deps.domain_name, summary_text + " " + suggestions_text
    )

    return PipelineResult(
        results=results,
        critiques=critiques,
        summary_text=summary_text,
        action_items=action_items,
        suggestions_text=suggestions_text,
        confidence_score=confidence_score,
        schema_checks=schema_checks,
        risk_classification=risk_classification,
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        model_name=totals.model_name,
    )
