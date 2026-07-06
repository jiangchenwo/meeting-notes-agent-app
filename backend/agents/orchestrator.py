import dataclasses
import datetime
import json
import time

from database import SessionLocal
from models import NoteBlock, Transcription, Summary, WorkflowRun, WorkflowStepResult
from transcript_format import build_speaker_transcript
import lm_config

from .base import WorkflowContext
from .core_agents import Critic
from .registry import create_agent_registry
from .verifiers import RiskClassifier, SchemaVerifier
from .workflows import DOMAIN_WORKFLOWS, _DEFAULT_WORKFLOW, select_workflow

# ---------------------------------------------------------------------------
# Agent registry — single shared instance per agent type
# ---------------------------------------------------------------------------
_AGENTS = create_agent_registry()

_CRITIC = Critic()
_SCHEMA_VERIFIER = SchemaVerifier()
_RISK_CLASSIFIER = RiskClassifier()

# ---------------------------------------------------------------------------
# Workflow definitions (deterministic rule engine — no LLM orchestrator call)
# ---------------------------------------------------------------------------
_select_workflow = select_workflow


# ---------------------------------------------------------------------------
# Assembly: build suggestions_text from domain-specific agent outputs
# ---------------------------------------------------------------------------
def _assemble_suggestions(results: dict, domain_name: str) -> str:
    parts: list[str] = []

    if domain_name == "Project":
        decisions = results.get("DecisionLogger", {}).get("decisions", [])
        if decisions:
            lines = "\n".join(
                "- **" + d.get("decision", "") + "**"
                + (f" *(rationale: {d['rationale']})*" if d.get("rationale") else "")
                for d in decisions
            )
            parts.append(f"## Decisions Made\n{lines}")

    elif domain_name == "Interview":
        r = results.get("InterviewAgent", {})
        if r.get("red_flags"):
            parts.append("## Red Flags\n" + "\n".join(f"- {f}" for f in r["red_flags"]))
        if r.get("green_flags"):
            parts.append("## Green Flags\n" + "\n".join(f"- {f}" for f in r["green_flags"]))
        if r.get("candidate_highlights"):
            parts.append("## Candidate Highlights\n" + "\n".join(f"- {h}" for h in r["candidate_highlights"]))
        if r.get("suggested_followups"):
            parts.append("## Suggested Follow-up Questions\n" + "\n".join(f"- {q}" for q in r["suggested_followups"]))

    elif domain_name == "Education":
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
# Step execution helpers
# ---------------------------------------------------------------------------
def _run_step(
    agent_name: str,
    ctx: WorkflowContext,
    run: WorkflowRun,
    db,
    attempt: int = 1,
    extra_instructions: list[str] | None = None,
) -> tuple[dict, WorkflowStepResult]:
    if extra_instructions:
        addendum = "\n\nQuality review notes to address:\n" + "\n".join(f"- {i}" for i in extra_instructions)
        ctx = dataclasses.replace(ctx, template_prompt=ctx.template_prompt + addendum)

    step_rec = WorkflowStepResult(run_id=run.id, step_name=agent_name, status="running", attempt=attempt)
    db.add(step_rec)
    run.current_step = f"extracting:{agent_name.lower()}"
    db.commit()

    t0 = time.time()
    try:
        result = _AGENTS[agent_name].run(ctx)
        step_rec.status = "done"
        step_rec.duration_ms = int((time.time() - t0) * 1000)
        result.pop("_tokens", None)
        step_rec.result_json = json.dumps(result)
        db.commit()
        return result, step_rec
    except Exception as exc:
        step_rec.status = "error"
        step_rec.duration_ms = int((time.time() - t0) * 1000)
        step_rec.result_json = json.dumps({"error": str(exc)})
        db.commit()
        raise


def _run_critique(
    agent_name: str,
    result: dict,
    ctx: WorkflowContext,
    run: WorkflowRun,
    db,
) -> dict:
    # Pick the main text field to critique
    if "summary" in result:
        content = result["summary"]
    elif "decisions" in result:
        content = json.dumps(result["decisions"], indent=2)
    elif "obligations" in result:
        content = json.dumps(result.get("obligations", []) + result.get("risks", []), indent=2)
    elif "questions_asked" in result:
        content = json.dumps(result["questions_asked"], indent=2)
    else:
        content = json.dumps(result, indent=2)[:1000]

    step_rec = WorkflowStepResult(
        run_id=run.id, step_name=f"Critic:{agent_name}", status="running", attempt=1
    )
    db.add(step_rec)
    run.current_step = f"critiquing:{agent_name.lower()}"
    db.commit()

    t0 = time.time()
    try:
        critique = _CRITIC.run_critique(ctx, agent_name, content)
        critique.pop("_tokens", None)
        step_rec.status = "done"
        step_rec.duration_ms = int((time.time() - t0) * 1000)
        step_rec.critique_score = critique["score"]
        step_rec.result_json = json.dumps(critique)
        db.commit()
        return critique
    except Exception as exc:
        step_rec.status = "error"
        step_rec.duration_ms = int((time.time() - t0) * 1000)
        step_rec.result_json = json.dumps({"error": str(exc)})
        db.commit()
        # Return a neutral critique so execution continues
        return {"score": 7.0, "issues": [], "improved_version": content}


# ---------------------------------------------------------------------------
# Helpers for full-transcript processing (no silent truncation)
# ---------------------------------------------------------------------------

def _split_transcript(transcript: str, max_chars: int, overlap: int = 500) -> list[str]:
    """Split at paragraph/sentence boundaries; 10% overlap to avoid cutting mid-context."""
    target = max(500, int(max_chars * 0.8))
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
        start = end - overlap
    return [c for c in chunks if c.strip()]


def _chunked_summarize(
    transcript: str,
    domain_name: str,
    template_prompt: str,
    system_prompt: str,
    knowledge_base: str,
    cfg: dict,
    run: "WorkflowRun",
    db,
) -> str:
    """
    Map-reduce for transcripts too long for a single context window.
    Runs Summarizer on each chunk, then returns the concatenated partial summaries
    as a condensed substitute transcript for downstream agents.
    DB step records are written for transparency in the workflow run view.
    """
    max_chars = max(500, (cfg.get("max_tokens", 4096) - 800) * 4)
    chunks = _split_transcript(transcript, max_chars)
    n = len(chunks)
    partial_summaries: list[str] = []

    for i, chunk in enumerate(chunks):
        step_rec = WorkflowStepResult(
            run_id=run.id,
            step_name=f"Summarizer[chunk {i + 1}/{n}]",
            status="running",
            attempt=1,
        )
        db.add(step_rec)
        run.current_step = f"chunking:{i + 1}/{n}"
        db.commit()

        chunk_ctx = WorkflowContext(
            note_id=run.note_block_id,
            transcript=chunk,
            domain_name=domain_name,
            template_name="Default",
            template_prompt=template_prompt + f"\n[Segment {i + 1} of {n} — full synthesis follows]",
            project_system_prompt=system_prompt,
            project_knowledge_base=knowledge_base,
            cfg=cfg,
        )
        t0 = time.time()
        try:
            result = _AGENTS["Summarizer"].run(chunk_ctx)
            result.pop("_tokens", None)
            step_rec.status = "done"
            step_rec.duration_ms = int((time.time() - t0) * 1000)
            step_rec.result_json = json.dumps(result)
            partial_summaries.append(result.get("summary", ""))
        except Exception as exc:
            step_rec.status = "error"
            step_rec.duration_ms = int((time.time() - t0) * 1000)
            step_rec.result_json = json.dumps({"error": str(exc)})
        db.commit()

    return "\n\n---\n\n".join(
        f"[Segment {i + 1}]\n{s}" for i, s in enumerate(partial_summaries) if s
    )


# ---------------------------------------------------------------------------
# Main entry point — called as a FastAPI BackgroundTask
# ---------------------------------------------------------------------------
def run_workflow(note_id: int) -> None:
    db = SessionLocal()
    run: WorkflowRun | None = None
    try:
        note = db.get(NoteBlock, note_id)
        if not note:
            return

        transcription = db.query(Transcription).filter_by(note_block_id=note_id).first()
        if not transcription or not transcription.full_text:
            note.status = "error"
            db.commit()
            return

        run = WorkflowRun(note_block_id=note_id, status="planning")
        db.add(run)
        note.status = "summarizing"
        note.updated_at = datetime.datetime.utcnow()
        db.commit()

        cfg = lm_config.load()
        domain_name = note.domain.name if note.domain else "Project"
        template_prompt = (
            note.template.prompt_template
            if note.template and note.template.prompt_template
            else "Summarize the meeting: key decisions, action items, and blockers."
        )
        template_workflow_config = note.template.workflow_config if note.template else None
        project_system_prompt = note.project.custom_system_prompt if note.project else ""
        project_knowledge_base = note.project.knowledge_base if note.project else ""

        # Layer 1: inflate max_tokens so _truncate_transcript passes the full transcript.
        # With user-configured max_tokens=40960 this gives ~160K chars — fits most meetings.
        transcript_text = build_speaker_transcript(transcription.full_text, transcription.segments_json)
        cfg = {**cfg, "max_tokens": max(cfg.get("max_tokens", 4096), len(transcript_text) // 4 + 1000)}

        # Layer 2: chunked map-reduce for transcripts that still exceed the context window.
        max_chars = max(500, (cfg["max_tokens"] - 800) * 4)
        if len(transcript_text) > max_chars:
            run.status = "chunking"
            db.commit()
            transcript_text = _chunked_summarize(
                transcript=transcript_text,
                domain_name=domain_name,
                template_prompt=template_prompt,
                system_prompt=project_system_prompt,
                knowledge_base=project_knowledge_base,
                cfg=cfg,
                run=run,
                db=db,
            )

        ctx = WorkflowContext(
            note_id=note_id,
            transcript=transcript_text,
            domain_name=domain_name,
            template_name=note.template.name if note.template else "Default",
            template_prompt=template_prompt,
            project_system_prompt=project_system_prompt,
            project_knowledge_base=project_knowledge_base,
            cfg=cfg,
        )

        workflow = _select_workflow(domain_name, template_workflow_config)
        run.workflow_plan_json = json.dumps(workflow)
        run.status = "extracting"
        db.commit()

        # --- Extraction phase (serial — LM Studio is single-model) ---
        for agent_name in workflow["steps"]:
            if agent_name not in _AGENTS:
                continue
            try:
                result, _ = _run_step(agent_name, ctx, run, db)
                ctx.results[agent_name] = result
            except Exception:
                ctx.results[agent_name] = {}

        # --- Critique + optional retry phase ---
        run.status = "critiquing"
        db.commit()

        critique_threshold: float = float(workflow.get("critique_threshold", 7))
        max_retries: int = int(workflow.get("max_retries", 1))

        for agent_name in workflow.get("critique_steps", []):
            if agent_name not in ctx.results or not ctx.results[agent_name]:
                continue

            critique = _run_critique(agent_name, ctx.results[agent_name], ctx, run, db)
            ctx.critique_results[agent_name] = critique

            # Extract the critiqued content so the retry can reference the previous attempt
            _r = ctx.results.get(agent_name, {})
            if "summary" in _r:
                current_content = _r["summary"]
            elif "decisions" in _r:
                current_content = json.dumps(_r["decisions"], indent=2)
            elif "questions_asked" in _r:
                current_content = json.dumps(_r["questions_asked"], indent=2)
            else:
                current_content = json.dumps(_r, indent=2)[:1000]

            attempt = 2
            while critique["score"] < critique_threshold and attempt <= max_retries + 1:
                ctx = dataclasses.replace(ctx, previous_attempt=current_content)
                try:
                    result, _ = _run_step(
                        agent_name, ctx, run, db,
                        attempt=attempt,
                        extra_instructions=critique.get("issues") or [],
                    )
                    ctx.results[agent_name] = result
                    critique = _run_critique(agent_name, result, ctx, run, db)
                    ctx.critique_results[agent_name] = critique
                except Exception:
                    break
                attempt += 1

        # --- Assembly phase (pure Python, no LLM call) ---
        run.status = "assembling"
        run.current_step = "assembling"
        db.commit()

        summary_text = ctx.results.get("Summarizer", {}).get("summary", "")
        action_items = ctx.results.get("ActionItemExtractor", {}).get("action_items", [])
        suggestions_text = _assemble_suggestions(ctx.results, domain_name)

        scores = [c["score"] for c in ctx.critique_results.values() if "score" in c]
        confidence_score = sum(scores) / len(scores) if scores else None

        # --- Verification phase (non-LLM) ---
        schema_results = _SCHEMA_VERIFIER.verify_all(ctx.results)
        risk_result = _RISK_CLASSIFIER.classify(
            domain_name, summary_text + " " + suggestions_text
        )

        raw_sections = {
            **ctx.results,
            "_schema_checks": schema_results,
            "_risk_classification": risk_result,
        }

        now = datetime.datetime.utcnow()
        model = cfg.get("model", "")
        existing = db.query(Summary).filter_by(note_block_id=note_id).first()
        if existing:
            existing.summary_text = summary_text
            existing.action_items_json = json.dumps(action_items)
            existing.suggestions_text = suggestions_text
            existing.llm_model_used = model
            existing.generated_at = now
            existing.workflow_run_id = run.id
            existing.confidence_score = confidence_score
            existing.raw_sections_json = json.dumps(raw_sections)
        else:
            db.add(Summary(
                note_block_id=note_id,
                summary_text=summary_text,
                action_items_json=json.dumps(action_items),
                suggestions_text=suggestions_text,
                llm_model_used=model,
                generated_at=now,
                workflow_run_id=run.id,
                confidence_score=confidence_score,
                raw_sections_json=json.dumps(raw_sections),
            ))

        note.status = "done"
        note.updated_at = now
        run.status = "done"
        run.finished_at = now
        db.commit()

    except Exception as exc:
        db.rollback()
        try:
            note = db.get(NoteBlock, note_id)
            if note:
                note.status = "error"
                note.updated_at = datetime.datetime.utcnow()
            if run:
                run.status = "error"
                run.error_message = str(exc)
                run.finished_at = datetime.datetime.utcnow()
            db.commit()
        except Exception:
            pass
    finally:
        db.close()
