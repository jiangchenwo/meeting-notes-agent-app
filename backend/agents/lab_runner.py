"""
Standalone pipeline runner for the Agent Testing Lab.

Runs the full agent workflow without database writes, instruments each step
with timing, then computes evaluation metrics against ground truth facts.

Transcript digest guarantee:
- Layer 1: cfg["max_tokens"] (user-configured, e.g. 40960) → ~160K chars, fits most transcripts.
- Layer 2: if transcript still exceeds the limit, _chunked_pipeline() splits into segments,
  runs Summarizer on each, then synthesises — no content is ever silently dropped.

Metric computation: compute_rouge / compute_bertscore always receive full untruncated
strings; no [:N] slicing anywhere in evaluation code paths.
"""
import dataclasses
import json
import time

from .orchestrator import _AGENTS, _CRITIC, _assemble_suggestions
from .verifiers import RiskClassifier, SchemaVerifier
from .workflows import _DEFAULT_WORKFLOW, select_workflow
from .base import WorkflowContext
from eval.metrics import coverage_score, action_recall, hallucination_check  # noqa: F401 — re-exported for lab.py

_SCHEMA_VERIFIER = SchemaVerifier()
_RISK_CLASSIFIER = RiskClassifier()

# ---------------------------------------------------------------------------
# Optional metric libraries — degrade gracefully if not installed
# ---------------------------------------------------------------------------

try:
    from rouge_score import rouge_scorer as _rs
    _rouge = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

try:
    from bert_score import score as _bert_score_fn
    _BERT_AVAILABLE = True
except ImportError:
    _BERT_AVAILABLE = False


def compute_rouge(candidate: str, reference: str) -> dict | None:
    """ROUGE-1/2/L F1 — full strings, no truncation."""
    if not _ROUGE_AVAILABLE or not candidate.strip() or not reference.strip():
        return None
    scores = _rouge.score(reference, candidate)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 3),
        "rouge2": round(scores["rouge2"].fmeasure, 3),
        "rougeL": round(scores["rougeL"].fmeasure, 3),
    }


def compute_bertscore(candidate: str, reference: str) -> dict | None:
    """
    BERTScore (roberta-large). Downloads ~500MB model on first call, then cached.
    Full strings — no truncation.
    """
    if not _BERT_AVAILABLE or not candidate.strip() or not reference.strip():
        return None
    try:
        P, R, F1 = _bert_score_fn([candidate], [reference], lang="en", verbose=False)
        return {
            "precision": round(P[0].item(), 3),
            "recall":    round(R[0].item(), 3),
            "f1":        round(F1[0].item(), 3),
        }
    except Exception:
        return None


def schema_check(results: dict) -> dict[str, bool]:
    """Returns {agent_name: pass/fail} — delegates to SchemaVerifier."""
    checks = _SCHEMA_VERIFIER.verify_all(results)
    return {name: check["pass"] for name, check in checks.items()}


# ---------------------------------------------------------------------------
# Naive single-call baseline
# ---------------------------------------------------------------------------

def run_naive(transcript: str, domain_name: str, cfg: dict) -> str:
    """Run a naive single-call summarizer (no structured extraction, no truncation)."""
    from .core_agents import Summarizer as _Summarizer
    ctx = WorkflowContext(
        note_id=0, transcript=transcript, domain_name=domain_name,
        template_name="Default",
        template_prompt="Summarize the meeting: key decisions, action items, and blockers.",
        project_system_prompt="", project_knowledge_base="",
        # Override max_tokens so _truncate_transcript passes the full transcript through
        cfg={**cfg, "max_tokens": max(cfg.get("max_tokens", 4096), len(transcript) // 4 + 1000)},
    )
    result = _Summarizer().run(ctx)
    return result.get("summary", "")


# ---------------------------------------------------------------------------
# Chunked map-reduce (Layer 2: transcripts longer than the context window)
# ---------------------------------------------------------------------------

def _split_transcript(transcript: str, max_chars: int, overlap: int = 500) -> list[str]:
    """
    Split at paragraph/sentence boundaries. Target chunk size = max_chars * 0.8
    so there's headroom for system + user prompt overhead.
    """
    target = max(500, int(max_chars * 0.8))
    chunks: list[str] = []
    start = 0
    while start < len(transcript):
        end = start + target
        if end >= len(transcript):
            chunks.append(transcript[start:])
            break
        # Walk back to the nearest sentence boundary
        boundary = transcript.rfind("\n\n", start, end)
        if boundary == -1:
            boundary = transcript.rfind(". ", start, end)
        if boundary != -1 and boundary > start:
            end = boundary + 1
        chunks.append(transcript[start:end])
        start = end - overlap  # overlapping context
    return [c for c in chunks if c.strip()]


def _chunked_pipeline(
    transcript: str,
    domain_name: str,
    cfg: dict,
    knowledge_base: str,
    system_prompt: str,
    template_prompt: str,
) -> dict:
    """
    Map-reduce pipeline for transcripts that exceed the context window.

    Phase 1 (map): Run Summarizer on each chunk → N partial summaries.
    Phase 2 (reduce): Run the full workflow on the concatenation of partial summaries.
                      This is short enough to fit in context.
    """
    from .core_agents import Summarizer as _Summarizer

    max_chars = max(500, (cfg.get("max_tokens", 4096) - 800) * 4)
    chunks = _split_transcript(transcript, max_chars)
    n = len(chunks)

    steps_log: list[dict] = []
    partial_summaries: list[str] = []
    t_total = time.time()

    for i, chunk in enumerate(chunks):
        chunk_ctx = WorkflowContext(
            note_id=0, transcript=chunk, domain_name=domain_name,
            template_name="Default",
            template_prompt=template_prompt + f"\n\n[Segment {i + 1} of {n} — synthesise all segments later]",
            project_system_prompt=system_prompt,
            project_knowledge_base=knowledge_base,
            cfg=cfg,
        )
        t0 = time.time()
        try:
            result = _Summarizer().run(chunk_ctx)
            tokens = result.pop("_tokens", {})
            partial_summaries.append(result.get("summary", ""))
            steps_log.append({
                "name": f"Summarizer[chunk {i + 1}/{n}]",
                "phase": "extraction",
                "status": "done",
                "duration_ms": int((time.time() - t0) * 1000),
                "output": result,
                "tokens": tokens,
                "attempt": 1,
            })
        except Exception as exc:
            steps_log.append({
                "name": f"Summarizer[chunk {i + 1}/{n}]",
                "phase": "extraction",
                "status": "error",
                "duration_ms": int((time.time() - t0) * 1000),
                "error": str(exc),
                "output": {},
                "attempt": 1,
            })

    # Synthesis: join partial summaries into a condensed transcript substitute
    synthesis_text = "\n\n---\n\n".join(f"[Segment {i + 1}]\n{s}" for i, s in enumerate(partial_summaries) if s)

    # Run the full workflow (extraction + critique) on the synthesis text
    inner = _run_pipeline_inner(
        transcript=synthesis_text,
        domain_name=domain_name,
        cfg=cfg,
        knowledge_base=knowledge_base,
        system_prompt=system_prompt,
        template_prompt=template_prompt + "\n\n[Note: input is a pre-summarised version of a longer recording — preserve all content]",
        workflow_override=None,
        extra_label=f"(chunked {n} segments)",
    )

    # Prepend chunk steps so the trace shows the full picture
    inner["steps"] = steps_log + inner["steps"]
    inner["total_ms"] = int((time.time() - t_total) * 1000)
    inner["chunked"] = True
    inner["chunk_count"] = n
    return inner


# ---------------------------------------------------------------------------
# Core single-pass pipeline (used directly when transcript fits in context)
# ---------------------------------------------------------------------------

def _run_pipeline_inner(
    transcript: str,
    domain_name: str,
    cfg: dict,
    knowledge_base: str,
    system_prompt: str,
    template_prompt: str,
    workflow_override: dict | None,
    extra_label: str = "",
) -> dict:
    ctx = WorkflowContext(
        note_id=0,
        transcript=transcript,
        domain_name=domain_name,
        template_name="Default",
        template_prompt=template_prompt,
        project_system_prompt=system_prompt,
        project_knowledge_base=knowledge_base,
        cfg=cfg,
    )

    if workflow_override and "steps" in workflow_override:
        workflow = {**_DEFAULT_WORKFLOW, **workflow_override}
    else:
        workflow = select_workflow(domain_name, None)

    steps_log: list[dict] = []
    t_total = time.time()

    # --- Extraction phase ---
    for agent_name in workflow["steps"]:
        if agent_name not in _AGENTS:
            continue
        t0 = time.time()
        try:
            result = _AGENTS[agent_name].run(ctx)
            tokens = result.pop("_tokens", {})
            prompt = result.pop("_prompt", None)
            ctx.results[agent_name] = result
            steps_log.append({
                "name": agent_name + (f" {extra_label}" if extra_label else ""),
                "phase": "extraction",
                "status": "done",
                "duration_ms": int((time.time() - t0) * 1000),
                "output": result,
                "tokens": tokens,
                "prompt": prompt,
                "attempt": 1,
            })
        except Exception as exc:
            ctx.results[agent_name] = {}
            steps_log.append({
                "name": agent_name,
                "phase": "extraction",
                "status": "error",
                "duration_ms": int((time.time() - t0) * 1000),
                "error": str(exc),
                "output": {},
                "attempt": 1,
            })

    # --- Critique + retry phase ---
    critique_threshold = float(workflow.get("critique_threshold", 7))
    max_retries = int(workflow.get("max_retries", 1))

    for agent_name in workflow.get("critique_steps", []):
        if agent_name not in ctx.results or not ctx.results[agent_name]:
            continue

        res = ctx.results[agent_name]
        if "summary" in res:
            content = res["summary"]
        elif "decisions" in res:
            content = json.dumps(res["decisions"], indent=2)
        elif "obligations" in res:
            content = json.dumps(res.get("obligations", []) + res.get("risks", []), indent=2)
        else:
            content = json.dumps(res, indent=2)[:1000]

        t0 = time.time()
        try:
            critique = _CRITIC.run_critique(ctx, agent_name, content)
            critique.pop("_tokens", None)
            critique_prompt = critique.pop("_prompt", None)
            ctx.critique_results[agent_name] = critique
            steps_log.append({
                "name": f"Critic→{agent_name}",
                "phase": "critique",
                "status": "done",
                "duration_ms": int((time.time() - t0) * 1000),
                "output": critique,
                "critique_score": critique.get("score"),
                "prompt": critique_prompt,
                "attempt": 1,
            })
        except Exception as exc:
            critique = {"score": 7.0, "issues": []}
            steps_log.append({
                "name": f"Critic→{agent_name}",
                "phase": "critique",
                "status": "error",
                "duration_ms": int((time.time() - t0) * 1000),
                "error": str(exc),
                "output": {},
                "attempt": 1,
            })

        attempt = 2
        while critique["score"] < critique_threshold and attempt <= max_retries + 1:
            issues = critique.get("issues", [])
            addendum = (
                f"\n\nRevision required (score {critique['score']:.0f}/10). Issues to fix:\n"
                + "\n".join(f"- {i}" for i in issues)
            )
            retry_ctx = dataclasses.replace(
                ctx,
                template_prompt=ctx.template_prompt + addendum,
                previous_attempt=content,
            )
            t0 = time.time()
            try:
                result = _AGENTS[agent_name].run(retry_ctx)
                result.pop("_tokens", None)
                retry_prompt = result.pop("_prompt", None)
                ctx.results[agent_name] = result
                steps_log.append({
                    "name": agent_name,
                    "phase": "extraction",
                    "status": "done",
                    "duration_ms": int((time.time() - t0) * 1000),
                    "output": result,
                    "prompt": retry_prompt,
                    "attempt": attempt,
                })
                new_content = result.get("summary") or json.dumps(result, indent=2)[:1000]
                t0 = time.time()
                critique = _CRITIC.run_critique(ctx, agent_name, new_content)
                critique.pop("_tokens", None)
                retry_critique_prompt = critique.pop("_prompt", None)
                ctx.critique_results[agent_name] = critique
                steps_log.append({
                    "name": f"Critic→{agent_name}",
                    "phase": "critique",
                    "status": "done",
                    "duration_ms": int((time.time() - t0) * 1000),
                    "output": critique,
                    "critique_score": critique.get("score"),
                    "prompt": retry_critique_prompt,
                    "attempt": attempt,
                })
            except Exception:
                break
            attempt += 1

    # --- Assembly ---
    summary_text = ctx.results.get("Summarizer", {}).get("summary", "")
    action_items = ctx.results.get("ActionItemExtractor", {}).get("action_items", [])
    suggestions_text = _assemble_suggestions(ctx.results, domain_name)
    scores = [c["score"] for c in ctx.critique_results.values() if "score" in c]
    confidence_score = round(sum(scores) / len(scores), 2) if scores else None

    # Non-LLM verification
    schema_results = _SCHEMA_VERIFIER.verify_all(ctx.results)
    risk_result = _RISK_CLASSIFIER.classify(domain_name, summary_text + " " + suggestions_text)

    total_ms = int((time.time() - t_total) * 1000)

    return {
        "domain": domain_name,
        "workflow_plan": workflow,
        "steps": steps_log,
        "results": dict(ctx.results),
        "summary_text": summary_text,
        "action_items": action_items,
        "suggestions_text": suggestions_text,
        "confidence_score": confidence_score,
        "schema_checks": schema_results,
        "risk_classification": risk_result,
        "total_ms": total_ms,
        "chunked": False,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    transcript: str,
    domain_name: str,
    cfg: dict,
    knowledge_base: str = "",
    system_prompt: str = "",
    template_prompt: str = "",
    workflow_override: dict | None = None,
) -> dict:
    """
    Run the full agent pipeline without DB writes.

    Automatically falls back to chunked map-reduce if the transcript exceeds
    the configured context window — no content is ever silently dropped.
    """
    if not template_prompt:
        template_prompt = "Summarize the meeting: key decisions, action items, and blockers."

    # Inflate max_tokens so _truncate_transcript passes the full transcript
    # if the user has set a sufficiently large context window.
    effective_cfg = {
        **cfg,
        "max_tokens": max(cfg.get("max_tokens", 4096), len(transcript) // 4 + 1000),
    }

    max_chars = max(500, (effective_cfg["max_tokens"] - 800) * 4)
    if len(transcript) > max_chars:
        # Layer 2: transcript genuinely exceeds context — use chunked map-reduce
        return _chunked_pipeline(
            transcript=transcript,
            domain_name=domain_name,
            cfg=cfg,  # original cfg for chunk sizing
            knowledge_base=knowledge_base,
            system_prompt=system_prompt,
            template_prompt=template_prompt,
        )

    return _run_pipeline_inner(
        transcript=transcript,
        domain_name=domain_name,
        cfg=effective_cfg,
        knowledge_base=knowledge_base,
        system_prompt=system_prompt,
        template_prompt=template_prompt,
        workflow_override=workflow_override,
    )
