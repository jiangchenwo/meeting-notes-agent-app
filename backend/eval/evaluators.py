"""pydantic-evals evaluators wrapping the pure-Python metric helpers.

Ground truth comes from Case.metadata (the EvalCase.ground_truth dict); the
task output is the dict produced by __main__.pipeline_task.
"""
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from .metrics import (
    action_recall,
    build_full_output_text,
    coverage_score,
    hallucination_check,
    summary_alignment,
)


def _full_text(output: dict) -> str:
    return build_full_output_text(
        output.get("results", {}),
        output.get("suggestions_text", ""),
        output.get("summary_text", ""),
    )


@dataclass
class SummaryProduced(Evaluator):
    """Assertion: the pipeline produced a non-empty summary."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output.get("summary_text", "").strip())


@dataclass
class FactCoverage(Evaluator):
    """Fraction of ground-truth facts found anywhere in the output."""

    def evaluate(self, ctx: EvaluatorContext) -> dict:
        facts = (ctx.metadata or {}).get("facts", [])
        if not facts:
            return {}
        return {"fact_coverage": coverage_score(_full_text(ctx.output), facts)}


@dataclass
class ActionRecall(Evaluator):
    """Fraction of expected owners/tasks mentioned across the action items."""

    def evaluate(self, ctx: EvaluatorContext) -> dict:
        md = ctx.metadata or {}
        score = action_recall(
            ctx.output.get("action_items", []),
            md.get("action_owners", []),
            md.get("action_tasks", []),
        )
        if score is None:  # case defines no expectations — unscored, not 0
            return {}
        return {"action_recall": score}


@dataclass
class ReferenceAlignment(Evaluator):
    """Content-token recall/precision of the generated summary against the
    dataset's human reference summary (public-dataset cases only)."""

    def evaluate(self, ctx: EvaluatorContext) -> dict:
        reference = (ctx.metadata or {}).get("reference_summary", "")
        if not reference:
            return {}
        scores = summary_alignment(ctx.output.get("summary_text", ""), reference)
        if scores is None:
            return {}
        # The app's deliverable is the whole notes document (summary + action
        # items + suggestions), so also score the assembled output — this is
        # the like-for-like number against a single-call baseline, whose one
        # text blob contains everything.
        full = summary_alignment(_full_text(ctx.output), reference)
        if full:
            scores["reference_recall_full"] = full["reference_recall"]
        return scores


@dataclass
class HallucinationFlags(Evaluator):
    """Heuristic count of proper nouns in the output absent from the transcript
    (lower is better; not definitive hallucination detection)."""

    def evaluate(self, ctx: EvaluatorContext) -> dict:
        flags = hallucination_check(_full_text(ctx.output), ctx.inputs["transcript"])
        return {"hallucination_flags": float(len(flags))}


@dataclass
class PipelineConfidence(Evaluator):
    """Surfaces the pipeline's own Critic confidence score in the report."""

    def evaluate(self, ctx: EvaluatorContext) -> dict:
        c = ctx.output.get("confidence_score")
        return {} if c is None else {"pipeline_confidence": float(c)}
