"""Builds the pydantic-evals Dataset from hand-authored or public EvalCases."""
from pydantic_evals import Case, Dataset

from .cases import get_cases
from .evaluators import (
    ActionRecall,
    FactCoverage,
    HallucinationFlags,
    PipelineConfidence,
    ReferenceAlignment,
    SummaryProduced,
)
from .public_cases import get_public_cases

JUDGE_RUBRIC = (
    "The output contains meeting notes generated from the transcript in the input. "
    "Judge whether the notes faithfully capture the meeting: all key topics, decisions, "
    "and commitments are present; nothing is invented; concrete names, numbers, and "
    "dates from the transcript are preserved rather than replaced with vague language."
)


def build_dataset(
    domain: str | None = None,
    case_id: str | None = None,
    judge: bool = False,
    public: str | None = None,
    limit: int | None = None,
) -> Dataset:
    if public:
        cases = get_public_cases(public, limit)
        if domain:
            cases = [c for c in cases if c.domain.lower() == domain.lower()]
    else:
        cases = get_cases(domain)
    if case_id:
        cases = [c for c in cases if c.id == case_id]
    if not cases:
        raise SystemExit(
            f"no eval cases match domain={domain!r} case={case_id!r} public={public!r}"
        )

    evaluators = [
        SummaryProduced(),
        FactCoverage(),
        ActionRecall(),
        ReferenceAlignment(),
        HallucinationFlags(),
        PipelineConfidence(),
    ]
    if judge:
        import lm_config
        from agents.llm import build_model, build_model_settings
        from pydantic_evals.evaluators import LLMJudge

        cfg = lm_config.load()
        evaluators.append(
            LLMJudge(
                rubric=JUDGE_RUBRIC,
                model=build_model(cfg),
                include_input=True,
                model_settings=build_model_settings(cfg),
            )
        )

    return Dataset(
        name="meeting-notes-pipeline",
        cases=[
            Case(
                name=c.id,
                inputs={"case_id": c.id, "transcript": c.transcript, "domain": c.domain},
                metadata=dict(c.ground_truth),
            )
            for c in cases
        ],
        evaluators=evaluators,
    )
