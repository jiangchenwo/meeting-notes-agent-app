"""Offline eval CLI for the agent pipeline.

Runs the DB-free pipeline over hand-authored eval cases against the live LLM
endpoint from lm_config, scores the outputs, and prints a report.

    uv run --group eval python -m eval                     # all hand-authored cases
    uv run --group eval python -m eval --list
    uv run --group eval python -m eval --domain General
    uv run --group eval python -m eval --case general-budget --judge
    uv run --group eval python -m eval --json-out report.json
    uv run --group eval python -m eval --public all        # MeetingBank + QMSum samples
    uv run --group eval python -m eval --public qmsum --limit 3
    uv run --group eval python -m eval --public all --baseline   # vanilla single-call baseline
    uv run --group eval python -m eval --case general-budget --trace   # spans -> local Phoenix

Public datasets need a one-time `uv run python -m eval.download_datasets` first.
Exits nonzero when a case errors or the SummaryProduced assertion fails.
"""
import argparse
import json
import logging
import sys

import lm_config
import telemetry
from agents.context import NoteDeps
from agents.pipeline import _truncate_transcript, run_pipeline
from agents.workflow_spec import select_workflow

from .cases import get_cases
from .dataset import build_dataset
from .public_cases import get_public_cases

EVAL_TEMPLATE_PROMPT = "Summarize the meeting: key decisions, action items, and blockers."


def pipeline_task(inputs: dict) -> dict:
    cfg = lm_config.load()
    deps = NoteDeps(
        note_id=0,
        domain_name=inputs["domain"],
        template_name="Eval",
        template_prompt=EVAL_TEMPLATE_PROMPT,
        global_system_prompt=cfg.get("global_system_prompt") or "",
    )
    spec = select_workflow(inputs["domain"], None)
    # One Phoenix trace per case (no-op when tracing is off).
    with telemetry.trace_span(f"eval {inputs.get('case_id', 'case')}", "eval"):
        result = run_pipeline(
            transcript=inputs["transcript"], spec=spec, deps=deps, cfg=cfg
        )
    return {
        "summary_text": result.summary_text,
        "action_items": result.action_items,
        "suggestions_text": result.suggestions_text,
        "results": {
            name: {k: v for k, v in r.items() if not k.startswith("_")}
            for name, r in result.results.items()
        },
        "confidence_score": result.confidence_score,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }


def baseline_task(inputs: dict) -> dict:
    """Vanilla baseline: one plain-text LLM call — no agents, critic, or verifiers.

    Same model, settings, and template prompt as the pipeline, so score deltas
    measure the workflow itself, not the model.
    """
    from pydantic_ai import Agent

    from agents.llm import build_model, build_model_settings

    cfg = lm_config.load()
    instructions = (cfg.get("global_system_prompt") or "").strip()
    agent = Agent(name="VanillaBaseline", instructions=instructions or None)
    transcript = _truncate_transcript(inputs["transcript"], cfg)
    with telemetry.trace_span(f"baseline {inputs.get('case_id', 'case')}", "eval"):
        run_result = agent.run_sync(
            f"{EVAL_TEMPLATE_PROMPT}\n\nTranscript:\n{transcript}",
            model=build_model(cfg),
            model_settings=build_model_settings(cfg),
        )
    return {
        "summary_text": str(run_result.output),
        "action_items": [],
        "suggestions_text": "",
        "results": {},
        "confidence_score": None,
        "input_tokens": run_result.usage.input_tokens or 0,
        "output_tokens": run_result.usage.output_tokens or 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval", description="Offline eval harness for the agent pipeline"
    )
    parser.add_argument("--domain", help="Only cases for this domain (General, Education, Healthcare, Interview, Project)")
    parser.add_argument("--case", help="Run a single case by id")
    parser.add_argument("--judge", action="store_true", help="Add an LLM-as-judge evaluator (extra LLM calls)")
    parser.add_argument("--public", choices=["meetingbank", "qmsum", "all"],
                        help="Run downloaded public-dataset cases instead of the hand-authored ones")
    parser.add_argument("--limit", type=int, help="Max cases per public dataset")
    parser.add_argument("--baseline", action="store_true",
                        help="Run a vanilla single-LLM-call baseline instead of the agent pipeline")
    parser.add_argument("--trace", action="store_true",
                        help="Send every LLM call (prompts, outputs, latency, tokens) to the local "
                             "Arize Phoenix from Settings -> Tracing, one trace per case "
                             "(start it with: uvx arize-phoenix serve; UI at http://localhost:6006)")
    parser.add_argument("--json-out", help="Write the report as JSON to this path")
    parser.add_argument("--list", action="store_true", help="List available cases and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show pipeline step logs")
    args = parser.parse_args()

    if args.list:
        cases = (
            get_public_cases(args.public, args.limit) if args.public else get_cases(args.domain)
        )
        for c in cases:
            print(f"{c.id:26s} {c.domain:12s} {c.title}")
        return 0

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    # Tracing: --trace forces it on for this run; otherwise the persisted
    # Settings -> Tracing toggle applies, same as app workflow runs.
    if telemetry.configure_telemetry(force_enable=args.trace):
        print("tracing -> local Phoenix (UI: http://localhost:6006, one trace per case)")

    dataset = build_dataset(
        args.domain, args.case, judge=args.judge, public=args.public, limit=args.limit
    )
    task, run_name = (
        (baseline_task, "vanilla-baseline") if args.baseline else (pipeline_task, "agent-pipeline")
    )
    try:
        # Serial: LM Studio loads one model at a time.
        report = dataset.evaluate_sync(task, name=run_name, max_concurrency=1)
    finally:
        telemetry.shutdown_telemetry()  # flush batched spans before exiting
    report.print(include_averages=True)

    if args.json_out:
        payload = {
            "task": run_name,
            "cases": [
                {
                    "name": rc.name,
                    "scores": {k: v.value for k, v in rc.scores.items()},
                    "assertions": {k: v.value for k, v in rc.assertions.items()},
                    "task_duration": round(rc.task_duration, 1),
                    "output": rc.output,
                }
                for rc in report.cases
            ],
            "failures": [
                {"name": f.name, "error": getattr(f, "error_message", None) or repr(f)}
                for f in report.failures
            ],
        }
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json_out}")

    failed_assertions = any(
        not v.value for rc in report.cases for v in rc.assertions.values()
    )
    return 1 if (report.failures or failed_assertions) else 0


if __name__ == "__main__":
    sys.exit(main())
