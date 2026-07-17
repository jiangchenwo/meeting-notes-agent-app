from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import httpx

from notes_agent_v2.evaluation.judge_calibration import (
    estimate_calibration_budget,
    load_calibration_suite,
    run_judge_calibration,
    validate_fixed_suite,
    write_calibration_bundle,
)
from notes_agent_v2.evaluation.judge_settings import load_judge_settings
from notes_agent_v2.evaluation.judges import (
    EvaluationJudgeGateway,
    JudgeQualificationStatus,
    OpenAICompatibleJudgeProvider,
)
from notes_agent_v2.evaluation.tracing import JsonlTraceRecorder, validate_trace


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed remote judge qualification suite.")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--private-out", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-remote-judge", action="store_true")
    args = parser.parse_args()

    settings = load_judge_settings(env_file=args.env_file)
    suite = load_calibration_suite(args.suite)
    validate_fixed_suite(suite)
    budget = estimate_calibration_budget(suite, settings)
    print(
        f"requests={budget.request_cap} input_tokens={budget.input_token_reservation} "
        f"output_token_cap={budget.output_token_cap} time_cap_seconds={budget.time_cap_seconds:g} "
        f"estimated_cost_usd={budget.estimated_cost_usd:.6f} configured_cost_cap_usd={settings.max_cost_usd:.2f}"
    )
    if args.preflight_only:
        return 0
    if not args.allow_remote_judge:
        parser.error("live qualification requires --allow-remote-judge")
    if args.private_out is None:
        parser.error("live qualification requires --private-out")
    if args.private_out.exists():
        parser.error("private output already exists")
    args.private_out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="judge-qualification-", dir=args.private_out.parent) as temporary:
        trace_path = Path(temporary) / "events.jsonl"
        recorder = JsonlTraceRecorder(trace_path)
        with httpx.Client(timeout=settings.timeout_seconds) as client:
            provider = OpenAICompatibleJudgeProvider(settings, client=client)
            gateway = EvaluationJudgeGateway(
                settings,
                provider,
                allow_remote_judge=True,
                trace_recorder=recorder,
            )
            report = run_judge_calibration(suite, gateway)
        trace = validate_trace(trace_path)
        if trace.request_count != budget.request_cap or gateway.accounting.requests != budget.request_cap:
            raise RuntimeError("judge qualification request accounting does not match the fixed schedule")
        manifest = write_calibration_bundle(
            args.private_out,
            suite=suite,
            report=report,
            trace_path=trace_path,
        )

    print(
        f"status={report.qualification.status.value} requests={report.accounting.requests} "
        f"estimated_cost_usd={report.accounting.estimated_cost_usd:.6f} "
        f"bundle_digest={manifest.bundle_digest}"
    )
    return 0 if report.qualification.status is JudgeQualificationStatus.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
