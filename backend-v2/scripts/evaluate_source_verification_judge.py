from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import httpx

from notes_agent_v2.evaluation.evidence_effectiveness import (
    candidate_from_reference,
    development_utterances,
    inject_polarity_defect,
)
from notes_agent_v2.evaluation.judge_settings import load_judge_settings
from notes_agent_v2.evaluation.judges import (
    EvaluationJudgeGateway,
    OpenAICompatibleJudgeProvider,
)
from notes_agent_v2.evaluation.source_verification_judge import (
    SourceVerificationJudgeCase,
    estimate_source_verification_judge_budget,
    run_source_verification_judge,
    validate_judge_qualification_bundle,
    write_source_verification_judge_bundle,
)
from notes_agent_v2.evaluation.tracing import JsonlTraceRecorder, validate_trace


CASE_IDS = tuple(
    [f"ami-structured-{index:02d}" for index in range(1, 9)]
    + [f"long-context-{index:02d}" for index in range(1, 5)]
)


def _normalized_evidence_id(identifier: str) -> str:
    return f"u{int(identifier[1:]):06d}"


def _selected_reference(label: dict[str, object]) -> dict[str, object]:
    references = label.get("references")
    if not isinstance(references, list):
        raise ValueError("development label references are missing")
    selected = [
        item
        for item in references
        if isinstance(item, dict)
        and item.get("role")
        in {"decision_summary", "action_summary", "issue_summary"}
    ]
    if not selected:
        selected = [item for item in references if isinstance(item, dict)]
    if not selected:
        raise ValueError("development label has no usable reference")
    reference = selected[0]
    evidence_ids = reference.get("evidence_ids")
    if not isinstance(evidence_ids, list) or not evidence_ids:
        raise ValueError("development reference evidence IDs are missing")
    return {
        **reference,
        "evidence_ids": tuple(_normalized_evidence_id(str(item)) for item in evidence_ids),
    }


def _load_cases(root: Path) -> tuple[SourceVerificationJudgeCase, ...]:
    cases: list[SourceVerificationJudgeCase] = []
    for index, case_id in enumerate(CASE_IDS, start=1):
        case = json.loads((root / "cases" / f"{case_id}.json").read_text())
        label = json.loads((root / "labels" / f"{case_id}.json").read_text())
        if case.get("case_id") != case_id or label.get("case_id") != case_id:
            raise ValueError(f"development case identity mismatch: {case_id}")
        if case.get("transcript_sha256") != label.get("transcript_sha256"):
            raise ValueError(f"development case transcript mismatch: {case_id}")
        utterances = development_utterances(case)
        utterance_by_id = {item.id: item for item in utterances}
        reference = _selected_reference(label)
        candidate = candidate_from_reference(
            f"fc{index:06d}", reference, utterance_by_id
        )
        evidence_ids = candidate.evidence[0].utterance_ids
        source = "\n".join(
            f"[{identifier}] {utterance_by_id[identifier].text}"
            for identifier in evidence_ids
        )
        cases.extend(
            (
                SourceVerificationJudgeCase(
                    case_id=f"{case_id}-clean",
                    candidate=candidate.text,
                    reference=source,
                    evidence_ids=evidence_ids,
                    injected_error=False,
                ),
                SourceVerificationJudgeCase(
                    case_id=f"{case_id}-injected",
                    candidate=inject_polarity_defect(candidate.text),
                    reference=source,
                    evidence_ids=evidence_ids,
                    injected_error=True,
                ),
            )
        )
    return tuple(cases)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate source verification with a qualified remote judge."
    )
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--judge-qualification", type=Path, required=True)
    parser.add_argument("--development-set-authorization", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--private-out", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-remote-judge", action="store_true")
    args = parser.parse_args()

    settings = load_judge_settings(env_file=args.env_file)
    judge_authorization = validate_judge_qualification_bundle(
        args.judge_qualification, settings
    )
    development = json.loads(args.development_set_authorization.read_text())
    if development.get("status") != "development_set_qualified":
        raise ValueError("development set is not qualified")
    manifest = json.loads((args.development_root / "manifest.json").read_text())
    if (
        manifest.get("selection_digest") != development.get("selection_digest")
        or manifest.get("case_count") != development.get("case_count")
    ):
        raise ValueError("development authorization does not match the selected root")
    development_fingerprint = str(development.get("tree_sha256", ""))
    if len(development_fingerprint) != 64:
        raise ValueError("development authorization fingerprint is invalid")
    cases = _load_cases(args.development_root)
    if len(cases) != 24 or sum(case.injected_error for case in cases) != 12:
        raise ValueError("source verification judge schedule must contain 12 clean and 12 injected cases")
    budget = estimate_source_verification_judge_budget(cases, settings)
    print(
        f"requests={budget.request_cap} input_tokens={budget.input_token_reservation} "
        f"output_token_cap={budget.output_token_cap} "
        f"time_cap_seconds={budget.time_cap_seconds:g} "
        f"estimated_cost_usd={budget.estimated_cost_usd:.6f} "
        f"configured_cost_cap_usd={settings.max_cost_usd:.2f}"
    )
    if args.preflight_only:
        return 0
    if not args.allow_remote_judge:
        parser.error("live evaluation requires --allow-remote-judge")
    if args.private_out is None:
        parser.error("live evaluation requires --private-out")
    if args.private_out.exists():
        parser.error("private output already exists")
    args.private_out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="source-verification-judge-", dir=args.private_out.parent
    ) as temporary:
        trace_path = Path(temporary) / "events.jsonl"
        recorder = JsonlTraceRecorder(trace_path, trace_id="source-verification-judge")
        with httpx.Client(timeout=settings.timeout_seconds) as client:
            gateway = EvaluationJudgeGateway(
                settings,
                OpenAICompatibleJudgeProvider(settings, client=client),
                allow_remote_judge=True,
                trace_recorder=recorder,
            )
            report = run_source_verification_judge(
                cases,
                gateway,
                judge_authorization_fingerprint=judge_authorization.run_fingerprint,
                judge_qualification_bundle_digest=judge_authorization.bundle_digest,
                development_fingerprint=development_fingerprint,
            )
        trace = validate_trace(trace_path)
        if trace.request_count != budget.request_cap:
            raise RuntimeError("source verification judge trace is incomplete")
        bundle = write_source_verification_judge_bundle(
            args.private_out,
            cases=cases,
            report=report,
            trace_path=trace_path,
        )

    print(
        f"verdict={report.verdict} requests={report.accounting.requests} "
        f"estimated_cost_usd={report.accounting.estimated_cost_usd:.6f} "
        f"result_digest={report.result_digest} bundle_digest={bundle.bundle_digest}"
    )
    return 0 if report.verdict == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
