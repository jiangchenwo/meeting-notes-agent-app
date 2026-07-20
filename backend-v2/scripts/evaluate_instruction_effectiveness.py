from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import lmstudio

from notes_agent_v2.evaluation.artifacts import EvaluationBundleWriter, verify_bundle
from notes_agent_v2.evaluation.evidence_effectiveness import (
    validate_development_runtime_authorization,
)
from notes_agent_v2.evaluation.instruction_effectiveness import (
    InstructionObservation,
    build_instruction_effectiveness_report,
    evaluate_brief_case,
    evaluate_capability_case,
    evaluate_salience_case,
    load_instruction_cases,
)
from notes_agent_v2.evaluation.tracing import JsonlTraceRecorder, validate_trace
from notes_agent_v2.runtime.budget import RunBudget
from notes_agent_v2.runtime.context import (
    LMStudioSDKPromptTokenizer,
    get_loaded_lm_studio_model,
)
from notes_agent_v2.runtime.contracts import RuntimeReport, assert_runtime_ready
from notes_agent_v2.runtime.gateway import GatewayDependencies, RuntimeGateway
from notes_agent_v2.runtime.http_provider import OpenAICompatibleRuntimeProvider
from notes_agent_v2.runtime.lm_studio import LMStudioControlClient
from notes_agent_v2.runtime.profiles import ProfileCatalog
from notes_agent_v2.runtime.settings import load_runtime_settings
from notes_agent_v2.workflow.audience import GenerationBrief
from notes_agent_v2.workflow.planner import CapabilityPlan
from notes_agent_v2.workflow.salience import RelevancePayload


LIMITS = {
    "planning.generation_brief": 132,
    "planning.salience_selection": 88,
    "planning.closed_capability_plan": 176,
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _feature_fingerprints(feature_id: str, project_root: Path):
    workflow_name = {
        "planning.generation_brief": "audience.py",
        "planning.salience_selection": "salience.py",
        "planning.closed_capability_plan": "planner.py",
    }[feature_id]
    workflow_path = project_root / "src/notes_agent_v2/workflow" / workflow_name
    evaluator_path = project_root / "src/notes_agent_v2/evaluation/instruction_effectiveness.py"
    script_path = Path(__file__)
    registry_path = project_root / "config/evaluation/features.json"
    digest = hashlib.sha256()
    for path in (workflow_path, evaluator_path, script_path, registry_path):
        digest.update(path.relative_to(project_root).as_posix().encode())
        digest.update(path.read_bytes())
    schema = (
        GenerationBrief.model_json_schema()
        if feature_id == "planning.generation_brief"
        else RelevancePayload.model_json_schema()
        if feature_id == "planning.salience_selection"
        else CapabilityPlan.model_json_schema()
    )
    return digest.hexdigest(), hashlib.sha256(workflow_path.read_bytes()).hexdigest(), _digest(schema)


class TracingGateway:
    def __init__(self, gateway, recorder, budget) -> None:
        self.gateway = gateway
        self.recorder = recorder
        self.budget = budget

    def call(self, request, *, budget, validate):
        before = self.budget.model_requests
        with self.recorder.span(
            "model",
            feature_id=request.role,
            case_id=request.run_id,
            stage_id=request.stage,
            profile_id=request.profile_name,
        ) as span:
            try:
                result = self.gateway.call(request, budget=budget, validate=validate)
            except Exception as exc:
                span.terminal(
                    status="failed",
                    accounting={"requests": self.budget.model_requests - before},
                    error_code=type(exc).__name__,
                )
                raise
            span.terminal(
                accounting={"requests": self.budget.model_requests - before}
            )
            return result


def _markdown(report) -> str:
    lines = [
        f"# {report.feature_id}",
        "",
        f"- Verdict: `{report.verdict}`",
        f"- Cases: `{report.case_count}` (`{report.authored_case_count}` authored, `{report.public_case_count}` public)",
        f"- Provider requests: `{report.provider_requests}/{report.request_limit}`",
        f"- Runtime fingerprint: `{report.runtime_fingerprint}`",
        f"- Fixture fingerprint: `{report.fixture_fingerprint}`",
        f"- Result digest: `{report.result_digest}`",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(f"- `{name}`: `{value}`" for name, value in report.metrics.items())
    lines.extend(["", "## Hard gates", ""])
    lines.extend(
        f"- `{name}`: `{'passed' if value else 'failed'}`"
        for name, value in report.hard_gates.items()
    )
    return "\n".join(lines) + "\n"


def _write_checkpoint(path: Path, observations) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in observations],
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--runtime-authorization", required=True, type=Path)
    parser.add_argument("--development-set-authorization", required=True, type=Path)
    parser.add_argument("--development-runtime-authorization", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--feature", required=True, choices=tuple(LIMITS))
    parser.add_argument("--private-out", required=True, type=Path)
    parser.add_argument("--allow-live", action="store_true")
    args = parser.parse_args()
    if not args.allow_live:
        parser.error("live model calls require --allow-live")
    if args.private_out.exists():
        parser.error("private output already exists")

    settings = load_runtime_settings(env_file=args.env_file)
    runtime_report = RuntimeReport.model_validate_json(
        args.runtime_authorization.read_text()
    )
    assert_runtime_ready(runtime_report)
    development = json.loads(args.development_set_authorization.read_text())
    if development.get("status") != "development_set_qualified":
        raise ValueError("development set is not qualified")
    profile_fingerprint = hashlib.sha256(settings.profiles_path.read_bytes()).hexdigest()
    development_runtime = json.loads(
        args.development_runtime_authorization.read_text()
    )
    validate_development_runtime_authorization(
        development_runtime,
        runtime_fingerprint=runtime_report.fingerprint,
        profile_fingerprint=profile_fingerprint,
    )
    cases, development_fingerprint, fixture_fingerprint = load_instruction_cases(
        args.cases
    )
    code_fingerprint, prompt_fingerprint, schema_fingerprint = _feature_fingerprints(
        args.feature, Path(__file__).resolve().parents[1]
    )
    if development_fingerprint != development.get("tree_sha256"):
        raise ValueError("instruction cases do not match development authorization")

    work = args.private_out.parent / f".{args.private_out.name}.live-work"
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / "observations.json"
    observations = (
        [
            InstructionObservation.model_validate(item)
            for item in json.loads(checkpoint.read_text())
        ]
        if checkpoint.exists()
        else []
    )
    if any(item.feature_id != args.feature for item in observations):
        raise ValueError("checkpoint feature mismatch")
    completed = {item.case_id for item in observations}
    trace_path = work / "events.jsonl"
    recorder = JsonlTraceRecorder(
        trace_path, trace_id=f"live-{args.feature.replace('.', '-')}"
    )
    records_path = work / "safe-runtime-records.jsonl"
    records: list[dict[str, object]] = []

    host = urlparse(str(settings.control_base_url)).netloc
    sdk_client = lmstudio.Client(host)
    try:
        model = get_loaded_lm_studio_model(sdk_client)
        tokenizer = LMStudioSDKPromptTokenizer(
            model,
            model_key=runtime_report.identity.model_key,
            instance_id=runtime_report.identity.instance_id,
            loaded_context=runtime_report.identity.loaded_context,
        )

        def record(value):
            records.append(value)
            with records_path.open("a") as handle:
                handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")

        raw_gateway = RuntimeGateway(
            GatewayDependencies(
                control=LMStudioControlClient(
                    str(settings.control_base_url).rstrip("/"),
                    api_token=settings.api_token,
                    timeout_seconds=settings.control_timeout_seconds,
                ),
                runtime_report=lambda: runtime_report,
                tokenizer=tokenizer,
                provider=OpenAICompatibleRuntimeProvider(
                    base_url=str(settings.inference_base_url),
                    api_token=settings.api_token,
                ),
                profiles=ProfileCatalog.from_path(settings.profiles_path),
                expected_model=settings.model,
                context_envelope=settings.context,
                inference_timeout_seconds=settings.inference_timeout_seconds,
                record=record,
            )
        )
        budget = RunBudget(
            max_model_requests=LIMITS[args.feature],
            max_input_tokens=20_000_000,
            max_output_tokens=2_000_000,
            max_wall_seconds=14_400,
        )
        prior_requests = sum(item.provider_requests for item in observations)
        budget.model_requests = prior_requests
        gateway = TracingGateway(raw_gateway, recorder, budget)
        evaluator = (
            evaluate_brief_case
            if args.feature == "planning.generation_brief"
            else evaluate_salience_case
            if args.feature == "planning.salience_selection"
            else evaluate_capability_case
        )
        for case in cases:
            if case.case_id in completed:
                continue
            with recorder.span(
                "evaluator",
                feature_id=args.feature,
                case_id=case.case_id,
                cohort_id=case.cohort,
                fixture_id=fixture_fingerprint,
            ) as span:
                observation = evaluator(case, gateway, budget)
                span.terminal(
                    status="passed" if observation.valid else "invalid",
                    accounting={"requests": 0},
                    error_code=observation.error_code,
                )
            observations.append(observation)
            _write_checkpoint(checkpoint, observations)
            print(
                f"{args.feature} {len(observations)}/44 {case.case_id} "
                f"valid={observation.valid} requests={budget.model_requests}",
                flush=True,
            )
    finally:
        sdk_client.close()

    report = build_instruction_effectiveness_report(
        args.feature,
        observations,
        runtime_fingerprint=runtime_report.fingerprint,
        profile_fingerprint=profile_fingerprint,
        fixture_fingerprint=fixture_fingerprint,
        development_fingerprint=development_fingerprint,
        code_fingerprint=code_fingerprint,
        prompt_fingerprint=prompt_fingerprint,
        schema_fingerprint=schema_fingerprint,
        request_limit=LIMITS[args.feature],
    )
    with recorder.span(
        "report", feature_id=args.feature, fingerprint=report.result_digest
    ) as span:
        span.terminal(
            status="passed" if report.verdict == "passed" else "invalid",
            accounting={"requests": 0},
            artifact_digests={"results": report.result_digest},
        )
    validate_trace(trace_path)
    writer = EvaluationBundleWriter(
        args.private_out,
        run_id=f"{args.feature}-live",
        fingerprint=report.result_digest,
    )
    writer.write_json("report.json", report.model_dump(mode="json"))
    writer.write_text("report.md", _markdown(report))
    writer.write_json(
        "observations.json",
        [item.model_dump(mode="json") for item in observations],
    )
    writer.write_json(
        "input-fingerprints.json",
        {
            "runtime": runtime_report.fingerprint,
            "profiles": profile_fingerprint,
            "fixtures": fixture_fingerprint,
            "development": development_fingerprint,
            "code": code_fingerprint,
            "prompt": prompt_fingerprint,
            "schema": schema_fingerprint,
        },
    )
    writer.write_text(
        "safe-runtime-records.jsonl",
        records_path.read_text() if records_path.exists() else "",
    )
    writer.write_text("events.jsonl", trace_path.read_text())
    manifest = writer.seal()
    verify_bundle(args.private_out)
    print(
        f"verdict={report.verdict} requests={report.provider_requests} "
        f"result={report.result_digest} bundle={manifest.bundle_digest}"
    )
    return 0 if report.verdict == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
