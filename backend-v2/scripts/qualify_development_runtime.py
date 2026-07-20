from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import lmstudio

from notes_agent_v2.evaluation.development_qualification import (
    QualificationObservation,
    build_development_authorization,
    qualification_schedule,
)
from notes_agent_v2.runtime.budget import RunBudget
from notes_agent_v2.runtime.context import (
    LMStudioSDKPromptTokenizer,
    get_loaded_lm_studio_model,
)
from notes_agent_v2.runtime.contracts import RuntimeReport
from notes_agent_v2.runtime.gateway import GatewayDependencies, GatewayRequest, RuntimeGateway
from notes_agent_v2.runtime.http_provider import OpenAICompatibleRuntimeProvider
from notes_agent_v2.runtime.lm_studio import LMStudioControlClient
from notes_agent_v2.runtime.profiles import ProfileCatalog
from notes_agent_v2.runtime.settings import load_runtime_settings
from notes_agent_v2.runtime.tools import ToolDefinition, ToolPolicy, ToolSession


STRUCTURED_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"status": {"type": "string", "enum": ["ok"]}, "marker": {"type": "string"}},
    "required": ["status", "marker"],
}
CRITIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"critical": {"type": "boolean"}},
    "required": ["critical"],
}
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "lookup_qualification",
        "description": "Return the value for one authorized qualification key.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
}


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parsed(content: str) -> dict[str, object]:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("structured qualification output is not an object")
    return value


def _tool_schema(case_id: str) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "value": {"type": "string", "enum": [f"VALUE-{case_id.upper()}"]}
        },
        "required": ["value"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualify the configured development runtime.")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--runtime-authorization", type=Path, required=True)
    parser.add_argument("--development-set-authorization", type=Path, required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    parser.add_argument("--allow-live", action="store_true")
    args = parser.parse_args()
    if not args.allow_live:
        parser.error("live qualification requires --allow-live")
    if args.private_out.exists():
        parser.error("private output already exists")

    settings = load_runtime_settings(env_file=args.env_file)
    runtime_report = RuntimeReport.model_validate_json(args.runtime_authorization.read_text())
    development = json.loads(args.development_set_authorization.read_text())
    if development.get("status") != "development_set_qualified":
        raise ValueError("development set is not qualified")

    host = urlparse(str(settings.control_base_url)).netloc
    sdk_client = lmstudio.Client(host)
    model = get_loaded_lm_studio_model(sdk_client)
    tokenizer = LMStudioSDKPromptTokenizer(
        model,
        model_key=runtime_report.identity.model_key,
        instance_id=runtime_report.identity.instance_id,
        loaded_context=runtime_report.identity.loaded_context,
    )
    records: list[dict[str, object]] = []
    control = LMStudioControlClient(
        str(settings.control_base_url).rstrip("/"),
        api_token=settings.api_token,
        timeout_seconds=settings.control_timeout_seconds,
    )
    gateway = RuntimeGateway(
        GatewayDependencies(
            control=control,
            runtime_report=lambda: runtime_report,
            tokenizer=tokenizer,
            provider=OpenAICompatibleRuntimeProvider(
                base_url=str(settings.inference_base_url), api_token=settings.api_token
            ),
            profiles=ProfileCatalog.from_path(settings.profiles_path),
            expected_model=settings.model,
            context_envelope=settings.context,
            inference_timeout_seconds=settings.inference_timeout_seconds,
            record=records.append,
        )
    )
    budget = RunBudget(max_model_requests=40, max_tool_calls=9)
    observations: list[QualificationObservation] = []
    try:
        for case in qualification_schedule():
            before = budget.model_requests
            request = GatewayRequest(
                run_id="development-runtime-qualification",
                stage=case.kind,
                role="critic" if case.kind.startswith("critic") else case.kind,
                profile_name=(
                    "narrative_reasoned"
                    if case.kind == "narrative"
                    else "tool_reasoned"
                    if case.kind == "tool"
                    else "critic_structured_off"
                    if case.kind.startswith("critic")
                    else "structured_off"
                ),
                messages=({"role": "user", "content": _prompt(case.case_id, case.kind)},),
                tools=(TOOL_DEFINITION,) if case.kind == "tool" else (),
                output_schema=(
                    _tool_schema(case.case_id)
                    if case.kind == "tool"
                    else CRITIC_SCHEMA
                    if case.kind.startswith("critic")
                    else STRUCTURED_SCHEMA
                    if case.kind == "structured"
                    else None
                ),
            )
            tool_session = _tool_session(case.case_id, tokenizer) if case.kind == "tool" else None
            result = gateway.call(request, budget=budget, tools=tool_session)
            passed = _validate(case.case_id, case.kind, result.response.final_content)
            observations.append(
                QualificationObservation(
                    case_id=case.case_id,
                    passed=passed,
                    provider_requests=budget.model_requests - before,
                )
            )
            if not passed:
                raise ValueError(f"qualification failed for {case.case_id}")
    finally:
        sdk_client.close()

    fixtures = [(case.case_id, case.kind) for case in qualification_schedule()]
    authorization = build_development_authorization(
        observations,
        runtime_fingerprint=runtime_report.fingerprint,
        profile_fingerprint=hashlib.sha256(settings.profiles_path.read_bytes()).hexdigest(),
        prompt_fingerprint=digest([_prompt(case.case_id, case.kind) for case in qualification_schedule()]),
        schema_fingerprint=digest(
            [
                STRUCTURED_SCHEMA,
                *[_tool_schema(case.case_id) for case in qualification_schedule() if case.kind == "tool"],
                CRITIC_SCHEMA,
            ]
        ),
        fixture_fingerprint=digest([development["tree_sha256"], fixtures]),
        probe_requests=9,
    )
    args.private_out.mkdir(parents=True)
    payload = {
        "authorization": authorization.model_dump(mode="json"),
        "observations": [item.model_dump(mode="json") for item in observations],
        "records": records,
        "budget": budget.model_dump(mode="json"),
    }
    (args.private_out / "authorization.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(authorization.status)
    return 0


def _prompt(case_id: str, kind: str) -> str:
    marker = case_id.upper()
    if kind == "structured":
        return f"Return status ok and marker {marker} in the required object."
    if kind == "narrative":
        return f"Write one concise sentence containing the exact factual marker {marker}."
    if kind == "tool":
        return f"Call lookup_qualification with key {case_id}, then return its exact value."
    if kind == "critic_injected":
        return "Classify as critical: the candidate says the budget increased, but the source says it decreased."
    return "Classify as not critical: candidate and source both say the approved budget is 12 dollars."


def _validate(case_id: str, kind: str, content: str) -> bool:
    if kind == "narrative":
        return case_id.upper() in content and "<think" not in content.lower()
    value = parsed(content)
    if kind == "structured":
        return value.get("status") == "ok" and value.get("marker") == case_id.upper()
    if kind == "tool":
        return value.get("value") == f"VALUE-{case_id.upper()}"
    return value.get("critical") is (kind == "critic_injected")


def _tool_session(case_id: str, tokenizer: LMStudioSDKPromptTokenizer) -> ToolSession:
    return ToolSession(
        policy=ToolPolicy(
            run_id="development-runtime-qualification",
            stage="tool",
            allowed_tools=frozenset({"lookup_qualification"}),
            allowed_entity_ids=frozenset({case_id}),
            max_rounds=1,
            max_calls=1,
            max_result_tokens=64,
        ),
        definitions={
            "lookup_qualification": ToolDefinition(
                name="lookup_qualification",
                allowed_arguments=frozenset({"key"}),
                entity_fields=("key",),
                handler=lambda _arguments: f"VALUE-{case_id.upper()}",
            )
        },
        count_tokens=lambda content: tokenizer.count_tokens(content),
    )


if __name__ == "__main__":
    raise SystemExit(main())
