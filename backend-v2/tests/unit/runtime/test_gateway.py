from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from notes_agent_v2.runtime.budget import RunBudget
from notes_agent_v2.runtime.context import ContextEnvelope
from notes_agent_v2.runtime.contracts import (
    CapabilityProbe,
    ModelIdentity,
    ProbeStatus,
    RuntimeCapabilities,
    RuntimeReadiness,
    RuntimeReport,
)
from notes_agent_v2.runtime.gateway import (
    GatewayDependencies,
    GatewayError,
    GatewayRequest,
    RuntimeGateway,
)
from notes_agent_v2.runtime.lm_studio import EXPECTED_MODEL, LMStudioControlClient
from notes_agent_v2.runtime.profiles import ProfileCatalog
from notes_agent_v2.runtime.tools import ToolDefinition, ToolPolicy, ToolSession


ROOT = Path(__file__).parents[3]


class Tokenizer:
    model_key = EXPECTED_MODEL.model_key
    instance_id = "loaded-1"
    exact = True

    def render_chat(self, messages, tools=None, output_schema=None) -> str:
        return "safe rendered prompt"

    def count_tokens(self, rendered_prompt: str) -> int:
        return 3


class Provider:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, Any]] = []

    def complete(self, **payload: Any) -> dict[str, Any]:
        self.payloads.append(payload)
        return self.responses.pop(0)


def identity() -> ModelIdentity:
    return ModelIdentity(
        model_key=EXPECTED_MODEL.model_key,
        display_name="Gemma 4 26B A4B QAT",
        instance_id="loaded-1",
        architecture="gemma4",
        format="mlx",
        quantization_name="4bit",
        bits_per_weight=4,
        loaded_context=40960,
        maximum_context=131072,
        selected_variant="26b-a4b-qat",
    )


def runtime_report() -> RuntimeReport:
    probes = tuple(
        CapabilityProbe(name=name, status=ProbeStatus.passed, latency_ms=1)
        for name in (
            "system",
            "reasoning",
            "schema",
            "tool_request",
            "tool_round_trip",
            "tool_rejection",
            "reasoning_replay",
            "context",
        )
    )
    return RuntimeReport(
        schema_version="runtime-v1",
        identity=identity(),
        capabilities=RuntimeCapabilities(
            system_prompt=True,
            reasoning=True,
            tool_request=True,
            tool_round_trip=True,
            native_schema=True,
            exact_tokenizer=True,
        ),
        probes=probes,
        readiness=RuntimeReadiness.ready,
        generated_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )


def control() -> LMStudioControlClient:
    return LMStudioControlClient.from_models_response(
        {
            "data": [
                {
                    "key": EXPECTED_MODEL.model_key,
                    "display_name": "Gemma 4 26B A4B QAT",
                    "architecture": "gemma4",
                    "format": "mlx",
                    "quantization": {"name": "4bit", "bits_per_weight": 4},
                    "max_context_length": 131072,
                    "selected_variant": "26b-a4b-qat",
                    "instances": [
                        {"id": "loaded-1", "loaded": True, "context_length": 40960}
                    ],
                }
            ]
        }
    )


def gateway(provider: Provider, events: list[str], records: list[dict[str, Any]]) -> RuntimeGateway:
    return RuntimeGateway(
        GatewayDependencies(
            control=control(),
            runtime_report=lambda: runtime_report(),
            tokenizer=Tokenizer(),
            provider=provider,
            profiles=ProfileCatalog.from_path(ROOT / "config" / "profiles.json"),
            expected_model=EXPECTED_MODEL,
            context_envelope=ContextEnvelope(),
            inference_timeout_seconds=37,
            record=lambda value: records.append(value),
            event=lambda value: events.append(value),
        )
    )


def tool_session() -> ToolSession:
    return ToolSession(
        policy=ToolPolicy(
            run_id="run-1",
            stage="writer",
            allowed_tools=frozenset({"lookup"}),
            allowed_entity_ids=frozenset({"f1"}),
            max_rounds=1,
            max_calls=1,
            max_result_tokens=8,
        ),
        definitions={
            "lookup": ToolDefinition(
                name="lookup",
                allowed_arguments=frozenset({"fact_id"}),
                entity_fields=("fact_id",),
                handler=lambda args: "verified fact",
            )
        },
        count_tokens=lambda value: len(value.split()),
    )


def test_gateway_order_tool_finalization_and_reasoning_mapping() -> None:
    provider = Provider(
        [
            {
                "message": {
                    "reasoning_content": "private",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "lookup",
                                "arguments": {"fact_id": "f1"},
                            },
                        }
                    ],
                },
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
            {
                "message": {"content": "{\"status\": \"ok\"}"},
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                "finish_reason": "stop",
            },
        ]
    )
    events: list[str] = []
    records: list[dict[str, Any]] = []
    result = gateway(provider, events, records).call(
        GatewayRequest(
            run_id="run-1",
            stage="writer",
            role="writer",
            profile_name="tool_reasoned",
            messages=({"role": "user", "content": "safe"},),
            tools=({"name": "lookup"},),
            output_schema={"type": "object"},
        ),
        budget=RunBudget(),
        tools=tool_session(),
        validate=lambda content: content.startswith("{"),
    )
    assert result.response.final_content == '{"status": "ok"}'
    assert events == [
        "runtime_ready",
        "profile_resolved",
        "prompt_counted",
        "budgets_reserved",
        "provider_invoked",
        "response_normalized",
        "tools_authorized",
        "structured_finalized",
        "result_validated",
        "safe_metadata_recorded",
        "budgets_reconciled",
    ]
    assert "reasoning_effort" not in provider.payloads[0]["settings"]
    assert provider.payloads[1]["settings"]["reasoning_effort"] == "none"
    assert provider.payloads[0]["timeout"] == 37
    assert "private" not in str(provider.payloads[1])
    assert "private" not in str(records)


def test_structured_parse_retries_once() -> None:
    provider = Provider(
        [
            {"message": {"content": "bad"}, "usage": {}},
            {"message": {"content": "{\"ok\": true}"}, "usage": {}},
        ]
    )
    result = gateway(provider, [], []).call(
        GatewayRequest(
            run_id="run-1",
            stage="extract",
            role="extractor",
            profile_name="structured_off",
            messages=({"role": "user", "content": "safe"},),
            output_schema={"type": "object"},
        ),
        budget=RunBudget(),
        validate=lambda content: content.startswith("{"),
    )
    assert result.response.final_content.startswith("{")
    assert len(provider.payloads) == 2
    assert all(payload["settings"]["reasoning_effort"] == "none" for payload in provider.payloads)


def test_hidden_token_usage_is_budgeted_but_not_serialized() -> None:
    provider = Provider(
        [
            {
                "message": {"content": "safe"},
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "reasoning_tokens": 4,
                },
            }
        ]
    )
    budget = RunBudget()
    records: list[dict[str, Any]] = []
    gateway(provider, [], records).call(
        GatewayRequest(
            run_id="run-1",
            stage="write",
            role="writer",
            profile_name="narrative_reasoned",
            messages=({"role": "user", "content": "safe"},),
        ),
        budget=budget,
    )
    assert budget.output_tokens == 6
    assert "reasoning_tokens" not in str(records)


def test_critic_failure_is_typed_and_never_fabricated() -> None:
    provider = Provider(
        [
            {"message": {"content": "bad"}, "usage": {}},
            {"message": {"content": "still bad"}, "usage": {}},
        ]
    )
    with pytest.raises(GatewayError) as exc_info:
        gateway(provider, [], []).call(
            GatewayRequest(
                run_id="run-1",
                stage="critic",
                role="critic",
                profile_name="critic_structured_off",
                messages=({"role": "user", "content": "safe"},),
                output_schema={"type": "array"},
            ),
            budget=RunBudget(),
            validate=lambda content: False,
        )
    assert exc_info.value.error_code == "critic_failure"
    assert len(provider.payloads) == 2


def test_provider_generation_occurs_only_in_gateway_or_probe_script() -> None:
    offenders: list[str] = []
    allowed = {
        ROOT / "src" / "notes_agent_v2" / "runtime" / "gateway.py",
        ROOT / "scripts" / "probe_lm_studio.py",
    }
    for path in [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"complete", "post"} and path not in allowed:
                    offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_gateway_has_no_process_wide_expected_model_import() -> None:
    path = ROOT / "src" / "notes_agent_v2" / "runtime" / "gateway.py"
    tree = ast.parse(path.read_text())
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "EXPECTED_MODEL" not in imported_names


def test_gateway_does_not_depend_on_lm_studio_control_class() -> None:
    path = ROOT / "src" / "notes_agent_v2" / "runtime" / "gateway.py"
    tree = ast.parse(path.read_text())
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "LMStudioControlClient" not in imported_names


def test_all_phase_one_profiles_are_candidate_and_fail_production_closed() -> None:
    catalog = ProfileCatalog.from_path(ROOT / "config" / "profiles.json")
    assert catalog.names == {
        "narrative_reasoned",
        "tool_reasoned",
        "structured_off",
        "critic_structured_off",
    }
    for name in catalog.names:
        assert catalog.resolve(name).status == "candidate"
        with pytest.raises(RuntimeError, match="not production eligible"):
            catalog.resolve(name, production=True)
