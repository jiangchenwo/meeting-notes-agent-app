from __future__ import annotations

from datetime import datetime, timezone
import ast
import importlib.util
from pathlib import Path

import pytest

from notes_agent_v2.runtime.lm_studio import ExpectedModel

from notes_agent_v2.runtime.contracts import (
    CapabilityProbe,
    ModelIdentity,
    ProbeStatus,
    RuntimeCapabilities,
    RuntimeReadiness,
)
from notes_agent_v2.runtime.probes import (
    PROBE_NAMES,
    ScriptedProbeRunner,
    build_public_report,
    run_capability_probes,
)


def identity() -> ModelIdentity:
    return ModelIdentity(
        model_key="google/gemma-4-26b-a4b-qat",
        display_name="Gemma 4 QAT",
        instance_id="i1",
        architecture="gemma4",
        format="mlx",
        quantization_name="4bit",
        bits_per_weight=4,
        loaded_context=40960,
        maximum_context=131072,
        selected_variant="qat",
    )


def capabilities() -> RuntimeCapabilities:
    return RuntimeCapabilities(
        system_prompt=True,
        reasoning=True,
        tool_request=True,
        tool_round_trip=True,
        native_schema=True,
        exact_tokenizer=True,
    )


def test_all_eight_probes_run_in_order() -> None:
    runner = ScriptedProbeRunner(
        {
            name: CapabilityProbe(
                name=name,
                status=ProbeStatus.passed,
                latency_ms=index,
                observed={"raw_prompt": "must not reach public report"},
            )
            for index, name in enumerate(PROBE_NAMES)
        }
    )
    report = run_capability_probes(
        identity(), capabilities(), runner, now=datetime(2026, 7, 16, tzinfo=timezone.utc)
    )
    assert tuple(probe.name for probe in report.probes) == PROBE_NAMES
    assert report.readiness is RuntimeReadiness.ready

    public = build_public_report(report)
    assert "raw_prompt" not in str(public)
    assert "observed" not in str(public)
    assert public["probe_count"] == 8


def test_any_required_probe_failure_blocks_runtime() -> None:
    script = {
        name: CapabilityProbe(name=name, status=ProbeStatus.passed, latency_ms=1)
        for name in PROBE_NAMES
    }
    script["tool_rejection"] = CapabilityProbe(
        name="tool_rejection",
        status=ProbeStatus.failed,
        latency_ms=1,
        error_code="unauthorized_tool_executed",
    )
    report = run_capability_probes(
        identity(), capabilities(), ScriptedProbeRunner(script)
    )
    assert report.readiness is RuntimeReadiness.blocked
    assert build_public_report(report)["readiness"] == "blocked"


def test_probe_runner_exception_becomes_typed_blocked_probe() -> None:
    class BrokenRunner:
        def run_probe(self, name: str) -> CapabilityProbe:
            if name == "schema":
                raise TimeoutError("provider stalled")
            return CapabilityProbe(name=name, status=ProbeStatus.passed, latency_ms=1)

    report = run_capability_probes(identity(), capabilities(), BrokenRunner())
    failed = next(probe for probe in report.probes if probe.name == "schema")
    assert report.readiness is RuntimeReadiness.blocked
    assert failed.error_code == "TimeoutError"


def test_probe_script_validates_authorization_and_keeps_trace_outside_repo(tmp_path: Path) -> None:
    script_path = Path(__file__).parents[3] / "scripts" / "probe_lm_studio.py"
    spec = importlib.util.spec_from_file_location("probe_lm_studio", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    authorization = tmp_path / "runtime.json"
    authorization.write_text(
        '{"model":{"model_key":"google/gemma-4-26b-a4b-qat",'
        '"architecture":"gemma4","format":"mlx","bits_per_weight":4,'
        '"loaded_context":40960}}'
    )
    module.validate_runtime_authorization(
        authorization,
        ExpectedModel(
            model_key="google/gemma-4-26b-a4b-qat",
            architecture="gemma4",
            format="mlx",
            quantization_name="4bit",
            bits_per_weight=4,
            loaded_context=40960,
        ),
    )

    with pytest.raises(ValueError, match="private trace"):
        module.require_private_trace_path(Path(__file__).parents[3] / "trace.json")
    outside = module.require_private_trace_path(tmp_path / "trace.json")
    assert outside == (tmp_path / "trace.json").resolve()


def test_probe_script_has_no_process_wide_expected_model_import() -> None:
    script_path = Path(__file__).parents[3] / "scripts" / "probe_lm_studio.py"
    tree = ast.parse(script_path.read_text())
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "EXPECTED_MODEL" not in imported_names


def _load_probe_script():
    script_path = Path(__file__).parents[3] / "scripts" / "probe_lm_studio.py"
    spec = importlib.util.spec_from_file_location("probe_lm_studio_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _live_runner(module):
    return module.LiveProbeRunner(
        base_url="http://runtime.test/v1",
        model_key="model-key",
        api_token=None,
        timeout_seconds=1,
        loaded_context=40960,
    )


def test_system_probe_disables_reasoning_for_exact_instruction_check() -> None:
    module = _load_probe_script()
    runner = _live_runner(module)
    payloads: list[dict[str, object]] = []

    def respond(payload):
        payloads.append(payload)
        instruction = payload["messages"][0]["content"]
        return {"content": instruction.removeprefix("Reply with exactly ").removesuffix(".")}

    runner._chat = respond

    assert runner._system() == {"matched": True}
    assert payloads[0]["reasoning_effort"] == "none"


def test_reasoning_probe_reserves_enough_tokens_for_a_final_answer() -> None:
    module = _load_probe_script()
    runner = _live_runner(module)
    payloads: list[dict[str, object]] = []

    def respond(payload):
        payloads.append(payload)
        sentinel = payload["messages"][0]["content"].removeprefix("Reply with exactly ").removesuffix(".")
        return {"content": sentinel, "reasoning_content": "reasoning observed"}

    runner._chat = respond

    assert runner._reasoning() == {"reasoning_observed": True}
    assert payloads[0]["max_tokens"] == 512


def test_reasoning_replay_reserves_enough_tokens_for_safe_final_content() -> None:
    module = _load_probe_script()
    runner = _live_runner(module)
    payloads: list[dict[str, object]] = []

    def respond(payload):
        payloads.append(payload)
        return {"content": "REPLAY-OK", "reasoning_content": "reasoning observed"}

    runner._chat = respond

    assert runner._reasoning_replay() == {"safe_final": True}
    assert payloads[0]["max_tokens"] == 512


def test_live_probe_trace_records_only_safe_exchange_metadata(monkeypatch) -> None:
    module = _load_probe_script()
    runner = _live_runner(module)

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "model": "model-key",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "PRIVATE RESPONSE",
                            "reasoning_content": "PRIVATE REASONING",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            }

    class Client:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(module.httpx, "Client", Client)
    runner._chat({"messages": [{"role": "user", "content": "PRIVATE PROMPT"}]})

    serialized = str(runner.trace)
    assert "PRIVATE PROMPT" not in serialized
    assert "PRIVATE RESPONSE" not in serialized
    assert "PRIVATE REASONING" not in serialized
    assert runner.trace == [
        {
            "finish_reason": "stop",
            "model": "model-key",
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
            "reasoning_observed": True,
            "tool_call_count": 0,
        }
    ]
