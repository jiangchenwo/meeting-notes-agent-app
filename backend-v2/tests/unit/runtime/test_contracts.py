from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from notes_agent_v2.runtime.contracts import (
    CapabilityProbe,
    ModelIdentity,
    NormalizedResponse,
    NormalizedToolCall,
    NormalizedUsage,
    ProbeStatus,
    RuntimeCapabilities,
    RuntimeReadiness,
    RuntimeReport,
    assert_runtime_ready,
)


def identity(**overrides: object) -> ModelIdentity:
    values = {
        "model_key": "google/gemma-4-26b-a4b-qat",
        "display_name": "Gemma 4 26B A4B QAT",
        "instance_id": "gemma4-qat-1",
        "architecture": "gemma4",
        "format": "mlx",
        "quantization_name": "4bit",
        "bits_per_weight": 4,
        "loaded_context": 40960,
        "maximum_context": 131072,
        "selected_variant": "26b-a4b-qat",
    }
    values.update(overrides)
    return ModelIdentity(**values)


def capabilities(**overrides: object) -> RuntimeCapabilities:
    values = {
        "system_prompt": True,
        "reasoning": True,
        "tool_request": True,
        "tool_round_trip": True,
        "native_schema": True,
        "exact_tokenizer": True,
    }
    values.update(overrides)
    return RuntimeCapabilities(**values)


def report(**overrides: object) -> RuntimeReport:
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
    values = {
        "schema_version": "runtime-v1",
        "identity": identity(),
        "capabilities": capabilities(),
        "probes": probes,
        "readiness": RuntimeReadiness.ready,
        "generated_at": datetime(2026, 7, 16, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return RuntimeReport(**values)


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (lambda: identity(bits_per_weight=0), "bits_per_weight"),
        (lambda: identity(loaded_context=0), "loaded_context"),
        (lambda: NormalizedUsage(prompt_tokens=-1), "prompt_tokens"),
        (lambda: CapabilityProbe(name="x", status="passed", latency_ms=-1), "latency_ms"),
    ],
)
def test_contracts_reject_invalid_numbers(factory, field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        factory()


def test_contracts_are_frozen_and_forbid_extra_fields() -> None:
    model = identity()
    with pytest.raises(ValidationError):
        ModelIdentity(**model.model_dump(), secret="nope")
    with pytest.raises(ValidationError):
        model.loaded_context = 1


def test_runtime_report_fingerprint_is_stable_and_material() -> None:
    first = report(generated_at=datetime(2026, 7, 16, tzinfo=timezone.utc))
    second = report(generated_at=datetime(2026, 7, 17, tzinfo=timezone.utc))
    changed = report(identity=identity(instance_id="other"))

    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != changed.fingerprint
    assert len(first.fingerprint) == 64


def test_ready_report_rejects_failed_required_probe() -> None:
    with pytest.raises(ValidationError, match="system"):
        report(
            probes=(
                CapabilityProbe(
                    name="system",
                    status=ProbeStatus.failed,
                    latency_ms=1,
                    error_code="mismatch",
                ),
            )
        )


def test_ready_report_requires_all_eight_probes() -> None:
    with pytest.raises(ValidationError, match="missing required probes"):
        report(probes=())


def test_blocked_runtime_fails_closed() -> None:
    blocked = report(readiness=RuntimeReadiness.blocked, probes=())
    with pytest.raises(RuntimeError, match="blocked"):
        assert_runtime_ready(blocked)


def test_normalized_response_contains_only_safe_contract_fields() -> None:
    response = NormalizedResponse(
        final_content="safe",
        tool_calls=(NormalizedToolCall(call_id="c1", name="lookup", arguments={"id": "f1"}),),
        usage=NormalizedUsage(prompt_tokens=2, output_tokens=1),
        reasoning_observed=True,
    )
    keys = set(response.model_dump(mode="json"))

    assert keys == {
        "final_content",
        "tool_calls",
        "usage",
        "finish_reason",
        "reasoning_observed",
    }
