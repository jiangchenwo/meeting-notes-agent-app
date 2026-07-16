from __future__ import annotations

from datetime import datetime
from enum import StrEnum
import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeReadiness(StrEnum):
    ready = "ready"
    degraded = "degraded"
    blocked = "blocked"


class ProbeStatus(StrEnum):
    passed = "passed"
    failed = "failed"


class RuntimeContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelIdentity(RuntimeContract):
    model_key: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    format: str = Field(min_length=1)
    quantization_name: str = Field(min_length=1)
    bits_per_weight: int = Field(gt=0)
    loaded_context: int = Field(gt=0)
    maximum_context: int = Field(gt=0)
    selected_variant: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_context(self) -> ModelIdentity:
        if self.loaded_context > self.maximum_context:
            raise ValueError("loaded_context must not exceed maximum_context")
        return self


class RuntimeCapabilities(RuntimeContract):
    system_prompt: bool
    reasoning: bool
    tool_request: bool
    tool_round_trip: bool
    native_schema: bool
    exact_tokenizer: bool


class CapabilityProbe(RuntimeContract):
    name: str = Field(min_length=1)
    status: ProbeStatus
    latency_ms: int = Field(ge=0)
    observed: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    trace_id: str | None = None


REQUIRED_PROBES = frozenset(
    {
        "system",
        "reasoning",
        "schema",
        "tool_request",
        "tool_round_trip",
        "tool_rejection",
        "reasoning_replay",
        "context",
    }
)


class RuntimeReport(RuntimeContract):
    schema_version: str = Field(min_length=1)
    identity: ModelIdentity
    capabilities: RuntimeCapabilities
    probes: tuple[CapabilityProbe, ...]
    readiness: RuntimeReadiness
    generated_at: datetime
    lm_studio_version: str | None = None
    warnings: tuple[str, ...] = ()
    fingerprint: str = ""

    @model_validator(mode="after")
    def validate_and_fingerprint(self) -> RuntimeReport:
        if self.readiness is RuntimeReadiness.ready:
            failed = sorted(
                probe.name
                for probe in self.probes
                if probe.name in REQUIRED_PROBES and probe.status is ProbeStatus.failed
            )
            if failed:
                raise ValueError("ready runtime has failed required probes: " + ", ".join(failed))
            present = {probe.name for probe in self.probes}
            missing = sorted(REQUIRED_PROBES - present)
            if missing:
                raise ValueError("ready runtime has missing required probes: " + ", ".join(missing))
        payload = self.model_dump(mode="json", exclude={"fingerprint", "generated_at"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "fingerprint", hashlib.sha256(canonical.encode()).hexdigest())
        return self


class NormalizedUsage(RuntimeContract):
    prompt_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    tool_result_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    detail_available: bool = False

    @model_validator(mode="after")
    def calculate_total(self) -> NormalizedUsage:
        total = (
            self.prompt_tokens
            + self.output_tokens
            + self.tool_result_tokens
        )
        if self.total_tokens not in (0, total):
            raise ValueError("total_tokens must equal the normalized token total")
        object.__setattr__(self, "total_tokens", total)
        return self


class NormalizedToolCall(RuntimeContract):
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class NormalizedResponse(RuntimeContract):
    final_content: str
    tool_calls: tuple[NormalizedToolCall, ...] = ()
    usage: NormalizedUsage = Field(default_factory=NormalizedUsage)
    finish_reason: str | None = None
    reasoning_observed: bool = False


def assert_runtime_ready(report: RuntimeReport) -> None:
    if report.readiness is not RuntimeReadiness.ready:
        raise RuntimeError(f"runtime is {report.readiness.value}; fingerprint={report.fingerprint}")
