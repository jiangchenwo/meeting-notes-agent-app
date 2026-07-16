from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from .contracts import (
    CapabilityProbe,
    ModelIdentity,
    ProbeStatus,
    RuntimeCapabilities,
    RuntimeReadiness,
    RuntimeReport,
)


PROBE_NAMES = (
    "system",
    "reasoning",
    "schema",
    "tool_request",
    "tool_round_trip",
    "tool_rejection",
    "reasoning_replay",
    "context",
)


class ProbeRunner(Protocol):
    def run_probe(self, name: str) -> CapabilityProbe: ...


class ScriptedProbeRunner:
    def __init__(self, script: dict[str, CapabilityProbe]) -> None:
        self.script = script

    def run_probe(self, name: str) -> CapabilityProbe:
        return self.script[name]


def run_capability_probes(
    identity: ModelIdentity,
    capabilities: RuntimeCapabilities,
    runner: ProbeRunner,
    *,
    now: datetime | None = None,
    lm_studio_version: str | None = None,
) -> RuntimeReport:
    collected: list[CapabilityProbe] = []
    for name in PROBE_NAMES:
        try:
            collected.append(runner.run_probe(name))
        except Exception as exc:
            collected.append(
                CapabilityProbe(
                    name=name,
                    status=ProbeStatus.failed,
                    latency_ms=0,
                    error_code=type(exc).__name__,
                )
            )
    probes = tuple(collected)
    readiness = (
        RuntimeReadiness.ready
        if all(probe.status is ProbeStatus.passed for probe in probes)
        else RuntimeReadiness.blocked
    )
    return RuntimeReport(
        schema_version="runtime-v1",
        identity=identity,
        capabilities=capabilities,
        probes=probes,
        readiness=readiness,
        generated_at=now or datetime.now(timezone.utc),
        lm_studio_version=lm_studio_version,
    )


def build_public_report(report: RuntimeReport) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "readiness": report.readiness.value,
        "runtime_fingerprint": report.fingerprint,
        "model": {
            "model_key": report.identity.model_key,
            "instance_id": report.identity.instance_id,
            "architecture": report.identity.architecture,
            "format": report.identity.format,
            "bits_per_weight": report.identity.bits_per_weight,
            "loaded_context": report.identity.loaded_context,
        },
        "capabilities": report.capabilities.model_dump(mode="json"),
        "probe_count": len(report.probes),
        "probes": [
            {
                "name": probe.name,
                "status": probe.status.value,
                "latency_ms": probe.latency_ms,
                "error_code": probe.error_code,
                "trace_id": probe.trace_id,
            }
            for probe in report.probes
        ],
    }
