"""Optional OpenTelemetry tracing of agent runs to a local Arize Phoenix.

Uses a private TracerProvider (never the global OTel state) so that disabling
tracing is a true no-op. `configure_telemetry()` is safe to re-run at any time
— PUT /api/settings/telemetry calls it so Settings changes apply without a
restart. A dead Phoenix endpoint only produces background export warnings; it
never blocks or fails a workflow run.
"""
import logging
from contextlib import contextmanager

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

import telemetry_config

logger = logging.getLogger("telemetry")

_provider: TracerProvider | None = None


def configure_telemetry(force_enable: bool = False) -> bool:
    """(Re)apply the persisted telemetry config. Returns True when tracing is on.

    `force_enable` turns tracing on for this process regardless of the saved
    setting (used by the eval CLI's --trace flag); it never persists anything.
    """
    global _provider

    if _provider is not None:
        try:
            _provider.shutdown()
        except Exception:
            logger.warning("shutting down previous tracer provider failed", exc_info=True)
        _provider = None

    cfg = telemetry_config.load()
    if force_enable:
        cfg = {**cfg, "enabled": True}
    if not cfg.get("enabled"):
        Agent.instrument_all(False)
        return False

    endpoint = (cfg.get("endpoint") or "http://localhost:6006").rstrip("/")
    provider = TracerProvider(
        resource=Resource.create({"service.name": "meeting-notes-agents"})
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
    )
    Agent.instrument_all(
        InstrumentationSettings(
            tracer_provider=provider,
            include_content=bool(cfg.get("capture_content", True)),
        )
    )
    _provider = provider
    logger.info("agent tracing enabled -> %s", endpoint)
    return True


def shutdown_telemetry() -> None:
    """Flush batched spans on app shutdown."""
    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None


@contextmanager
def trace_span(name: str, tracer_name: str = "agents"):
    """Root span grouping child agent spans under one trace.

    Yields the hex trace id, or None when tracing is off (a true no-op).
    """
    if _provider is None:
        yield None
        return
    tracer = _provider.get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as span:
        yield format(span.get_span_context().trace_id, "032x")


@contextmanager
def workflow_span(note_id: int):
    """Root span for one workflow run; the hex trace id is persisted to
    WorkflowRun.trace_id so the UI can link to Phoenix."""
    with trace_span(f"workflow note:{note_id}", "agents.orchestrator") as trace_id:
        yield trace_id
