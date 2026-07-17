from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Literal, Protocol
import uuid

from pydantic import BaseModel, ConfigDict, Field


class TraceError(RuntimeError):
    pass


FORBIDDEN_KEYS = ("transcript", "prompt", "output", "reasoning", "secret", "authorization", "api_key", "arguments", "results")


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["trace-v1"] = "trace-v1"
    trace_id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)
    parent_span_id: str | None = None
    sequence: int = Field(gt=0)
    timestamp_ns: int = Field(gt=0)
    kind: Literal["stage", "model", "judge", "tool", "evaluator", "metric", "artifact", "report"]
    phase: Literal["start", "terminal"]
    status: Literal["started", "passed", "failed", "invalid", "cancelled"]
    metadata: dict[str, Any] = {}
    accounting: dict[str, int | float] = {}
    artifact_digests: dict[str, str] = {}
    error_code: str | None = None


class TraceRecorder(Protocol):
    def start(self, kind: str, *, parent_span_id: str | None = None, metadata: dict[str, Any] | None = None) -> str: ...
    def terminal(self, span_id: str, *, status: str = "passed", accounting: dict[str, int | float] | None = None, artifact_digests: dict[str, str] | None = None, error_code: str | None = None) -> None: ...


class _Span(AbstractContextManager["_Span"]):
    def __init__(self, recorder: JsonlTraceRecorder, span_id: str) -> None:
        self.recorder, self.span_id, self.done = recorder, span_id, False

    def terminal(self, **kwargs: Any) -> None:
        if self.done:
            raise TraceError("duplicate terminal event")
        self.recorder.terminal(self.span_id, **kwargs)
        self.done = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if not self.done:
            self.terminal(status="failed" if exc is not None else "passed", error_code=type(exc).__name__ if exc else None)
        return False


class JsonlTraceRecorder:
    def __init__(self, path: Path, *, trace_id: str | None = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing_lines = path.read_text().splitlines() if path.exists() else []
        existing_trace_id = json.loads(existing_lines[0])["trace_id"] if existing_lines else None
        self.trace_id = trace_id or existing_trace_id or uuid.uuid4().hex
        if existing_trace_id is not None and self.trace_id != existing_trace_id:
            raise TraceError("trace ID does not match existing trace")
        self.sequence = len(existing_lines)
        self.started: set[str] = set()
        self.terminated: set[str] = set()

    def _append(self, event: TraceEvent) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps(event.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()

    def start(self, kind: str, *, parent_span_id: str | None = None, metadata: dict[str, Any] | None = None) -> str:
        _assert_safe(metadata or {})
        if parent_span_id is not None and parent_span_id not in self.started:
            raise TraceError("parent span was not started")
        span_id = uuid.uuid4().hex
        self.sequence += 1
        self._append(TraceEvent(trace_id=self.trace_id, span_id=span_id, parent_span_id=parent_span_id, sequence=self.sequence, timestamp_ns=time.monotonic_ns(), kind=kind, phase="start", status="started", metadata=metadata or {}))
        self.started.add(span_id)
        return span_id

    def terminal(self, span_id: str, *, status: str = "passed", accounting: dict[str, int | float] | None = None, artifact_digests: dict[str, str] | None = None, error_code: str | None = None) -> None:
        if span_id not in self.started or span_id in self.terminated:
            raise TraceError("terminal event has missing start or is duplicate")
        self.sequence += 1
        self._append(TraceEvent(trace_id=self.trace_id, span_id=span_id, sequence=self.sequence, timestamp_ns=time.monotonic_ns(), kind=_span_kind(self.path, span_id), phase="terminal", status=status, accounting=accounting or {}, artifact_digests=artifact_digests or {}, error_code=error_code))
        self.terminated.add(span_id)

    def span(self, kind: str, *, parent_span_id: str | None = None, **metadata: Any) -> _Span:
        return _Span(self, self.start(kind, parent_span_id=parent_span_id, metadata=metadata))


def _span_kind(path: Path, span_id: str) -> str:
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event["span_id"] == span_id and event["phase"] == "start":
            return str(event["kind"])
    raise TraceError("span start is missing")


def _assert_safe(value: Any, path: str = "metadata") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_KEYS):
                raise TraceError(f"forbidden trace field: {path}.{key}")
            _assert_safe(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_safe(child, f"{path}[{index}]")


@dataclass(frozen=True)
class TraceValidation:
    span_count: int
    request_count: int
    failure_count: int


def validate_trace(path: Path) -> TraceValidation:
    starts: dict[str, TraceEvent] = {}
    terminals: dict[str, TraceEvent] = {}
    previous_sequence = 0
    previous_time = 0
    trace_id: str | None = None
    for line in path.read_text().splitlines():
        try:
            event = TraceEvent.model_validate_json(line)
        except Exception as exc:
            raise TraceError("trace event is malformed") from exc
        _assert_safe(event.metadata)
        if trace_id is None:
            trace_id = event.trace_id
        elif event.trace_id != trace_id:
            raise TraceError("trace contains mixed trace IDs")
        if event.sequence <= previous_sequence or event.timestamp_ns < previous_time:
            raise TraceError("trace sequence or timestamp is not monotonic")
        previous_sequence, previous_time = event.sequence, event.timestamp_ns
        target = starts if event.phase == "start" else terminals
        if event.span_id in target:
            raise TraceError("duplicate trace event")
        target[event.span_id] = event
    if set(starts) != set(terminals):
        raise TraceError("trace has orphan, missing, or duplicate terminal spans")
    if any(item.parent_span_id is not None and item.parent_span_id not in starts for item in starts.values()):
        raise TraceError("trace has orphan parent")
    requests = sum(int(item.accounting.get("requests", 0)) for item in terminals.values())
    failures = sum(item.status != "passed" for item in terminals.values())
    return TraceValidation(len(starts), requests, failures)
