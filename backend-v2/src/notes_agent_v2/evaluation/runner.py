from __future__ import annotations

from collections.abc import Callable, Iterable
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .tracing import JsonlTraceRecorder


class RunError(RuntimeError):
    pass


class EvaluationCell(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    feature_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    variant: Literal["baseline", "treatment"]
    seed: int
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def key(self) -> str:
        return f"{self.feature_id}--{self.case_id}--{self.variant}--{self.seed}".replace("/", "_")


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    cell: EvaluationCell
    trace_id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)
    valid: bool
    score: float | None
    requests: int = Field(ge=0)
    error_code: str | None = None


class EvaluationRunner:
    def __init__(self, output: Path, execute: Callable[[EvaluationCell], dict[str, Any]]) -> None:
        self.output, self.execute = output, execute
        output.mkdir(parents=True, exist_ok=True)
        self.trace = JsonlTraceRecorder(output / "events.jsonl")

    def run(self, cells: Iterable[EvaluationCell]) -> tuple[EvaluationResult, ...]:
        results = []
        for cell in cells:
            path = self.output / f"{cell.key}.json"
            with self.trace.span("evaluator", feature_id=cell.feature_id, case_id=cell.case_id, variant=cell.variant, seed=cell.seed, fingerprint=cell.fingerprint) as span:
                if path.exists():
                    existing = EvaluationResult.model_validate_json(path.read_text())
                    if existing.cell.fingerprint != cell.fingerprint:
                        span.terminal(status="invalid", error_code="fingerprint_mismatch")
                        raise RunError("resume fingerprint mismatch")
                    span.terminal(accounting={"requests": 0})
                    results.append(existing)
                    continue
                try:
                    payload = self.execute(cell)
                    result = EvaluationResult(cell=cell, trace_id=self.trace.trace_id, span_id=span.span_id, **payload)
                except Exception as exc:
                    result = EvaluationResult(cell=cell, trace_id=self.trace.trace_id, span_id=span.span_id, valid=False, score=None, requests=0, error_code=type(exc).__name__)
                span.terminal(status="passed" if result.valid else "invalid", accounting={"requests": result.requests}, error_code=result.error_code)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n")
            temporary.replace(path)
            results.append(result)
        return tuple(results)
