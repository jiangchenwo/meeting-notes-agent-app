from __future__ import annotations

import hashlib
import json
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .runner import EvaluationResult


class ReportError(RuntimeError):
    pass


class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    feature_id: str
    verdict: Literal["passed", "failed", "invalid"]
    pair_count: int
    paired_delta: float | None
    bootstrap_low: float | None
    bootstrap_high: float | None
    invalid_count: int
    request_count: int
    result_digest: str


def build_report(feature_id: str, results: tuple[EvaluationResult, ...]) -> EvaluationReport:
    grouped: dict[tuple[str, int], dict[str, EvaluationResult]] = {}
    seen: set[tuple[str, int, str]] = set()
    for result in results:
        if result.cell.feature_id != feature_id:
            raise ReportError("mixed feature results")
        key = (result.cell.case_id, result.cell.seed, result.cell.variant)
        if key in seen:
            raise ReportError("duplicate evaluation result cell")
        seen.add(key)
        grouped.setdefault((result.cell.case_id, result.cell.seed), {})[result.cell.variant] = result
    if not grouped or any(set(pair) != {"baseline", "treatment"} for pair in grouped.values()):
        raise ReportError("every requested case/seed requires a complete pair")
    if any(pair["baseline"].cell.fingerprint != pair["treatment"].cell.fingerprint for pair in grouped.values()):
        raise ReportError("evaluation pair fingerprint mismatch")
    invalid = sum(not result.valid for result in results)
    deltas = [pair["treatment"].score - pair["baseline"].score for pair in grouped.values() if pair["treatment"].valid and pair["baseline"].valid and pair["treatment"].score is not None and pair["baseline"].score is not None]
    delta = sum(deltas) / len(deltas) if len(deltas) == len(grouped) else None
    low, high = _bootstrap_interval(deltas) if delta is not None else (None, None)
    verdict = "invalid" if invalid or delta is None else "passed" if delta > 0 else "failed"
    payload = [item.model_dump(mode="json") for item in sorted(results, key=lambda item: item.cell.key)]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return EvaluationReport(feature_id=feature_id, verdict=verdict, pair_count=len(grouped), paired_delta=delta, bootstrap_low=low, bootstrap_high=high, invalid_count=invalid, request_count=sum(item.requests for item in results), result_digest=digest)


def _bootstrap_interval(deltas: list[float], *, seed: int = 41, samples: int = 1000) -> tuple[float, float]:
    if len(deltas) == 1:
        return deltas[0], deltas[0]
    generator = random.Random(seed)
    means = sorted(sum(generator.choice(deltas) for _ in deltas) / len(deltas) for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def render_report_json(report: EvaluationReport) -> str:
    return json.dumps(report.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"


def render_report_markdown(report: EvaluationReport) -> str:
    delta = "not applicable" if report.paired_delta is None else f"{report.paired_delta:.6f}"
    return (
        f"# Evaluation report: {report.feature_id}\n\n"
        f"- Verdict: `{report.verdict}`\n"
        f"- Complete pairs: `{report.pair_count}`\n"
        f"- Invalid results: `{report.invalid_count}`\n"
        f"- Paired delta: `{delta}`\n"
        f"- Bootstrap interval: `{report.bootstrap_low}` to `{report.bootstrap_high}`\n"
        f"- Requests: `{report.request_count}`\n"
        f"- Result digest: `{report.result_digest}`\n"
    )
