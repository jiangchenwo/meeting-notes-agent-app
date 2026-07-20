from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import (
    EvaluationBundleManifest,
    EvaluationBundleWriter,
    verify_bundle,
)
from .judge_calibration import JudgeCalibrationRunReport
from .judge_settings import JudgeSettings
from .judges import (
    DataClassification,
    EvaluationJudgeGateway,
    JudgeAccounting,
    JudgeIssue,
    JudgeQualificationStatus,
    qualify_judge,
)


class JudgeQualificationAuthorization(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceVerificationJudgeCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    injected_error: bool


class SourceVerificationJudgeBudget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_cap: int = Field(gt=0)
    input_token_reservation: int = Field(gt=0)
    output_token_cap: int = Field(gt=0)
    time_cap_seconds: float = Field(gt=0)
    estimated_cost_usd: float = Field(gt=0)


class SourceVerificationJudgeResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    injected_error: bool
    schema_valid: bool
    critical_issue_detected: bool
    citation_valid: bool
    issues: tuple[JudgeIssue, ...] = ()
    error_code: str | None = None


class SourceVerificationJudgeReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["source-verification-judge-report-v1"] = (
        "source-verification-judge-report-v1"
    )
    verdict: Literal["passed", "failed"]
    case_count: int = Field(gt=0)
    judge_authorization_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_qualification_bundle_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: dict[str, float]
    hard_gates: dict[str, bool]
    accounting: JudgeAccounting
    results: tuple[SourceVerificationJudgeResult, ...]
    result_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


def judge_configuration_fingerprint(settings: JudgeSettings) -> str:
    canonical = json.dumps(
        settings.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_judge_qualification_bundle(
    path: Path, settings: JudgeSettings
) -> JudgeQualificationAuthorization:
    manifest = verify_bundle(path)
    try:
        report = JudgeCalibrationRunReport.model_validate_json(
            (path / "report.json").read_text()
        )
    except Exception as exc:
        raise ValueError("judge qualification report is missing or invalid") from exc
    if manifest.run_id != "remote-judge-qualification":
        raise ValueError("judge qualification bundle has the wrong purpose")
    if manifest.fingerprint != report.run_fingerprint:
        raise ValueError("judge qualification fingerprint mismatch")
    if report.qualification.status is not JudgeQualificationStatus.qualified:
        raise ValueError("remote judge is not qualified")
    if qualify_judge(report.calibration) != report.qualification:
        raise ValueError("judge qualification metrics are inconsistent")
    if report.accounting.requests != report.calibration.schema_total:
        raise ValueError("judge qualification request accounting is incomplete")
    current = judge_configuration_fingerprint(settings)
    if report.configuration_fingerprint != current:
        raise ValueError("judge qualification configuration drift")
    return JudgeQualificationAuthorization(
        run_fingerprint=report.run_fingerprint,
        configuration_fingerprint=current,
        bundle_digest=manifest.bundle_digest,
    )


def _estimated_tokens(candidate: str, reference: str) -> int:
    return 150 + (len(candidate) + 3) // 4 + (len(reference) + 3) // 4


def estimate_source_verification_judge_budget(
    cases: tuple[SourceVerificationJudgeCase, ...], settings: JudgeSettings
) -> SourceVerificationJudgeBudget:
    if not cases:
        raise ValueError("source verification judge cases are empty")
    if settings.input_cost_per_million <= 0 or settings.output_cost_per_million <= 0:
        raise ValueError(
            "source verification judge requires positive input and output token prices"
        )
    input_tokens = sum(
        _estimated_tokens(case.candidate, case.reference) for case in cases
    )
    output_tokens = 256 * len(cases)
    cost = (
        input_tokens * settings.input_cost_per_million
        + output_tokens * settings.output_cost_per_million
    ) / 1_000_000
    if cost > settings.max_cost_usd:
        raise ValueError("source verification judge exceeds configured cost cap")
    return SourceVerificationJudgeBudget(
        request_cap=len(cases),
        input_token_reservation=input_tokens,
        output_token_cap=output_tokens,
        time_cap_seconds=(
            settings.timeout_seconds * len(cases)
            + settings.min_interval_seconds * max(0, len(cases) - 1)
        ),
        estimated_cost_usd=cost,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def run_source_verification_judge(
    cases: tuple[SourceVerificationJudgeCase, ...],
    gateway: EvaluationJudgeGateway,
    *,
    judge_authorization_fingerprint: str,
    judge_qualification_bundle_digest: str,
    development_fingerprint: str,
) -> SourceVerificationJudgeReport:
    if not cases:
        raise ValueError("source verification judge cases are empty")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("source verification judge case IDs must be unique")
    results: list[SourceVerificationJudgeResult] = []
    for case in cases:
        try:
            response = gateway.evaluate_issues(
                classification=DataClassification.public_benchmark,
                candidate=case.candidate,
                reference=case.reference,
                estimated_input_tokens=_estimated_tokens(
                    case.candidate, case.reference
                ),
                max_output_tokens=256,
            )
            issues = response.issues
            schema_valid = True
            error_code = None
        except Exception as exc:
            issues = ()
            schema_valid = False
            error_code = type(exc).__name__
        cited = tuple(ref for issue in issues for ref in issue.evidence_refs)
        citation_valid = not cited or (
            all(cited) and set(cited).issubset(case.evidence_ids)
        )
        results.append(
            SourceVerificationJudgeResult(
                case_id=case.case_id,
                injected_error=case.injected_error,
                schema_valid=schema_valid,
                critical_issue_detected=any(
                    issue.severity == "critical" for issue in issues
                ),
                citation_valid=citation_valid,
                issues=issues,
                error_code=error_code,
            )
        )

    injected = tuple(item for item in results if item.injected_error)
    clean = tuple(item for item in results if not item.injected_error)
    metrics = {
        "schema_validity": _ratio(sum(item.schema_valid for item in results), len(results)),
        "injected_error_recall": _ratio(
            sum(item.critical_issue_detected for item in injected), len(injected)
        ),
        "clean_critical_false_positive_rate": _ratio(
            sum(item.critical_issue_detected for item in clean), len(clean)
        ),
        "evidence_citation_precision": _ratio(
            sum(item.citation_valid for item in results), len(results)
        ),
    }
    hard_gates = {
        "complete_schedule": len(results) == len(cases),
        "request_accounting": gateway.accounting.requests == len(cases),
        "schema_validity": metrics["schema_validity"] == 1,
        "injected_error_recall": metrics["injected_error_recall"] == 1,
        "clean_critical_false_positives": metrics[
            "clean_critical_false_positive_rate"
        ]
        == 0,
        "evidence_citation_precision": metrics["evidence_citation_precision"] == 1,
    }
    encoded = json.dumps(
        [item.model_dump(mode="json") for item in results],
        sort_keys=True,
        separators=(",", ":"),
    )
    return SourceVerificationJudgeReport(
        verdict="passed" if all(hard_gates.values()) else "failed",
        case_count=len(cases),
        judge_authorization_fingerprint=judge_authorization_fingerprint,
        judge_qualification_bundle_digest=judge_qualification_bundle_digest,
        judge_configuration_fingerprint=judge_configuration_fingerprint(
            gateway.settings
        ),
        development_fingerprint=development_fingerprint,
        metrics=metrics,
        hard_gates=hard_gates,
        accounting=gateway.accounting,
        results=tuple(results),
        result_digest=hashlib.sha256(encoded.encode()).hexdigest(),
    )


def _render_markdown(report: SourceVerificationJudgeReport) -> str:
    return "\n".join(
        (
            "# Source verification judge report",
            "",
            f"Verdict: `{report.verdict}`",
            "",
            f"Cases: {report.case_count}",
            f"Requests: {report.accounting.requests}",
            f"Estimated cost: ${report.accounting.estimated_cost_usd:.6f}",
            "",
            "## Metrics",
            "",
            *(f"- {key}: {value:.6f}" for key, value in report.metrics.items()),
            "",
            "## Hard gates",
            "",
            *(f"- {key}: {value}" for key, value in report.hard_gates.items()),
            "",
            f"Result digest: `{report.result_digest}`",
            f"Judge authorization: `{report.judge_authorization_fingerprint}`",
            f"Judge qualification bundle: `{report.judge_qualification_bundle_digest}`",
            f"Development set: `{report.development_fingerprint}`",
            "",
        )
    )


def write_source_verification_judge_bundle(
    output: Path,
    *,
    cases: tuple[SourceVerificationJudgeCase, ...],
    report: SourceVerificationJudgeReport,
    trace_path: Path,
) -> EvaluationBundleManifest:
    writer = EvaluationBundleWriter(
        output,
        run_id="source-verification-judge",
        fingerprint=report.result_digest,
    )
    writer.write_json(
        "cases.json", [case.model_dump(mode="json") for case in cases]
    )
    writer.write_json(
        "results.json", [item.model_dump(mode="json") for item in report.results]
    )
    writer.write_json("report.json", report.model_dump(mode="json"))
    writer.write_text("report.md", _render_markdown(report))
    writer.write_text("events.jsonl", trace_path.read_text())
    return writer.seal()
