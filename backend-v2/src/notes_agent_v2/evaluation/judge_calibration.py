from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import EvaluationBundleManifest, EvaluationBundleWriter
from .judge_settings import JudgeSettings
from .judges import (
    DataClassification,
    EvaluationJudgeGateway,
    JudgeAccounting,
    JudgeCalibration,
    JudgeIssue,
    JudgeQualification,
    PairwiseResult,
    qualify_judge,
)


class IssueCalibrationCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    case_id: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    expected_critical: bool


class PairCalibrationCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    case_id: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    better: str = Field(min_length=1)
    worse: str = Field(min_length=1)


class JudgeCalibrationSuite(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["judge-calibration-suite-v1"] = "judge-calibration-suite-v1"
    issues: tuple[IssueCalibrationCase, ...]
    pairs: tuple[PairCalibrationCase, ...]


class CalibrationCallResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    case_id: str
    evaluation_type: Literal["issues", "pairwise"]
    order: Literal["single", "better_first", "better_second"]
    schema_valid: bool
    expected_grade: int | None = Field(default=None, ge=0, le=3)
    predicted_grade: int | None = Field(default=None, ge=0, le=3)
    winner: Literal["A", "B", "tie"] | None = None
    issues: tuple[JudgeIssue, ...] = ()
    pairwise: PairwiseResult | None = None
    error_code: str | None = None


class JudgeCalibrationRunReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["judge-calibration-report-v1"] = "judge-calibration-report-v1"
    adapter_version: Literal["openai-compatible-judge-v1"] = "openai-compatible-judge-v1"
    suite_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration: JudgeCalibration
    qualification: JudgeQualification
    accounting: JudgeAccounting
    results: tuple[CalibrationCallResult, ...]


class CalibrationBudget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    request_cap: int = Field(gt=0)
    input_token_reservation: int = Field(gt=0)
    output_token_cap: int = Field(gt=0)
    time_cap_seconds: float = Field(gt=0)
    estimated_cost_usd: float = Field(gt=0)


class CalibrationSuiteError(RuntimeError):
    pass


def load_calibration_suite(path: Path) -> JudgeCalibrationSuite:
    try:
        return JudgeCalibrationSuite.model_validate_json(path.read_text())
    except Exception as exc:
        raise CalibrationSuiteError("judge calibration suite is missing or invalid") from exc


def validate_fixed_suite(suite: JudgeCalibrationSuite) -> None:
    critical = sum(item.expected_critical for item in suite.issues)
    clean = len(suite.issues) - critical
    if (critical, clean, len(suite.pairs)) != (20, 20, 10):
        raise ValueError("fixed judge calibration requires 20 critical, 20 clean, and 10 pair cases")
    case_ids = [item.case_id for item in suite.issues] + [item.case_id for item in suite.pairs]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("fixed judge calibration case IDs must be unique")


def _estimated_tokens(*values: str) -> int:
    return 150 + sum((len(value) + 3) // 4 for value in values)


def estimate_calibration_budget(
    suite: JudgeCalibrationSuite,
    settings: JudgeSettings,
) -> CalibrationBudget:
    if settings.input_cost_per_million <= 0 or settings.output_cost_per_million <= 0:
        raise ValueError("judge qualification requires positive input and output token prices")
    issue_inputs = sum(_estimated_tokens(item.candidate, item.reference) for item in suite.issues)
    pair_inputs = 2 * sum(_estimated_tokens(item.better, item.worse, item.reference) for item in suite.pairs)
    request_cap = len(suite.issues) + 2 * len(suite.pairs)
    input_tokens = issue_inputs + pair_inputs
    output_tokens = 256 * len(suite.issues) + 256 * len(suite.pairs)
    cost = (
        input_tokens * settings.input_cost_per_million
        + output_tokens * settings.output_cost_per_million
    ) / 1_000_000
    if cost > settings.max_cost_usd:
        raise ValueError("judge qualification reservation exceeds the configured cost cap")
    return CalibrationBudget(
        request_cap=request_cap,
        input_token_reservation=input_tokens,
        output_token_cap=output_tokens,
        time_cap_seconds=(
            request_cap * settings.timeout_seconds
            + max(0, request_cap - 1) * settings.min_interval_seconds
        ),
        estimated_cost_usd=cost,
    )


def _grade(severities: tuple[str, ...]) -> int:
    ranks = {"minor": 1, "major": 2, "critical": 3}
    return max((ranks[item] for item in severities), default=0)


def _quadratic_weighted_kappa(expected: list[int], predicted: list[int]) -> float:
    if len(expected) != len(predicted) or not expected:
        raise ValueError("kappa inputs must be non-empty and aligned")
    size = 4
    expected_histogram = [expected.count(index) for index in range(size)]
    predicted_histogram = [predicted.count(index) for index in range(size)]
    observed_disagreement = sum((left - right) ** 2 for left, right in zip(expected, predicted, strict=True))
    chance_disagreement = sum(
        (left - right) ** 2 * expected_histogram[left] * predicted_histogram[right] / len(expected)
        for left in range(size)
        for right in range(size)
    )
    if chance_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    return 1 - observed_disagreement / chance_disagreement


def _error_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "privacy preflight" in message:
        return "judge_privacy_failure"
    if "schema validation" in message:
        return "judge_schema_invalid"
    return "judge_call_failed"


def run_judge_calibration(
    suite: JudgeCalibrationSuite,
    gateway: EvaluationJudgeGateway,
) -> JudgeCalibrationRunReport:
    results: list[CalibrationCallResult] = []
    expected_grades: list[int] = []
    predicted_grades: list[int] = []
    schema_valid = 0
    privacy_failures = 0
    critical_true_positive = 0
    clean_critical_false_positive = 0

    for case in suite.issues:
        expected_grade = 3 if case.expected_critical else 0
        expected_grades.append(expected_grade)
        try:
            response = gateway.evaluate_issues(
                classification=DataClassification.public_benchmark,
                candidate=case.candidate,
                reference=case.reference,
                estimated_input_tokens=_estimated_tokens(case.candidate, case.reference),
                max_output_tokens=256,
            )
            predicted_grade = _grade(tuple(issue.severity for issue in response.issues))
            issues = response.issues
            schema_valid += 1
            error_code = None
        except Exception as exc:
            predicted_grade = 0
            issues = ()
            error_code = _error_code(exc)
            privacy_failures += error_code == "judge_privacy_failure"
        predicted_grades.append(predicted_grade)
        critical_true_positive += case.expected_critical and predicted_grade == 3
        clean_critical_false_positive += not case.expected_critical and predicted_grade == 3
        results.append(CalibrationCallResult(
            case_id=case.case_id,
            evaluation_type="issues",
            order="single",
            schema_valid=error_code is None,
            expected_grade=expected_grade,
            predicted_grade=predicted_grade,
            issues=issues,
            error_code=error_code,
        ))

    pair_order_agree = 0
    for case in suite.pairs:
        for order, candidate_a, candidate_b, expected_winner in (
            ("better_first", case.better, case.worse, "A"),
            ("better_second", case.worse, case.better, "B"),
        ):
            try:
                response = gateway.evaluate_pairwise(
                    classification=DataClassification.public_benchmark,
                    candidate_a=candidate_a,
                    candidate_b=candidate_b,
                    reference=case.reference,
                    estimated_input_tokens=_estimated_tokens(candidate_a, candidate_b, case.reference),
                    max_output_tokens=128,
                )
                winner = response.winner
                pairwise = response
                schema_valid += 1
                pair_order_agree += winner == expected_winner
                error_code = None
            except Exception as exc:
                winner = None
                pairwise = None
                error_code = _error_code(exc)
                privacy_failures += error_code == "judge_privacy_failure"
            results.append(CalibrationCallResult(
                case_id=case.case_id,
                evaluation_type="pairwise",
                order=order,
                schema_valid=error_code is None,
                winner=winner,
                pairwise=pairwise,
                error_code=error_code,
            ))

    calibration = JudgeCalibration(
        schema_valid=schema_valid,
        schema_total=len(suite.issues) + 2 * len(suite.pairs),
        critical_true_positive=critical_true_positive,
        critical_total=sum(item.expected_critical for item in suite.issues),
        clean_critical_false_positive=clean_critical_false_positive,
        clean_total=sum(not item.expected_critical for item in suite.issues),
        weighted_kappa=_quadratic_weighted_kappa(expected_grades, predicted_grades),
        pair_order_agree=pair_order_agree,
        pair_order_total=2 * len(suite.pairs),
        privacy_failures=privacy_failures,
    )
    canonical_suite = json.dumps(suite.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    canonical_configuration = json.dumps(
        gateway.settings.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    suite_fingerprint = hashlib.sha256(canonical_suite.encode()).hexdigest()
    configuration_fingerprint = hashlib.sha256(canonical_configuration.encode()).hexdigest()
    adapter_version = "openai-compatible-judge-v1"
    canonical_run = json.dumps(
        {
            "adapter_version": adapter_version,
            "configuration_fingerprint": configuration_fingerprint,
            "suite_fingerprint": suite_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return JudgeCalibrationRunReport(
        adapter_version=adapter_version,
        suite_fingerprint=suite_fingerprint,
        model_fingerprint=hashlib.sha256(str(gateway.settings.model).encode()).hexdigest(),
        configuration_fingerprint=configuration_fingerprint,
        run_fingerprint=hashlib.sha256(canonical_run.encode()).hexdigest(),
        calibration=calibration,
        qualification=qualify_judge(calibration),
        accounting=gateway.accounting,
        results=tuple(results),
    )


def _render_markdown(report: JudgeCalibrationRunReport) -> str:
    qualification = report.qualification
    accounting = report.accounting
    return "\n".join((
        "# Remote judge qualification",
        "",
        f"Status: `{qualification.status.value}`",
        "",
        "| Gate | Result |",
        "|---|---:|",
        f"| Schema validity | {qualification.schema_validity:.3f} |",
        f"| Critical recall | {qualification.critical_recall:.3f} |",
        f"| Clean critical false-positive rate | {qualification.clean_critical_false_positive_rate:.3f} |",
        f"| Weighted kappa | {qualification.weighted_kappa:.3f} |",
        f"| Pair-order agreement | {qualification.pair_order_agreement:.3f} |",
        f"| Privacy failures | {qualification.privacy_failures} |",
        "",
        "## Reserved request budget",
        "",
        f"- Requests: {accounting.requests}",
        f"- Input tokens: {accounting.reserved_input_tokens}",
        f"- Output-token cap: {accounting.reserved_output_tokens}",
        f"- Estimated maximum cost: `${accounting.estimated_cost_usd:.6f}`",
        "",
        f"Suite fingerprint: `{report.suite_fingerprint}`",
        f"Model fingerprint: `{report.model_fingerprint}`",
        f"Configuration fingerprint: `{report.configuration_fingerprint}`",
        f"Run fingerprint: `{report.run_fingerprint}`",
        "",
    ))


def write_calibration_bundle(
    output: Path, *, suite: JudgeCalibrationSuite, report: JudgeCalibrationRunReport,
    trace_path: Path,
) -> EvaluationBundleManifest:
    writer = EvaluationBundleWriter(
        output,
        run_id="remote-judge-qualification",
        fingerprint=report.run_fingerprint,
    )
    writer.write_json("suite.json", suite.model_dump(mode="json"))
    writer.write_json("calibration-results.json", {
        "schema_version": "judge-calibration-results-v1",
        "results": [item.model_dump(mode="json") for item in report.results],
    })
    writer.write_json("report.json", report.model_dump(mode="json"))
    writer.write_text("report.md", _render_markdown(report))
    writer.write_text("events.jsonl", trace_path.read_text())
    return writer.seal()
