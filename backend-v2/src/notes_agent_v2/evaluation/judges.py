from __future__ import annotations

from enum import StrEnum
import hashlib
import json
from threading import Lock
import time
from typing import Any, Callable, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .judge_settings import JudgeSettings
from .tracing import JsonlTraceRecorder


class JudgeError(RuntimeError):
    pass


class DataClassification(StrEnum):
    public_benchmark = "public_benchmark"
    private_user = "private_user"
    private_project = "private_project"
    legacy = "legacy"


class JudgeIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    code: str = Field(min_length=1)
    severity: str = Field(pattern=r"^(critical|major|minor)$")
    evidence_refs: tuple[str, ...]
    justification: str = Field(min_length=1, max_length=500)


class JudgeIssueResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    issues: tuple[JudgeIssue, ...]


class PairwiseResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    winner: str = Field(pattern=r"^(A|B|tie)$")
    evidence_refs: tuple[str, ...]
    justification: str = Field(min_length=1, max_length=500)


class JudgeProvider(Protocol):
    def complete_json(self, request: dict[str, Any]) -> dict[str, Any]: ...


class ScriptedJudgeProvider:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def complete_json(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        if not self.responses:
            raise JudgeError("scripted judge response exhausted")
        return self.responses.pop(0)


class OpenAICompatibleJudgeProvider:
    def __init__(self, settings: JudgeSettings, *, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.Client(timeout=settings.timeout_seconds)

    def complete_json(self, request: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(f"{str(self.settings.base_url).rstrip('/')}/chat/completions", headers={"Authorization": f"Bearer {self.settings.api_token}"}, json=request)
        response.raise_for_status()
        try:
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as exc:
            raise JudgeError("judge response was not valid structured JSON") from exc


class JudgeAccounting(BaseModel):
    requests: int = 0
    reserved_input_tokens: int = 0
    reserved_output_tokens: int = 0
    estimated_cost_usd: float = 0


class JudgeQualificationStatus(StrEnum):
    qualified = "qualified"
    diagnostic_unqualified = "diagnostic_unqualified"


class JudgeCalibration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_valid: int = Field(ge=0)
    schema_total: int = Field(gt=0)
    critical_true_positive: int = Field(ge=0)
    critical_total: int = Field(gt=0)
    clean_critical_false_positive: int = Field(ge=0)
    clean_total: int = Field(gt=0)
    weighted_kappa: float = Field(ge=-1, le=1)
    pair_order_agree: int = Field(ge=0)
    pair_order_total: int = Field(gt=0)
    privacy_failures: int = Field(ge=0)


class JudgeQualification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: JudgeQualificationStatus
    schema_validity: float
    critical_recall: float
    clean_critical_false_positive_rate: float
    weighted_kappa: float
    pair_order_agreement: float
    privacy_failures: int


def qualify_judge(calibration: JudgeCalibration) -> JudgeQualification:
    schema = calibration.schema_valid / calibration.schema_total
    recall = calibration.critical_true_positive / calibration.critical_total
    false_positive = calibration.clean_critical_false_positive / calibration.clean_total
    order = calibration.pair_order_agree / calibration.pair_order_total
    passed = (
        schema == 1
        and recall >= 0.95
        and false_positive <= 0.05
        and calibration.weighted_kappa >= 0.70
        and order >= 0.90
        and calibration.privacy_failures == 0
    )
    return JudgeQualification(
        status=JudgeQualificationStatus.qualified if passed else JudgeQualificationStatus.diagnostic_unqualified,
        schema_validity=schema,
        critical_recall=recall,
        clean_critical_false_positive_rate=false_positive,
        weighted_kappa=calibration.weighted_kappa,
        pair_order_agreement=order,
        privacy_failures=calibration.privacy_failures,
    )


class EvaluationJudgeGateway:
    def __init__(
        self, settings: JudgeSettings, provider: JudgeProvider, *, allow_remote_judge: bool,
        trace_recorder: JsonlTraceRecorder | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if settings.provider == "disabled":
            raise JudgeError("judge provider is disabled")
        if not allow_remote_judge:
            raise JudgeError("remote judge requires explicit opt-in")
        self.settings, self.provider = settings, provider
        self.accounting = JudgeAccounting()
        self._lock = Lock()
        self.trace_recorder = trace_recorder
        self._clock = clock
        self._sleeper = sleeper
        self._last_request_started: float | None = None

    def _complete_json(self, request: dict[str, Any]) -> dict[str, Any]:
        now = self._clock()
        if self._last_request_started is not None:
            wait = self.settings.min_interval_seconds - (now - self._last_request_started)
            if wait > 0:
                self._sleeper(wait)
                now = self._clock()
        self._last_request_started = now
        return self.provider.complete_json(request)

    def evaluate_issues(self, *, classification: DataClassification, candidate: str, reference: str, estimated_input_tokens: int, max_output_tokens: int) -> JudgeIssueResult:
        if self.trace_recorder is None:
            return self._evaluate_issues(classification=classification, candidate=candidate, reference=reference, estimated_input_tokens=estimated_input_tokens, max_output_tokens=max_output_tokens)
        model_fingerprint = hashlib.sha256(str(self.settings.model).encode()).hexdigest()
        rubric_fingerprint = hashlib.sha256(self.settings.rubric.encode()).hexdigest()
        with self.trace_recorder.span("judge", classification=classification.value, model_fingerprint=model_fingerprint, rubric_fingerprint=rubric_fingerprint) as span:
            before = self.accounting
            try:
                result = self._evaluate_issues(classification=classification, candidate=candidate, reference=reference, estimated_input_tokens=estimated_input_tokens, max_output_tokens=max_output_tokens)
            except Exception as exc:
                span.terminal(status="failed", accounting=self._trace_accounting(before), error_code=type(exc).__name__)
                raise
            span.terminal(accounting=self._trace_accounting(before))
            return result

    def _evaluate_issues(self, *, classification: DataClassification, candidate: str, reference: str, estimated_input_tokens: int, max_output_tokens: int) -> JudgeIssueResult:
        if classification is not DataClassification.public_benchmark:
            raise JudgeError("remote judge accepts public benchmark material only")
        if any(marker in candidate.lower() or marker in reference.lower() for marker in ("api_key", "authorization:", "<reasoning>")):
            raise JudgeError("remote judge privacy preflight failed")
        cost = (estimated_input_tokens * self.settings.input_cost_per_million + max_output_tokens * self.settings.output_cost_per_million) / 1_000_000
        with self._lock:
            if self.accounting.estimated_cost_usd + cost > self.settings.max_cost_usd:
                raise JudgeError("judge cost budget exhausted")
            self.accounting = JudgeAccounting(
                requests=self.accounting.requests + 1,
                reserved_input_tokens=self.accounting.reserved_input_tokens + estimated_input_tokens,
                reserved_output_tokens=self.accounting.reserved_output_tokens + max_output_tokens,
                estimated_cost_usd=self.accounting.estimated_cost_usd + cost,
            )
            payload = self._complete_json({
                "model": self.settings.model, "temperature": self.settings.temperature,
                "response_format": {"type": "json_object"},
                "messages": [{
                    "role": "system",
                    "content": (
                        "Compare candidate meeting notes with the reference. Return only a JSON object shaped as "
                        "{\"issues\":[{\"code\":\"short_code\",\"severity\":\"critical|major|minor\","
                        "\"evidence_refs\":[\"u1\"],\"justification\":\"concise explanation\"}]}. "
                        "Use an empty issues list when the candidate is faithful. Mark an issue critical only for a "
                        "material contradiction or unsupported material decision, action, owner, date, amount, "
                        "status, scope, or risk. Use evidence IDs present in the reference. Never provide "
                        "chain-of-thought."
                    ),
                }, {"role": "user", "content": json.dumps({"candidate": candidate, "reference": reference, "rubric": self.settings.rubric})}],
                "max_tokens": max_output_tokens,
            })
        try:
            return JudgeIssueResult.model_validate(payload)
        except Exception as exc:
            raise JudgeError("judge issue result failed schema validation") from exc

    def evaluate_pairwise(
        self, *, classification: DataClassification, candidate_a: str, candidate_b: str,
        reference: str, estimated_input_tokens: int, max_output_tokens: int,
    ) -> PairwiseResult:
        if self.trace_recorder is None:
            return self._evaluate_pairwise(
                classification=classification, candidate_a=candidate_a, candidate_b=candidate_b,
                reference=reference, estimated_input_tokens=estimated_input_tokens,
                max_output_tokens=max_output_tokens,
            )
        model_fingerprint = hashlib.sha256(str(self.settings.model).encode()).hexdigest()
        rubric_fingerprint = hashlib.sha256(self.settings.rubric.encode()).hexdigest()
        with self.trace_recorder.span(
            "judge", classification=classification.value, model_fingerprint=model_fingerprint,
            rubric_fingerprint=rubric_fingerprint, evaluation_type="pairwise",
        ) as span:
            before = self.accounting
            try:
                result = self._evaluate_pairwise(
                    classification=classification, candidate_a=candidate_a, candidate_b=candidate_b,
                    reference=reference, estimated_input_tokens=estimated_input_tokens,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as exc:
                span.terminal(status="failed", accounting=self._trace_accounting(before), error_code=type(exc).__name__)
                raise
            span.terminal(accounting=self._trace_accounting(before))
            return result

    def _trace_accounting(self, before: JudgeAccounting) -> dict[str, int | float]:
        return {
            "requests": self.accounting.requests - before.requests,
            "input_tokens": self.accounting.reserved_input_tokens - before.reserved_input_tokens,
            "output_token_cap": self.accounting.reserved_output_tokens - before.reserved_output_tokens,
            "cost_usd": self.accounting.estimated_cost_usd - before.estimated_cost_usd,
        }

    def _evaluate_pairwise(
        self, *, classification: DataClassification, candidate_a: str, candidate_b: str,
        reference: str, estimated_input_tokens: int, max_output_tokens: int,
    ) -> PairwiseResult:
        if classification is not DataClassification.public_benchmark:
            raise JudgeError("remote judge accepts public benchmark material only")
        values = (candidate_a, candidate_b, reference)
        if any(marker in value.lower() for value in values for marker in ("api_key", "authorization:", "<reasoning>")):
            raise JudgeError("remote judge privacy preflight failed")
        cost = (
            estimated_input_tokens * self.settings.input_cost_per_million
            + max_output_tokens * self.settings.output_cost_per_million
        ) / 1_000_000
        with self._lock:
            if self.accounting.estimated_cost_usd + cost > self.settings.max_cost_usd:
                raise JudgeError("judge cost budget exhausted")
            self.accounting = JudgeAccounting(
                requests=self.accounting.requests + 1,
                reserved_input_tokens=self.accounting.reserved_input_tokens + estimated_input_tokens,
                reserved_output_tokens=self.accounting.reserved_output_tokens + max_output_tokens,
                estimated_cost_usd=self.accounting.estimated_cost_usd + cost,
            )
            payload = self._complete_json({
                "model": self.settings.model,
                "temperature": self.settings.temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Choose the candidate that is more factually correct and complete relative to the "
                            "reference. Return only JSON with winner A, B, or tie; evidence_refs; and a concise "
                            "justification. Never provide chain-of-thought."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps({
                            "candidate_a": candidate_a, "candidate_b": candidate_b,
                            "reference": reference, "rubric": self.settings.rubric,
                        }),
                    },
                ],
                "max_tokens": max_output_tokens,
            })
        try:
            return PairwiseResult.model_validate(payload)
        except Exception as exc:
            raise JudgeError("judge pairwise result failed schema validation") from exc
