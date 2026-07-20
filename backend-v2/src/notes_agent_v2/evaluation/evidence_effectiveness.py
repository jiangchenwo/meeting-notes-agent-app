from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from notes_agent_v2.domain.evidence import EvidenceSpan, ExtractedFactCandidate
from notes_agent_v2.domain.transcript import Utterance
from notes_agent_v2.workflow.preflight import normalize_transcript


EvidenceFeature = Literal[
    "evidence.cited_atomic_extraction",
    "evidence.source_verification",
    "evidence.loss_aware_consolidation",
]


class ExtractionObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    baseline_reference_hits: int = Field(ge=0)
    treatment_reference_hits: int = Field(ge=0)
    reference_total: int = Field(gt=0)
    baseline_supported_candidates: int = Field(ge=0)
    baseline_candidate_total: int = Field(ge=0)
    treatment_supported_candidates: int = Field(ge=0)
    treatment_candidate_total: int = Field(ge=0)
    treatment_citations_valid: bool
    treatment_complete: bool
    provider_requests: int = Field(ge=0)


class VerificationObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    injected_error: bool
    baseline_accepted: bool
    treatment_status: Literal["supported", "uncertain", "contradicted"]
    citation_valid: bool
    provider_requests: int = Field(ge=0)


class ConsolidationObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    baseline_fact_count: int = Field(ge=0)
    treatment_fact_count: int = Field(ge=0)
    duplicate_candidate_count: int = Field(ge=2)
    duplicate_fact_count: int = Field(ge=1)
    expected_unique_count: int = Field(gt=0)
    observed_unique_count: int = Field(ge=0)
    evidence_preserved: bool
    relationships_preserved: bool
    false_semantic_merges: int = Field(ge=0)
    provider_requests: int = Field(ge=0)


EvidenceObservation = (
    ExtractionObservation | VerificationObservation | ConsolidationObservation
)


def candidate_from_reference(
    identifier: str,
    reference: Mapping[str, object],
    utterance_by_id: Mapping[str, Utterance],
    *,
    text: str | None = None,
) -> ExtractedFactCandidate:
    raw_ids = reference.get("evidence_ids")
    if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
        raise ValueError("reference evidence IDs are missing")
    evidence_ids = tuple(str(item) for item in raw_ids)
    try:
        source = "\n".join(utterance_by_id[item].text for item in evidence_ids)
    except KeyError as exc:
        raise ValueError("reference evidence is outside the transcript") from exc
    role = str(reference.get("role"))
    kind = (
        "decision"
        if role == "decision_summary"
        else "action"
        if role == "action_summary"
        else "fact"
    )
    return ExtractedFactCandidate(
        id=identifier,
        text=text or str(reference.get("text")),
        kind=kind,
        status="asserted",
        speaker_ids=(),
        owner=None,
        due_text=None,
        evidence=(EvidenceSpan(utterance_ids=evidence_ids, quote=source),),
    )


def valid_verification_citation(
    decision_ids: Sequence[str], candidate_ids: Sequence[str]
) -> bool:
    return bool(decision_ids) and set(decision_ids).issubset(candidate_ids)


def inject_polarity_defect(text: str) -> str:
    transformations = (
        (r"\bwill\s+not\b", "will"),
        (r"\bwon't\b", "will"),
        (r"\bdecided\s+to\s+use\b", "decided not to use"),
        (r"\bwill\b", "will not"),
        (r"\buses\b", "does not use"),
        (r"\bdoes\s+not\s+use\b", "uses"),
        (r"\bnot\b", ""),
        (r"\bis\b", "is not"),
        (r"\bare\b", "are not"),
        (r"\bhas\b", "does not have"),
        (r"\bhave\b", "do not have"),
    )
    for pattern, replacement in transformations:
        injected, count = re.subn(
            pattern, replacement, text, count=1, flags=re.IGNORECASE
        )
        if count:
            injected = " ".join(injected.split())
            if injected.casefold() == text.casefold():
                raise ValueError("polarity injection did not change candidate")
            return injected
    raise ValueError("candidate has no supported polarity transformation")


def development_utterances(case: Mapping[str, object]) -> tuple[Utterance, ...]:
    raw = case.get("utterances")
    if not isinstance(raw, list):
        raise ValueError("development case utterances must be a list")
    segments: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("development utterance must be an object")
        segments.append(
            {
                "speaker_id": item.get("speaker"),
                "speaker_name": item.get("speaker"),
                "text": item.get("text"),
            }
        )
    return normalize_transcript("", segments)


def validate_development_runtime_authorization(
    payload: Mapping[str, object],
    *,
    runtime_fingerprint: str,
    profile_fingerprint: str,
) -> None:
    authorization = payload.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("development runtime authorization is missing")
    if authorization.get("status") != "development_evaluation_qualified":
        raise ValueError("development runtime is not qualified")
    evidence = authorization.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("development runtime evidence is missing")
    if evidence.get("runtime_fingerprint") != runtime_fingerprint:
        raise ValueError("development runtime fingerprint drift")
    if evidence.get("profile_fingerprint") != profile_fingerprint:
        raise ValueError("development runtime profile drift")


class EvidenceEffectivenessReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_id: EvidenceFeature
    verdict: Literal["passed", "failed"]
    case_count: int = Field(gt=0)
    provider_requests: int = Field(ge=0)
    provider_request_limit: int = Field(gt=0)
    metrics: dict[str, float]
    hard_gates: dict[str, bool]
    runtime_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _digest(observations: Sequence[EvidenceObservation]) -> str:
    encoded = json.dumps(
        [item.model_dump(mode="json") for item in observations],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _extraction_metrics(
    observations: Sequence[ExtractionObservation],
) -> tuple[dict[str, float], dict[str, bool]]:
    references = sum(item.reference_total for item in observations)
    baseline_recall = _ratio(
        sum(item.baseline_reference_hits for item in observations), references
    )
    treatment_recall = _ratio(
        sum(item.treatment_reference_hits for item in observations), references
    )
    baseline_precision = _ratio(
        sum(item.baseline_supported_candidates for item in observations),
        sum(item.baseline_candidate_total for item in observations),
    )
    treatment_precision = _ratio(
        sum(item.treatment_supported_candidates for item in observations),
        sum(item.treatment_candidate_total for item in observations),
    )
    metrics = {
        "baseline_reference_recall": baseline_recall,
        "treatment_reference_recall": treatment_recall,
        "reference_recall_delta": treatment_recall - baseline_recall,
        "baseline_reference_precision": baseline_precision,
        "treatment_reference_precision": treatment_precision,
        "reference_precision_delta": treatment_precision - baseline_precision,
        "exact_evidence_citation_precision": _ratio(
            sum(item.treatment_citations_valid for item in observations),
            len(observations),
        ),
        "completion_and_validity": _ratio(
            sum(item.treatment_complete for item in observations), len(observations)
        ),
    }
    gates = {
        "complete_schedule": len(observations) == 12,
        "completion_and_validity": metrics["completion_and_validity"] == 1.0,
        "reference_recall_delta": metrics["reference_recall_delta"] >= 0.05,
        "reference_precision_delta": metrics["reference_precision_delta"] >= -0.01,
        "exact_evidence_citation_precision": metrics[
            "exact_evidence_citation_precision"
        ]
        == 1.0,
    }
    return metrics, gates


def _verification_metrics(
    observations: Sequence[VerificationObservation],
) -> tuple[dict[str, float], dict[str, bool]]:
    injected = [item for item in observations if item.injected_error]
    clean = [item for item in observations if not item.injected_error]
    caught = sum(item.treatment_status == "contradicted" for item in injected)
    clean_false_positives = sum(
        item.treatment_status == "contradicted" for item in clean
    )
    baseline_accepted = sum(item.baseline_accepted for item in injected)
    treatment_accepted = sum(
        item.treatment_status != "contradicted" for item in injected
    )
    reduction = _ratio(
        max(0, baseline_accepted - treatment_accepted), baseline_accepted
    )
    metrics = {
        "injected_error_recall": _ratio(caught, len(injected)),
        "clean_critical_false_positives": float(clean_false_positives),
        "unsupported_accepted_reduction": reduction,
        "evidence_citation_precision": _ratio(
            sum(item.citation_valid for item in observations), len(observations)
        ),
    }
    gates = {
        "complete_schedule": len(injected) == 12 and len(clean) == 12,
        "injected_error_recall": metrics["injected_error_recall"] == 1.0,
        "clean_critical_false_positives": clean_false_positives == 0,
        "unsupported_accepted_reduction": reduction >= 0.5,
        "evidence_citation_precision": metrics["evidence_citation_precision"] == 1.0,
    }
    return metrics, gates


def _consolidation_metrics(
    observations: Sequence[ConsolidationObservation],
) -> tuple[dict[str, float], dict[str, bool]]:
    expected_unique = sum(item.expected_unique_count for item in observations)
    observed_unique = sum(item.observed_unique_count for item in observations)
    redundant_before = sum(item.duplicate_candidate_count - 1 for item in observations)
    redundant_after = sum(item.duplicate_fact_count - 1 for item in observations)
    duplicate_reduction = _ratio(
        max(0, redundant_before - redundant_after), redundant_before
    )
    false_merges = sum(item.false_semantic_merges for item in observations)
    metrics = {
        "unique_fact_recall": _ratio(observed_unique, expected_unique),
        "correction_and_conflict_recall": _ratio(
            sum(item.relationships_preserved for item in observations),
            len(observations),
        ),
        "evidence_preservation": _ratio(
            sum(item.evidence_preserved for item in observations), len(observations)
        ),
        "false_semantic_merges": float(false_merges),
        "duplicate_reduction": duplicate_reduction,
    }
    gates = {
        "complete_schedule": len(observations) == 12,
        "unique_fact_recall": metrics["unique_fact_recall"] == 1.0,
        "correction_and_conflict_recall": metrics[
            "correction_and_conflict_recall"
        ]
        == 1.0,
        "evidence_preservation": metrics["evidence_preservation"] == 1.0,
        "false_semantic_merges": false_merges == 0,
        "duplicate_reduction": duplicate_reduction >= 0.8,
    }
    return metrics, gates


def build_effectiveness_report(
    feature_id: EvidenceFeature,
    observations: Sequence[EvidenceObservation],
    *,
    runtime_fingerprint: str,
    development_fingerprint: str,
    provider_request_limit: int,
    prior_provider_requests: int = 0,
) -> EvidenceEffectivenessReport:
    if not observations:
        raise ValueError("at least one observation is required")
    expected_type: type[EvidenceObservation]
    if feature_id == "evidence.cited_atomic_extraction":
        expected_type = ExtractionObservation
        if not all(isinstance(item, expected_type) for item in observations):
            raise TypeError("extraction observations required")
        metrics, gates = _extraction_metrics(observations)  # type: ignore[arg-type]
    elif feature_id == "evidence.source_verification":
        expected_type = VerificationObservation
        if not all(isinstance(item, expected_type) for item in observations):
            raise TypeError("verification observations required")
        metrics, gates = _verification_metrics(observations)  # type: ignore[arg-type]
    else:
        expected_type = ConsolidationObservation
        if not all(isinstance(item, expected_type) for item in observations):
            raise TypeError("consolidation observations required")
        metrics, gates = _consolidation_metrics(observations)  # type: ignore[arg-type]
    if prior_provider_requests < 0:
        raise ValueError("prior provider requests must be nonnegative")
    provider_requests = prior_provider_requests + sum(
        item.provider_requests for item in observations
    )
    gates = {
        **gates,
        "request_budget": provider_requests <= provider_request_limit,
        "qualified_runtime": bool(runtime_fingerprint),
        "qualified_development_set": bool(development_fingerprint),
    }
    return EvidenceEffectivenessReport(
        feature_id=feature_id,
        verdict="passed" if all(gates.values()) else "failed",
        case_count=len(observations),
        provider_requests=provider_requests,
        provider_request_limit=provider_request_limit,
        metrics=metrics,
        hard_gates=gates,
        runtime_fingerprint=runtime_fingerprint,
        development_fingerprint=development_fingerprint,
        result_digest=_digest(observations),
    )
