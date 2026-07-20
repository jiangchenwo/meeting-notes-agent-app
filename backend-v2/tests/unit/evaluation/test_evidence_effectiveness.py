from __future__ import annotations

import pytest

from notes_agent_v2.evaluation.evidence_effectiveness import (
    ConsolidationObservation,
    ExtractionObservation,
    VerificationObservation,
    build_effectiveness_report,
    candidate_from_reference,
    development_utterances,
    inject_polarity_defect,
    valid_verification_citation,
    validate_development_runtime_authorization,
)
from notes_agent_v2.workflow.preflight import normalize_transcript


def test_verification_candidate_preserves_every_gold_evidence_utterance() -> None:
    utterances = normalize_transcript(
        "unused",
        [{"text": "First supporting turn."}, {"text": "Second supporting turn."}],
    )
    utterance_by_id = {item.id: item for item in utterances}

    candidate = candidate_from_reference(
        "fc000001",
        {
            "text": "Combined supported claim.",
            "role": "decision_summary",
            "evidence_ids": ("u000001", "u000002"),
        },
        utterance_by_id,
    )

    assert candidate.evidence[0].utterance_ids == ("u000001", "u000002")
    assert candidate.evidence[0].quote == (
        "First supporting turn.\nSecond supporting turn."
    )


def test_verification_citation_accepts_only_nonempty_evidence_subsets() -> None:
    assert valid_verification_citation(("u000002",), ("u000001", "u000002"))
    assert not valid_verification_citation((), ("u000001", "u000002"))
    assert not valid_verification_citation(
        ("u000003",), ("u000001", "u000002")
    )


@pytest.mark.parametrize(
    ("clean", "expected"),
    [
        ("The team will not use teletext.", "The team will use teletext."),
        ("The remote will use rubber.", "The remote will not use rubber."),
        ("The remote uses a battery.", "The remote does not use a battery."),
        (
            "The group decided to use a scroll button.",
            "The group decided not to use a scroll button.",
        ),
    ],
)
def test_polarity_injection_is_direct_and_materially_different(
    clean: str, expected: str
) -> None:
    injected = inject_polarity_defect(clean)

    assert injected == expected
    assert injected.casefold() != clean.casefold()


def test_extraction_report_requires_recall_gain_without_precision_regression() -> None:
    observations = tuple(
        ExtractionObservation(
            case_id=f"case-{index}",
            baseline_reference_hits=1,
            treatment_reference_hits=2,
            reference_total=2,
            baseline_supported_candidates=2,
            baseline_candidate_total=2,
            treatment_supported_candidates=2,
            treatment_candidate_total=2,
            treatment_citations_valid=True,
            treatment_complete=True,
            provider_requests=3,
        )
        for index in range(12)
    )

    report = build_effectiveness_report(
        "evidence.cited_atomic_extraction",
        observations,
        runtime_fingerprint="a" * 64,
        development_fingerprint="b" * 64,
        provider_request_limit=64,
    )

    assert report.verdict == "passed"
    assert report.metrics["reference_recall_delta"] == 0.5
    assert report.metrics["reference_precision_delta"] == 0.0


def test_verification_report_fails_when_an_injected_error_is_accepted() -> None:
    observations = tuple(
        VerificationObservation(
            case_id=f"case-{index}",
            injected_error=index < 12,
            baseline_accepted=True,
            treatment_status=(
                "supported" if index == 0 else "contradicted" if index < 12 else "supported"
            ),
            citation_valid=True,
            provider_requests=1,
        )
        for index in range(24)
    )

    report = build_effectiveness_report(
        "evidence.source_verification",
        observations,
        runtime_fingerprint="a" * 64,
        development_fingerprint="b" * 64,
        provider_request_limit=48,
    )

    assert report.verdict == "failed"
    assert report.hard_gates["injected_error_recall"] is False


def test_consolidation_report_tracks_evidence_and_false_merges() -> None:
    observations = tuple(
        ConsolidationObservation(
            case_id=f"case-{index}",
            baseline_fact_count=4,
            treatment_fact_count=3,
            duplicate_candidate_count=2,
            duplicate_fact_count=1,
            expected_unique_count=3,
            observed_unique_count=3,
            evidence_preserved=True,
            relationships_preserved=True,
            false_semantic_merges=0,
            provider_requests=1,
        )
        for index in range(12)
    )

    report = build_effectiveness_report(
        "evidence.loss_aware_consolidation",
        observations,
        runtime_fingerprint="a" * 64,
        development_fingerprint="b" * 64,
        provider_request_limit=24,
    )

    assert report.verdict == "passed"
    assert report.metrics["duplicate_reduction"] == 1.0
    assert report.metrics["false_semantic_merges"] == 0.0


def test_report_fails_closed_when_request_ceiling_is_exceeded() -> None:
    observations = (
        ExtractionObservation(
            case_id="case-1",
            baseline_reference_hits=1,
            treatment_reference_hits=1,
            reference_total=1,
            baseline_supported_candidates=1,
            baseline_candidate_total=1,
            treatment_supported_candidates=1,
            treatment_candidate_total=1,
            treatment_citations_valid=True,
            treatment_complete=True,
            provider_requests=65,
        ),
    )

    report = build_effectiveness_report(
        "evidence.cited_atomic_extraction",
        observations,
        runtime_fingerprint="a" * 64,
        development_fingerprint="b" * 64,
        provider_request_limit=64,
    )

    assert report.verdict == "failed"
    assert report.hard_gates["request_budget"] is False


def test_report_accounts_for_prior_failed_live_requests() -> None:
    observations = tuple(
        ExtractionObservation(
            case_id=f"case-{index}",
            baseline_reference_hits=0,
            treatment_reference_hits=1,
            reference_total=1,
            baseline_supported_candidates=0,
            baseline_candidate_total=1,
            treatment_supported_candidates=1,
            treatment_candidate_total=1,
            treatment_citations_valid=True,
            treatment_complete=True,
            provider_requests=2,
        )
        for index in range(12)
    )

    report = build_effectiveness_report(
        "evidence.cited_atomic_extraction",
        observations,
        runtime_fingerprint="a" * 64,
        development_fingerprint="b" * 64,
        provider_request_limit=64,
        prior_provider_requests=38,
    )

    assert report.provider_requests == 62
    assert report.hard_gates["request_budget"] is True


def test_development_utterances_ignore_nonmonotonic_source_timing() -> None:
    utterances = development_utterances(
        {
            "utterances": [
                {"speaker": "A", "text": "First", "start_ms": 10, "end_ms": 30},
                {"speaker": "B", "text": "Second", "start_ms": 20, "end_ms": 25},
            ]
        }
    )

    assert [item.id for item in utterances] == ["u000001", "u000002"]
    assert [item.speaker_name for item in utterances] == ["A", "B"]
    assert [item.speaker_id for item in utterances] == ["A", "B"]
    assert all(item.start_ms is None and item.end_ms is None for item in utterances)


def test_live_authorization_binds_runtime_and_profile_fingerprints() -> None:
    payload = {
        "authorization": {
            "status": "development_evaluation_qualified",
            "evidence": {
                "runtime_fingerprint": "a" * 64,
                "profile_fingerprint": "b" * 64,
            },
        }
    }

    validate_development_runtime_authorization(
        payload,
        runtime_fingerprint="a" * 64,
        profile_fingerprint="b" * 64,
    )


def test_live_authorization_rejects_profile_drift() -> None:
    payload = {
        "authorization": {
            "status": "development_evaluation_qualified",
            "evidence": {
                "runtime_fingerprint": "a" * 64,
                "profile_fingerprint": "b" * 64,
            },
        }
    }

    try:
        validate_development_runtime_authorization(
            payload,
            runtime_fingerprint="a" * 64,
            profile_fingerprint="c" * 64,
        )
    except ValueError as exc:
        assert "profile" in str(exc)
    else:
        raise AssertionError("profile drift was accepted")
