from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import urlparse

import lmstudio
from pydantic import BaseModel, ConfigDict

from notes_agent_v2.domain.evidence import EvidenceSpan, ExtractedFactCandidate
from notes_agent_v2.evaluation.artifacts import EvaluationBundleWriter
from notes_agent_v2.evaluation.evidence_effectiveness import (
    ConsolidationObservation,
    EvidenceFeature,
    ExtractionObservation,
    VerificationObservation,
    build_effectiveness_report,
    candidate_from_reference,
    development_utterances,
    inject_polarity_defect,
    valid_verification_citation,
    validate_development_runtime_authorization,
)
from notes_agent_v2.evaluation.tracing import JsonlTraceRecorder
from notes_agent_v2.runtime.budget import RunBudget
from notes_agent_v2.runtime.context import (
    LMStudioSDKPromptTokenizer,
    get_loaded_lm_studio_model,
)
from notes_agent_v2.runtime.contracts import RuntimeReport, assert_runtime_ready
from notes_agent_v2.runtime.gateway import GatewayDependencies, GatewayRequest, RuntimeGateway
from notes_agent_v2.runtime.http_provider import OpenAICompatibleRuntimeProvider
from notes_agent_v2.runtime.lm_studio import LMStudioControlClient
from notes_agent_v2.runtime.profiles import ProfileCatalog
from notes_agent_v2.runtime.settings import load_runtime_settings
from notes_agent_v2.workflow.consolidate import consolidate_candidates
from notes_agent_v2.workflow.extract import extract_cited_facts
from notes_agent_v2.workflow.preflight import build_evidence_chunks, normalize_transcript
from notes_agent_v2.workflow.verify import verify_candidates


FEATURE_LIMITS: dict[EvidenceFeature, int] = {
    "evidence.cited_atomic_extraction": 102,
    "evidence.source_verification": 48,
    "evidence.loss_aware_consolidation": 24,
}
CASE_IDS = tuple(
    [f"ami-structured-{index:02d}" for index in range(1, 9)]
    + [f"long-context-{index:02d}" for index in range(1, 5)]
)


class GroupingPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    groups: tuple[tuple[str, ...], ...]


class GatewaySemanticGrouper:
    def __init__(self, gateway: RuntimeGateway, budget: RunBudget, run_id: str) -> None:
        self.gateway = gateway
        self.budget = budget
        self.run_id = run_id

    def propose(
        self, candidates: tuple[ExtractedFactCandidate, ...]
    ) -> tuple[tuple[str, ...], ...]:
        request = GatewayRequest(
            run_id=self.run_id,
            stage="consolidate",
            role="semantic_fact_grouper",
            profile_name="evaluation_structured_off",
            messages=(
                {
                    "role": "system",
                    "content": (
                        "Group only candidate IDs that express the same atomic claim. "
                        "Do not group corrections, conflicts, or merely related facts."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        [
                            {
                                "id": item.id,
                                "text": item.text,
                                "kind": item.kind,
                                "status": item.status,
                            }
                            for item in candidates
                        ],
                        sort_keys=True,
                    ),
                },
            ),
            output_schema=GroupingPayload.model_json_schema(),
        )
        result = self.gateway.call(
            request,
            budget=self.budget,
            validate=lambda value: _valid_model(GroupingPayload, value),
        )
        return GroupingPayload.model_validate_json(result.response.final_content).groups


def _valid_model(model: type[BaseModel], content: str) -> bool:
    try:
        model.model_validate_json(content)
    except Exception:
        return False
    return True


def _load_cases(root: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for case_id in CASE_IDS:
        case = json.loads((root / "cases" / f"{case_id}.json").read_text())
        label = json.loads((root / "labels" / f"{case_id}.json").read_text())
        if case.get("case_id") != case_id or label.get("case_id") != case_id:
            raise ValueError(f"case identity mismatch: {case_id}")
        if case.get("transcript_sha256") != label.get("transcript_sha256"):
            raise ValueError(f"case transcript mismatch: {case_id}")
        cases.append({"case": case, "label": label})
    return cases


def _utterances(case: dict[str, object]):
    return development_utterances(case)


def _normalized_evidence_id(identifier: str) -> str:
    return f"u{int(identifier[1:]):06d}"


def _references(label: dict[str, object]) -> list[dict[str, object]]:
    selected = []
    for item in label["references"]:  # type: ignore[index]
        if item["role"] not in {"decision_summary", "action_summary", "issue_summary"}:
            continue
        selected.append(
            {
                **item,
                "evidence_ids": tuple(
                    _normalized_evidence_id(value) for value in item["evidence_ids"]
                ),
            }
        )
    if not selected:
        selected = [
            {
                **item,
                "evidence_ids": tuple(
                    _normalized_evidence_id(value) for value in item["evidence_ids"]
                ),
            }
            for item in label["references"]  # type: ignore[index]
        ]
    return selected


def _candidate_evidence_ids(candidate: ExtractedFactCandidate) -> set[str]:
    return {
        identifier
        for span in candidate.evidence
        for identifier in span.utterance_ids
    }


def _coverage(
    candidates: tuple[ExtractedFactCandidate, ...], references: list[dict[str, object]]
) -> tuple[int, int]:
    reference_sets = [set(item["evidence_ids"]) for item in references]
    hits = sum(
        any(_candidate_evidence_ids(candidate) & reference for candidate in candidates)
        for reference in reference_sets
    )
    supported = sum(
        any(_candidate_evidence_ids(candidate) & reference for reference in reference_sets)
        for candidate in candidates
    )
    return hits, supported


def _diagnostic_chunk(
    utterances,
    *,
    evidence_ids: set[str] | None,
    tokenizer: LMStudioSDKPromptTokenizer,
    instruction: str,
):
    if evidence_ids:
        positions = [
            index for index, item in enumerate(utterances) if item.id in evidence_ids
        ]
        if not positions:
            raise ValueError("reference evidence is outside the transcript")
        center = max(positions)
    else:
        center = 0
    start = max(0, center - 20)
    stop = min(len(utterances), center + 21)
    chunks = build_evidence_chunks(
        utterances[start:stop],
        tokenizer,
        max_prompt_tokens=2_000,
        overlap_utterances=4,
        instruction=instruction,
    )
    if not evidence_ids:
        return chunks[0]
    return max(
        chunks,
        key=lambda item: len(set(item.utterance_ids) & evidence_ids),
    )


def _evaluate_extraction(
    cases: list[dict[str, object]],
    gateway: RuntimeGateway,
    tokenizer: LMStudioSDKPromptTokenizer,
    budget: RunBudget,
    recorder: JsonlTraceRecorder,
    checkpoint_path: Path,
) -> tuple[list[ExtractionObservation], list[dict[str, object]]]:
    instruction = (
        "Extract at most five highest-salience decisions, actions, corrections, "
        "or unresolved issues from this chunk."
    )
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        observations = [
            ExtractionObservation.model_validate(item)
            for item in checkpoint.get("observations", [])
        ]
        details = list(checkpoint.get("details", []))
    else:
        observations = []
        details = []
    completed = {item.case_id for item in observations}
    for loaded in cases:
        case = loaded["case"]
        label = loaded["label"]
        case_id = str(case["case_id"])
        if case_id in completed:
            continue
        utterances = _utterances(case)
        references = _references(label)
        target_reference = max(
            references,
            key=lambda item: max(
                int(identifier[1:]) for identifier in item["evidence_ids"]
            ),
        )
        references = [target_reference]
        baseline_chunk = _diagnostic_chunk(
            utterances,
            evidence_ids=None,
            tokenizer=tokenizer,
            instruction=instruction,
        )
        target_ids = set(target_reference["evidence_ids"])
        target_chunk = _diagnostic_chunk(
            utterances,
            evidence_ids=target_ids,
            tokenizer=tokenizer,
            instruction=instruction,
        )
        before = budget.model_requests
        with recorder.span(
            "evaluator", feature_id="evidence.cited_atomic_extraction", case_id=case_id
        ) as span:
            baseline = extract_cited_facts(
                run_id="evidence-extraction-live",
                instruction=instruction,
                chunks=(baseline_chunk,),
                utterances=utterances,
                gateway=gateway,
                budget=budget,
                profile_name="evaluation_structured_off",
            )
            treatment = extract_cited_facts(
                run_id="evidence-extraction-live",
                instruction=instruction,
                chunks=(target_chunk,),
                utterances=utterances,
                gateway=gateway,
                budget=budget,
                profile_name="evaluation_structured_off",
            )
            requests = budget.model_requests - before
            span.terminal(
                status="passed" if baseline.complete and treatment.complete else "invalid",
                accounting={"requests": requests},
            )
        baseline_hits, baseline_supported = _coverage(
            baseline.candidates, references
        )
        treatment_hits, treatment_supported = _coverage(
            treatment.candidates, references
        )
        observations.append(
            ExtractionObservation(
                case_id=case_id,
                baseline_reference_hits=baseline_hits,
                treatment_reference_hits=treatment_hits,
                reference_total=len(references),
                baseline_supported_candidates=baseline_supported,
                baseline_candidate_total=len(baseline.candidates),
                treatment_supported_candidates=treatment_supported,
                treatment_candidate_total=len(treatment.candidates),
                treatment_citations_valid=treatment.complete,
                treatment_complete=baseline.complete and treatment.complete,
                provider_requests=requests,
            )
        )
        details.append(
            {
                "case_id": case_id,
                "baseline_chunk_id": baseline_chunk.id,
                "target_chunk_id": target_chunk.id,
                "reference_ids": [item["reference_id"] for item in references],
                "baseline": baseline.model_dump(mode="json"),
                "treatment": treatment.model_dump(mode="json"),
            }
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps(
                {
                    "observations": [
                        item.model_dump(mode="json") for item in observations
                    ],
                    "details": details,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"completed {case_id}: {requests} requests", flush=True)
    return observations, details


def _evaluate_verification(
    cases: list[dict[str, object]],
    gateway: RuntimeGateway,
    budget: RunBudget,
    recorder: JsonlTraceRecorder,
) -> tuple[list[VerificationObservation], list[dict[str, object]]]:
    observations = []
    details = []
    for index, loaded in enumerate(cases, start=1):
        case = loaded["case"]
        label = loaded["label"]
        case_id = str(case["case_id"])
        utterances = _utterances(case)
        utterance_by_id = {item.id: item for item in utterances}
        reference = _references(label)[0]
        clean = candidate_from_reference(
            f"fc{index * 2 - 1:06d}", reference, utterance_by_id
        )
        corrupted = candidate_from_reference(
            f"fc{index * 2:06d}",
            reference,
            utterance_by_id,
            text=inject_polarity_defect(clean.text),
        )
        for injected, candidate in ((False, clean), (True, corrupted)):
            before = budget.model_requests
            observation_id = f"{case_id}-{'injected' if injected else 'clean'}"
            with recorder.span(
                "evaluator",
                feature_id="evidence.source_verification",
                case_id=observation_id,
            ) as span:
                decision = verify_candidates(
                    run_id="evidence-verification-live",
                    candidates=(candidate,),
                    utterances=utterances,
                    gateway=gateway,
                    budget=budget,
                    profile_name="evaluation_structured_off",
                )[0]
                requests = budget.model_requests - before
                span.terminal(
                    status=(
                        "passed"
                        if (not injected or decision.status == "contradicted")
                        else "invalid"
                    ),
                    accounting={"requests": requests},
                )
            observations.append(
                VerificationObservation(
                    case_id=observation_id,
                    injected_error=injected,
                    baseline_accepted=True,
                    treatment_status=decision.status,
                    citation_valid=valid_verification_citation(
                        decision.evidence_ids,
                        candidate.evidence[0].utterance_ids,
                    ),
                    provider_requests=requests,
                )
            )
            details.append(
                {
                    "case_id": observation_id,
                    "candidate": candidate.model_dump(mode="json"),
                    "decision": decision.model_dump(mode="json"),
                }
            )
        print(f"completed {case_id}: verification pair", flush=True)
    return observations, details


def _next_utterance_id(utterances) -> str:
    return f"u{len(utterances) + 1:06d}"


def _relationship_candidates(index: int, utterances, mode: int):
    start = len(utterances) + 1
    if mode == 0:
        segments = [
            {"text": "The approved budget is 12.", "speaker_id": "s1"},
            {"text": "Correction: the approved budget is 14, not 12.", "speaker_id": "s1"},
        ]
        kind_two = "correction"
    elif mode == 1:
        segments = [
            {"text": "The approved budget is 12.", "speaker_id": "s1"},
            {"text": "The approved budget is 13.", "speaker_id": "s2"},
        ]
        kind_two = "fact"
    else:
        segments = [
            {"text": "The mobile risk is latency.", "speaker_id": "s1"},
            {"text": "The desktop risk is battery life.", "speaker_id": "s2"},
        ]
        kind_two = "risk"
    extra = normalize_transcript("", segments)
    shifted = tuple(
        item.model_copy(update={"id": f"u{start + offset:06d}"})
        for offset, item in enumerate(extra)
    )
    first, second = shifted
    base = index * 10
    candidates = (
        ExtractedFactCandidate(
            id=f"fc{base + 3:06d}", text=first.text, kind="risk" if mode == 2 else "fact",
            status="asserted", speaker_ids=("s1",), owner=None, due_text=None,
            evidence=(EvidenceSpan(utterance_ids=(first.id,), quote=first.text),),
        ),
        ExtractedFactCandidate(
            id=f"fc{base + 4:06d}", text=second.text, kind=kind_two,
            status="asserted", speaker_ids=("s1" if mode == 0 else "s2",), owner=None, due_text=None,
            evidence=(EvidenceSpan(utterance_ids=(second.id,), quote=second.text),),
        ),
    )
    return shifted, candidates


def _evaluate_consolidation(
    cases: list[dict[str, object]],
    gateway: RuntimeGateway,
    budget: RunBudget,
    recorder: JsonlTraceRecorder,
) -> tuple[list[ConsolidationObservation], list[dict[str, object]]]:
    observations = []
    details = []
    for index, loaded in enumerate(cases, start=1):
        case = loaded["case"]
        label = loaded["label"]
        case_id = str(case["case_id"])
        utterances = _utterances(case)
        utterance_by_id = {item.id: item for item in utterances}
        reference = _references(label)[0]
        first = candidate_from_reference(
            f"fc{index * 10 + 1:06d}", reference, utterance_by_id
        )
        paraphrase = candidate_from_reference(
            f"fc{index * 10 + 2:06d}",
            reference,
            utterance_by_id,
            text=f"A confirmed outcome from the discussion was this: {first.text}",
        )
        extra_utterances, relation_candidates = _relationship_candidates(
            index, utterances, (index - 1) % 3
        )
        combined_utterances = tuple(utterances) + extra_utterances
        candidates = (first, paraphrase, *relation_candidates)
        before = budget.model_requests
        with recorder.span(
            "evaluator",
            feature_id="evidence.loss_aware_consolidation",
            case_id=case_id,
        ) as span:
            result = consolidate_candidates(
                candidates,
                combined_utterances,
                semantic_grouper=GatewaySemanticGrouper(
                    gateway, budget, "evidence-consolidation-live"
                ),
            )
            requests = budget.model_requests - before
            span.terminal(accounting={"requests": requests})
        duplicate_fact_ids = {
            result.candidate_to_fact[first.id], result.candidate_to_fact[paraphrase.id]
        }
        expected_groups = {
            frozenset((first.id, paraphrase.id)),
            frozenset((relation_candidates[0].id,)),
            frozenset((relation_candidates[1].id,)),
        }
        observed_groups = {
            frozenset(item.source_candidate_ids) for item in result.facts
        }
        false_merges = len([group for group in observed_groups if group not in expected_groups])
        evidence_before = {
            span.model_dump_json() for candidate in candidates for span in candidate.evidence
        }
        evidence_after = {
            span.model_dump_json() for fact in result.facts for span in fact.evidence
        }
        mode = (index - 1) % 3
        if mode == 0:
            relationships = any(item.supersedes_fact_ids for item in result.facts)
        elif mode == 1:
            relationships = sum(bool(item.conflicts_with_fact_ids) for item in result.facts) == 2
        else:
            relationships = all(
                not item.supersedes_fact_ids and not item.conflicts_with_fact_ids
                for item in result.facts
            )
        observations.append(
            ConsolidationObservation(
                case_id=case_id,
                baseline_fact_count=len(candidates),
                treatment_fact_count=len(result.facts),
                duplicate_candidate_count=2,
                duplicate_fact_count=len(duplicate_fact_ids),
                expected_unique_count=3,
                observed_unique_count=len(result.facts),
                evidence_preserved=evidence_before.issubset(evidence_after),
                relationships_preserved=relationships,
                false_semantic_merges=false_merges,
                provider_requests=requests,
            )
        )
        details.append(
            {
                "case_id": case_id,
                "candidates": [item.model_dump(mode="json") for item in candidates],
                "result": result.model_dump(mode="json"),
            }
        )
        print(f"completed {case_id}: {requests} requests", flush=True)
    return observations, details


def _markdown(report) -> str:
    lines = [
        f"# {report.feature_id}",
        "",
        f"- Verdict: `{report.verdict}`",
        f"- Cases: `{report.case_count}`",
        f"- Provider requests: `{report.provider_requests}/{report.provider_request_limit}`",
        f"- Runtime fingerprint: `{report.runtime_fingerprint}`",
        f"- Development fingerprint: `{report.development_fingerprint}`",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in sorted(report.metrics.items()))
    lines.extend(("", "## Hard gates", ""))
    lines.extend(f"- {key}: `{value}`" for key, value in sorted(report.hard_gates.items()))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate live evidence effectiveness.")
    parser.add_argument("--feature", required=True, choices=tuple(FEATURE_LIMITS))
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--runtime-authorization", required=True, type=Path)
    parser.add_argument("--development-set-authorization", required=True, type=Path)
    parser.add_argument("--development-runtime-authorization", required=True, type=Path)
    parser.add_argument("--development-root", required=True, type=Path)
    parser.add_argument("--private-out", required=True, type=Path)
    parser.add_argument("--prior-bundle", type=Path)
    parser.add_argument("--additional-prior-requests", type=int, default=0)
    parser.add_argument("--allow-live", action="store_true")
    args = parser.parse_args()
    if not args.allow_live:
        parser.error("live model calls require --allow-live")
    if args.private_out.exists():
        parser.error("private output already exists")

    settings = load_runtime_settings(env_file=args.env_file)
    runtime_report = RuntimeReport.model_validate_json(
        args.runtime_authorization.read_text()
    )
    assert_runtime_ready(runtime_report)
    development = json.loads(args.development_set_authorization.read_text())
    if development.get("status") != "development_set_qualified":
        raise ValueError("development set is not qualified")
    development_runtime = json.loads(
        args.development_runtime_authorization.read_text()
    )
    validate_development_runtime_authorization(
        development_runtime,
        runtime_fingerprint=runtime_report.fingerprint,
        profile_fingerprint=hashlib.sha256(settings.profiles_path.read_bytes()).hexdigest(),
    )
    manifest = json.loads((args.development_root / "manifest.json").read_text())
    if (
        manifest.get("selection_digest") != development.get("selection_digest")
        or manifest.get("case_count") != development.get("case_count")
    ):
        raise ValueError("development authorization does not match the selected root")
    cases = _load_cases(args.development_root)
    prior_provider_requests = 0
    prior_summary: dict[str, object] | None = None
    if args.prior_bundle is not None:
        prior_report = json.loads((args.prior_bundle / "report.json").read_text())
        if prior_report.get("feature_id") != args.feature:
            raise ValueError("prior bundle feature mismatch")
        prior_provider_requests = int(prior_report.get("provider_requests", -1))
        if not 0 <= prior_provider_requests < FEATURE_LIMITS[args.feature]:
            raise ValueError("prior bundle request accounting is invalid")
        prior_summary = {
            "provider_requests": prior_provider_requests,
            "result_digest": prior_report.get("result_digest"),
            "verdict": prior_report.get("verdict"),
        }
    if args.additional_prior_requests < 0:
        raise ValueError("additional prior requests must be nonnegative")
    prior_provider_requests += args.additional_prior_requests
    if prior_provider_requests >= FEATURE_LIMITS[args.feature]:
        raise ValueError("prior requests exhaust the feature ceiling")
    if args.additional_prior_requests:
        prior_summary = {
            **(prior_summary or {}),
            "additional_aborted_requests": args.additional_prior_requests,
            "aggregate_provider_requests": prior_provider_requests,
        }

    with tempfile.TemporaryDirectory(prefix="evidence-effectiveness-") as temporary:
        trace_path = Path(temporary) / "events.jsonl"
        recorder = JsonlTraceRecorder(
            trace_path,
            trace_id=f"live-{args.feature.replace('.', '-')}",
        )
        records: list[dict[str, object]] = []
        host = urlparse(str(settings.control_base_url)).netloc
        sdk_client = lmstudio.Client(host)
        try:
            model = get_loaded_lm_studio_model(sdk_client)
            tokenizer = LMStudioSDKPromptTokenizer(
                model,
                model_key=runtime_report.identity.model_key,
                instance_id=runtime_report.identity.instance_id,
                loaded_context=runtime_report.identity.loaded_context,
            )
            gateway = RuntimeGateway(
                GatewayDependencies(
                    control=LMStudioControlClient(
                        str(settings.control_base_url).rstrip("/"),
                        api_token=settings.api_token,
                        timeout_seconds=settings.control_timeout_seconds,
                    ),
                    runtime_report=lambda: runtime_report,
                    tokenizer=tokenizer,
                    provider=OpenAICompatibleRuntimeProvider(
                        base_url=str(settings.inference_base_url),
                        api_token=settings.api_token,
                    ),
                    profiles=ProfileCatalog.from_path(settings.profiles_path),
                    expected_model=settings.model,
                    context_envelope=settings.context,
                    inference_timeout_seconds=settings.inference_timeout_seconds,
                    record=records.append,
                )
            )
            limit = FEATURE_LIMITS[args.feature]
            budget = RunBudget(
                max_model_requests=limit - prior_provider_requests,
                max_wall_seconds=7_200,
            )
            if args.feature == "evidence.cited_atomic_extraction":
                observations, details = _evaluate_extraction(
                    cases,
                    gateway,
                    tokenizer,
                    budget,
                    recorder,
                    args.private_out.parent
                    / f".{args.private_out.name}.checkpoint.json",
                )
            elif args.feature == "evidence.source_verification":
                observations, details = _evaluate_verification(
                    cases, gateway, budget, recorder
                )
            else:
                observations, details = _evaluate_consolidation(
                    cases, gateway, budget, recorder
                )
        finally:
            sdk_client.close()

        report = build_effectiveness_report(
            args.feature,
            observations,
            runtime_fingerprint=runtime_report.fingerprint,
            development_fingerprint=str(development["tree_sha256"]),
            provider_request_limit=FEATURE_LIMITS[args.feature],
            prior_provider_requests=prior_provider_requests,
        )
        with recorder.span(
            "report", feature_id=args.feature, fingerprint=report.result_digest
        ) as span:
            span.terminal(
                status="passed" if report.verdict == "passed" else "invalid",
                accounting={"requests": report.provider_requests},
                artifact_digests={"results": report.result_digest},
            )
        writer = EvaluationBundleWriter(
            args.private_out,
            run_id=f"{args.feature}-live",
            fingerprint=report.result_digest,
        )
        writer.write_json("report.json", report.model_dump(mode="json"))
        writer.write_text("report.md", _markdown(report))
        writer.write_json(
            "observations.json",
            [item.model_dump(mode="json") for item in observations],
        )
        writer.write_json("details.json", details)
        if prior_summary is not None:
            writer.write_json("prior-run.json", prior_summary)
        writer.write_json("safe-runtime-records.json", records)
        writer.write_text("events.jsonl", trace_path.read_text())
        writer.seal()
    print(report.verdict)
    return 0 if report.verdict == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
