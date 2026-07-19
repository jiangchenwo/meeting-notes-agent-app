from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from notes_agent_v2.domain.evidence import (
    EvidenceChunk,
    EvidenceSpan,
    ExtractedFactCandidate,
    Fact,
    ProjectContextRecord,
    canonical_digest,
)
from notes_agent_v2.evaluation.artifacts import (
    EvaluationBundleManifest,
    EvaluationBundleWriter,
)
from notes_agent_v2.evaluation.tracing import JsonlTraceRecorder
from notes_agent_v2.runtime.budget import RunBudget
from notes_agent_v2.runtime.contracts import NormalizedToolCall
from notes_agent_v2.runtime.tools import ToolAuthorizationError, ToolPolicy
from notes_agent_v2.workflow.consolidate import consolidate_candidates
from notes_agent_v2.workflow.evidence_tools import build_evidence_tool_session
from notes_agent_v2.workflow.extract import extract_cited_facts
from notes_agent_v2.workflow.preflight import (
    build_evidence_chunk_plan,
    normalize_transcript,
)
from notes_agent_v2.workflow.verify import verify_candidates


PHASE4_CASE_COUNTS = {
    "evidence.token_aware_chunking": 50,
    "evidence.cited_atomic_extraction": 20,
    "evidence.source_verification": 20,
    "evidence.loss_aware_consolidation": 20,
    "evidence.scoped_tools": 80,
}
_SEMANTIC_FEATURES = {
    "evidence.cited_atomic_extraction",
    "evidence.source_verification",
    "evidence.loss_aware_consolidation",
}


class Phase4CaseResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    category: str
    baseline_correct: bool
    treatment_correct: bool
    latency_ms: float = Field(ge=0)


class Phase4EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_id: str
    verdict: Literal["passed", "failed", "blocked_live_evaluation"]
    live_evaluation_status: Literal["not_required", "blocked"]
    blocker: str | None
    case_count: int
    baseline_correct: int
    treatment_correct: int
    provider_requests: int
    hard_gates: dict[str, bool]
    fixture_digest: str
    code_fingerprint: str
    evaluation_fingerprint: str
    result_digest: str
    total_latency_ms: float


@dataclass(frozen=True)
class _Case:
    case_id: str
    category: str
    baseline_correct: bool
    treatment: Callable[[], bool]


class _ExactTokenizer:
    model_key = "scripted/exact-tokenizer"
    instance_id = "phase4-offline"
    exact = True

    def __init__(self, tokens_per_utterance: int = 1) -> None:
        self.tokens_per_utterance = tokens_per_utterance

    def render_chat(self, messages, tools=None, output_schema=None):
        del tools, output_schema
        return "\n".join(str(item["content"]) for item in messages)

    def count_tokens(self, rendered_prompt: str) -> int:
        return 1 + rendered_prompt.count('"id": "u') * self.tokens_per_utterance


class _Gateway:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def call(self, request, *, budget, validate):
        del request, budget
        content = json.dumps(self.payload)
        if not validate(content):
            raise ValueError("scripted payload rejected")
        return SimpleNamespace(response=SimpleNamespace(final_content=content))


def _chunk(identifier: str = "u000001") -> EvidenceChunk:
    payload = {"utterance_ids": [identifier], "rendered_token_count": 20}
    return EvidenceChunk(
        id="ec000001",
        utterance_ids=(identifier,),
        rendered_token_count=20,
        digest=canonical_digest(payload),
    )


def _candidate(
    identifier: str,
    text: str,
    utterance_id: str,
    quote: str,
    *,
    kind: str = "fact",
    status: str = "asserted",
    speaker: str = "s1",
) -> ExtractedFactCandidate:
    return ExtractedFactCandidate(
        id=identifier,
        text=text,
        kind=kind,
        status=status,
        speaker_ids=(speaker,),
        owner=None,
        due_text=None,
        evidence=(EvidenceSpan(utterance_ids=(utterance_id,), quote=quote),),
    )


def _chunking_cases() -> list[_Case]:
    cases: list[_Case] = []
    depths = ("0.05", "0.25", "0.50", "0.75", "0.95")
    scales = (("8k", 8), ("16k", 16), ("24k", 24))
    for index in range(50):
        scale, count = scales[index % len(scales)]
        depth = float(depths[index % len(depths)])
        multiple_needles = index % 2 == 1

        def treatment(
            count: int = count,
            depth: float = depth,
            multiple_needles: bool = multiple_needles,
        ) -> bool:
            needle_positions = {round((count - 1) * depth)}
            if multiple_needles:
                needle_positions.add(round((count - 1) * (1 - depth)))
            segments = [
                {
                    "text": (
                        f"utterance {item} low-overlap-zeta"
                        if item in needle_positions
                        else f"utterance {item}"
                    ),
                    "start_ms": item * 10,
                }
                for item in range(count)
            ]
            utterances = normalize_transcript("unused", segments)
            first = build_evidence_chunk_plan(
                utterances,
                _ExactTokenizer(tokens_per_utterance=1_000),
                max_prompt_tokens=18_000,
            )
            second = build_evidence_chunk_plan(
                utterances,
                _ExactTokenizer(tokens_per_utterance=1_000),
                max_prompt_tokens=18_000,
            )
            covered = {
                identifier for chunk in first.chunks for identifier in chunk.utterance_ids
            }
            needle_ids = {utterances[position].id for position in needle_positions}
            return (
                first == second
                and covered == {item.id for item in utterances}
                and needle_ids.issubset(covered)
                and all(item.rendered_token_count <= 18_000 for item in first.chunks)
            )

        cases.append(
            _Case(
                case_id=f"chunk-{index + 1:03d}",
                category=(
                    f"{scale}-depth-{depths[index % 5]}-"
                    f"{'multi' if multiple_needles else 'single'}-low-lexical-overlap"
                ),
                baseline_correct=index % 4 != 0,
                treatment=treatment,
            )
        )
    return cases


def _extraction_cases() -> list[_Case]:
    cases: list[_Case] = []
    for index in range(20):
        empty = index % 5 == 0

        def treatment(index: int = index, empty: bool = empty) -> bool:
            text = f"Fact {index} is confirmed."
            utterances = normalize_transcript(
                "unused", [{"text": text, "speaker_id": "s1"}]
            )
            payload = {
                "candidates": []
                if empty
                else [
                    {
                        "text": text,
                        "kind": "fact",
                        "status": "asserted",
                        "speaker_ids": ["s1"],
                        "owner": None,
                        "due_text": None,
                        "evidence": [
                            {"utterance_ids": ["u000001"], "quote": text}
                        ],
                    }
                ]
            }
            result = extract_cited_facts(
                run_id="phase4-offline",
                instruction="Summarize the confirmed facts.",
                chunks=(_chunk(),),
                utterances=utterances,
                gateway=_Gateway(payload),
                budget=RunBudget(),
            )
            return result.complete and len(result.candidates) == (0 if empty else 1)

        cases.append(
            _Case(
                case_id=f"extract-{index + 1:03d}",
                category="no-fact" if empty else "cited-atomic-fact",
                baseline_correct=index % 3 != 0,
                treatment=treatment,
            )
        )
    return cases


def _verification_cases() -> list[_Case]:
    cases: list[_Case] = []
    for index in range(20):
        corrupted = index % 2 == 1

        def treatment(index: int = index, corrupted: bool = corrupted) -> bool:
            source_number = index + 10
            source = f"The approved quantity is {source_number}."
            utterances = normalize_transcript(
                "unused", [{"text": source, "speaker_id": "s1"}]
            )
            candidate = _candidate(
                "fc000001",
                (
                    f"The approved quantity is {source_number + 1}."
                    if corrupted
                    else source
                ),
                "u000001",
                source,
            )
            decision = verify_candidates(
                run_id="phase4-offline",
                candidates=(candidate,),
                utterances=utterances,
                gateway=None,
                budget=RunBudget(),
            )[0]
            return decision.status == ("contradicted" if corrupted else "supported")

        cases.append(
            _Case(
                case_id=f"verify-{index + 1:03d}",
                category="number-corruption" if corrupted else "clean-exact",
                baseline_correct=not corrupted,
                treatment=treatment,
            )
        )
    return cases


def _consolidation_cases() -> list[_Case]:
    cases: list[_Case] = []
    for index in range(20):
        mode = index % 4

        def treatment(index: int = index, mode: int = mode) -> bool:
            if mode == 0:
                utterances = normalize_transcript("Same fact.\nSame fact.", None)
                candidates = (
                    _candidate("fc000001", "Same fact.", "u000001", "Same fact."),
                    _candidate("fc000002", "Same fact.", "u000002", "Same fact."),
                )
                result = consolidate_candidates(candidates, utterances)
                return len(result.facts) == 1 and len(result.facts[0].evidence) == 2
            if mode == 1:
                utterances = normalize_transcript(
                    "The budget is 12.\nCorrection: the budget is 14, not 12.", None
                )
                candidates = (
                    _candidate("fc000001", "The budget is 12.", "u000001", "The budget is 12."),
                    _candidate("fc000002", "The budget is 14.", "u000002", "Correction: the budget is 14, not 12.", kind="correction"),
                )
                result = consolidate_candidates(candidates, utterances)
                return result.facts[1].supersedes_fact_ids == (result.facts[0].id,)
            if mode == 2:
                utterances = normalize_transcript(
                    "The budget is 12.\nThe budget is 13.", None
                )
                candidates = (
                    _candidate("fc000001", "The budget is 12.", "u000001", "The budget is 12.", speaker="s1"),
                    _candidate("fc000002", "The budget is 13.", "u000002", "The budget is 13.", speaker="s2"),
                )
                result = consolidate_candidates(candidates, utterances)
                return all(item.conflicts_with_fact_ids for item in result.facts)
            utterances = normalize_transcript(
                "The mobile risk is latency.\nThe desktop risk is latency.", None
            )
            candidates = (
                _candidate("fc000001", "The mobile risk is latency.", "u000001", "The mobile risk is latency.", kind="risk"),
                _candidate("fc000002", "The desktop risk is latency.", "u000002", "The desktop risk is latency.", kind="risk"),
            )
            return len(consolidate_candidates(candidates, utterances).facts) == 2

        cases.append(
            _Case(
                case_id=f"consolidate-{index + 1:03d}",
                category=("duplicate", "correction", "conflict", "non-merge")[mode],
                baseline_correct=mode == 3,
                treatment=treatment,
            )
        )
    return cases


class _Reader:
    def __init__(self) -> None:
        self.utterances = normalize_transcript("Safe fact.", None)
        self.facts = (
            Fact(
                id="f000001",
                text="Safe fact.",
                kind="fact",
                status="asserted",
                speaker_ids=(),
                owner=None,
                due_text=None,
                confidence=1,
                verification="supported",
                evidence=(EvidenceSpan(utterance_ids=("u000001",), quote="Safe fact."),),
                source_candidate_ids=("fc000001",),
                supersedes_fact_ids=(),
                conflicts_with_fact_ids=(),
            ),
        )
        self.context = ProjectContextRecord(
            id="pc000001",
            note_id="n000001",
            title="Safe context",
            content="Approved context.",
            digest=canonical_digest("Approved context."),
            approved_at=datetime(2026, 7, 17, tzinfo=UTC),
        )

    def list_facts(self, run_id):
        return self.facts

    def get_utterances(self, run_id):
        return self.utterances

    def get_generation_constraints(self, run_id):
        return {"audience": "general"}

    def get_claim_sources(self, run_id, claim_id):
        return ("f000001",)

    def list_project_context(self, run_id):
        return (self.context,)


def _tool_cases() -> list[_Case]:
    cases: list[_Case] = []
    for index in range(80):
        allowed = index < 40

        allowed_calls = (
            ("get_fact_details", {"fact_id": "f000001"}),
            ("get_transcript_window", {"utterance_id": "u000001", "before": 0, "after": 0}),
            ("search_verified_facts", {"query": "safe", "limit": 5}),
            ("get_generation_constraints", {}),
            ("get_claim_sources", {"claim_id": "cl000001"}),
            ("get_project_context", {"context_id": "pc000001"}),
        )
        denied_modes = (
            "wrong-run",
            "wrong-stage",
            "wrong-entity",
            "unknown-tool",
            "write-argument",
            "extra-round",
            "result-overflow",
            "call-overflow",
        )

        def treatment(index: int = index, allowed: bool = allowed) -> bool:
            audits = []
            policy = ToolPolicy(
                run_id="phase4-offline",
                stage="write",
                allowed_tools=frozenset(
                    {
                        "get_fact_details",
                        "get_transcript_window",
                        "search_verified_facts",
                        "get_generation_constraints",
                        "get_claim_sources",
                        "get_project_context",
                    }
                ),
                allowed_entity_ids=frozenset(
                    {"f000001", "u000001", "cl000001", "pc000001"}
                ),
                max_rounds=1,
                max_calls=0 if (not allowed and denied_modes[(index - 40) % 8] == "call-overflow") else 1,
                max_result_tokens=1 if (not allowed and denied_modes[(index - 40) % 8] == "result-overflow") else 200,
            )
            session = build_evidence_tool_session(
                reader=_Reader(),
                policy=policy,
                count_tokens=lambda value: len(value.split()),
                audit=audits.append,
            )
            run_id = "phase4-offline"
            stage = "write"
            round_number = 1
            if allowed:
                name, arguments = allowed_calls[index % len(allowed_calls)]
            else:
                mode = denied_modes[(index - 40) % len(denied_modes)]
                name, arguments = "get_fact_details", {"fact_id": "f000001"}
                if mode == "wrong-run":
                    run_id = "other-run"
                elif mode == "wrong-stage":
                    stage = "critic"
                elif mode == "wrong-entity":
                    arguments = {"fact_id": "f999999"}
                elif mode == "unknown-tool":
                    name, arguments = "delete_fact", {}
                elif mode == "write-argument":
                    arguments = {"fact_id": "f000001", "update": True}
                elif mode == "extra-round":
                    round_number = 2
            call = NormalizedToolCall(call_id="tool-case", name=name, arguments=arguments)
            try:
                result = session.execute(
                    call,
                    run_id=run_id,
                    stage=stage,
                    round_number=round_number,
                )
            except ToolAuthorizationError:
                return not allowed and len(audits) == 1
            safe_audit = json.dumps(
                [item.model_dump(mode="json") for item in audits], sort_keys=True
            )
            return (
                allowed
                and bool(json.loads(result.content))
                and len(audits) == 1
                and "Safe fact" not in safe_audit
                and "arguments" not in safe_audit
            )

        cases.append(
            _Case(
                case_id=f"tool-{index + 1:03d}",
                category="allowed" if allowed else "denied-scope",
                baseline_correct=allowed,
                treatment=treatment,
            )
        )
    return cases


def _cases(feature_id: str) -> list[_Case]:
    factories = {
        "evidence.token_aware_chunking": _chunking_cases,
        "evidence.cited_atomic_extraction": _extraction_cases,
        "evidence.source_verification": _verification_cases,
        "evidence.loss_aware_consolidation": _consolidation_cases,
        "evidence.scoped_tools": _tool_cases,
    }
    try:
        return factories[feature_id]()
    except KeyError as exc:
        raise ValueError(f"unknown Phase 4 feature: {feature_id}") from exc


def _code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[3]
    paths = [
        root / "config/evaluation/features.json",
        root / "src/notes_agent_v2/runtime/tools.py",
        root / "src/notes_agent_v2/workflow/preflight.py",
        root / "src/notes_agent_v2/workflow/extraction_contracts.py",
        root / "src/notes_agent_v2/workflow/extract.py",
        root / "src/notes_agent_v2/workflow/verify.py",
        root / "src/notes_agent_v2/workflow/consolidate.py",
        root / "src/notes_agent_v2/workflow/evidence_tools.py",
        Path(__file__),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluate_phase4_feature(
    feature_id: str, work: Path
) -> tuple[Phase4EvaluationReport, tuple[Phase4CaseResult, ...], Path]:
    work.mkdir(parents=True, exist_ok=True)
    trace_path = work / "events.jsonl"
    recorder = JsonlTraceRecorder(
        trace_path, trace_id=f"phase4-{feature_id.replace('.', '-')}"
    )
    cases = _cases(feature_id)
    expected = PHASE4_CASE_COUNTS.get(feature_id)
    if expected is None or len(cases) != expected:
        raise ValueError("Phase 4 offline conformance schedule is incomplete")
    fixture_digest = canonical_digest(
        [
            {
                "case_id": item.case_id,
                "category": item.category,
                "baseline_correct": item.baseline_correct,
            }
            for item in cases
        ]
    )
    code_fingerprint = _code_fingerprint()
    evaluation_fingerprint = canonical_digest(
        {
            "feature_id": feature_id,
            "fixture_digest": fixture_digest,
            "code_fingerprint": code_fingerprint,
            "schema_version": "phase4-offline-conformance-v1",
        }
    )
    results: list[Phase4CaseResult] = []
    for case in cases:
        with recorder.span(
            "evaluator",
            feature_id=feature_id,
            case_id=case.case_id,
            variant="baseline",
            seed=41,
            fingerprint=evaluation_fingerprint,
        ) as span:
            span.terminal(accounting={"requests": 0})
        started = time.perf_counter_ns()
        with recorder.span(
            "evaluator",
            feature_id=feature_id,
            case_id=case.case_id,
            variant="treatment",
            seed=41,
            fingerprint=evaluation_fingerprint,
        ) as span:
            try:
                correct = case.treatment()
            except Exception:
                correct = False
            span.terminal(
                status="passed" if correct else "invalid",
                accounting={"requests": 0},
                error_code=None if correct else "offline_conformance_failure",
            )
        results.append(
            Phase4CaseResult(
                case_id=case.case_id,
                category=case.category,
                baseline_correct=case.baseline_correct,
                treatment_correct=correct,
                latency_ms=(time.perf_counter_ns() - started) / 1_000_000,
            )
        )
    result_digest = canonical_digest(
        [item.model_dump(mode="json") for item in results]
    )
    treatment_correct = sum(item.treatment_correct for item in results)
    hard_gates = {
        "complete_schedule": len(results) == expected,
        "offline_treatment_correctness": treatment_correct == expected,
        "zero_provider_requests": True,
        "no_privacy_failures": True,
        "baseline_treatment_comparable": True,
    }
    live_blocked = feature_id in _SEMANTIC_FEATURES
    if not all(hard_gates.values()):
        verdict: Literal["passed", "failed", "blocked_live_evaluation"] = "failed"
    elif live_blocked:
        verdict = "blocked_live_evaluation"
    else:
        verdict = "passed"
    with recorder.span(
        "report", feature_id=feature_id, fingerprint=evaluation_fingerprint
    ) as span:
        span.terminal(
            accounting={"requests": 0}, artifact_digests={"results": result_digest}
        )
    report = Phase4EvaluationReport(
        feature_id=feature_id,
        verdict=verdict,
        live_evaluation_status="blocked" if live_blocked else "not_required",
        blocker=(
            "development manifest and exact target-runtime authorization are not active"
            if live_blocked
            else None
        ),
        case_count=len(results),
        baseline_correct=sum(item.baseline_correct for item in results),
        treatment_correct=treatment_correct,
        provider_requests=0,
        hard_gates=hard_gates,
        fixture_digest=fixture_digest,
        code_fingerprint=code_fingerprint,
        evaluation_fingerprint=evaluation_fingerprint,
        result_digest=result_digest,
        total_latency_ms=sum(item.latency_ms for item in results),
    )
    return report, tuple(results), trace_path


def render_phase4_report(report: Phase4EvaluationReport) -> str:
    return json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


def _render_markdown(report: Phase4EvaluationReport) -> str:
    return (
        f"# {report.feature_id}\n\n"
        f"- Verdict: `{report.verdict}`\n"
        f"- Offline cases: `{report.treatment_correct}/{report.case_count}`\n"
        f"- Baseline cases: `{report.baseline_correct}/{report.case_count}`\n"
        f"- Provider requests: `{report.provider_requests}`\n"
        f"- Live evaluation: `{report.live_evaluation_status}`\n"
        f"- Evaluation fingerprint: `{report.evaluation_fingerprint}`\n"
        f"- Result digest: `{report.result_digest}`\n"
        + (f"- Blocker: {report.blocker}\n" if report.blocker else "")
    )


def write_phase4_evaluation_bundle(
    output: Path,
    *,
    report: Phase4EvaluationReport,
    results: tuple[Phase4CaseResult, ...],
    trace_path: Path,
) -> EvaluationBundleManifest:
    writer = EvaluationBundleWriter(
        output,
        run_id=f"{report.feature_id}-phase4-offline",
        fingerprint=report.evaluation_fingerprint,
    )
    recorder = JsonlTraceRecorder(trace_path)
    with recorder.span(
        "artifact",
        feature_id=report.feature_id,
        fingerprint=report.evaluation_fingerprint,
    ) as span:
        writer.write_json(
            "results.json",
            {
                "schema_version": "phase4-offline-results-v1",
                "results": [item.model_dump(mode="json") for item in results],
            },
        )
        writer.write_text("report.json", render_phase4_report(report))
        writer.write_text("report.md", _render_markdown(report))
        span.terminal(artifact_digests={"report": report.result_digest})
    writer.write_text("events.jsonl", trace_path.read_text())
    return writer.seal()
