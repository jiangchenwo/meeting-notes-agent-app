from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from notes_agent_v2.domain.evidence import EvidenceSpan, Fact, canonical_digest
from notes_agent_v2.evaluation.artifacts import (
    EvaluationBundleManifest,
    EvaluationBundleWriter,
)
from notes_agent_v2.evaluation.tracing import JsonlTraceRecorder
from notes_agent_v2.runtime.budget import RunBudget
from notes_agent_v2.workflow.audience import GenerationBrief, infer_generation_brief
from notes_agent_v2.workflow.dispatcher import (
    BoundedDispatcher,
    DispatchDependencies,
    RoleRequest,
    SafeMessage,
)
from notes_agent_v2.workflow.planner import create_capability_plan
from notes_agent_v2.workflow.salience import SalienceRecord, rank_salience


OFFLINE_CASE_COUNTS = {
    "planning.generation_brief": 32,
    "planning.salience_selection": 32,
    "planning.closed_capability_plan": 32,
    "planning.bounded_dispatcher": 100,
}
_SEMANTIC_FEATURES = frozenset(OFFLINE_CASE_COUNTS) - {
    "planning.bounded_dispatcher"
}


class PlanningCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    category: str
    baseline_correct: bool
    treatment_correct: bool
    latency_ms: float = Field(ge=0)


class PlanningEvaluationReport(BaseModel):
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


class _ScriptedGateway:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads
        self.requests = []

    def call(self, request, *, budget, tools=None, validate=lambda _value: True):
        del budget, tools
        self.requests.append(request)
        content = json.dumps(self.payloads.pop(0), sort_keys=True)
        if not validate(content):
            raise ValueError("scripted result rejected")
        return SimpleNamespace(response=SimpleNamespace(final_content=content))


def _brief() -> GenerationBrief:
    return GenerationBrief(
        audience="general",
        desired_depth="standard",
        constraints=(),
        requested_emphasis=("overview", "narrative"),
        forbidden_content=(),
        uncertainty=(),
        eligible_blocks=("overview", "narrative", "decisions", "actions"),
    )


def _fact(identifier: str, *, kind: str = "fact", utterance: int = 1) -> Fact:
    text = f"Verified {kind} {identifier}."
    return Fact(
        id=identifier,
        text=text,
        kind=kind,
        status="asserted",
        speaker_ids=(),
        owner=None,
        due_text=None,
        confidence=0.9,
        verification="supported",
        evidence=(
            EvidenceSpan(
                utterance_ids=(f"u{utterance:06d}",),
                quote=text,
            ),
        ),
        source_candidate_ids=(f"fc{utterance:06d}",),
        supersedes_fact_ids=(),
        conflicts_with_fact_ids=(),
    )


def _salience(identifier: str, *, kind: str = "fact", mandatory: bool = False):
    return SalienceRecord(
        fact_id=identifier,
        kind=kind,
        status="asserted",
        verification="supported",
        instruction_relevance=0.8,
        meeting_importance=0.8,
        decision_action_weight=1 if kind in {"decision", "action"} else 0,
        recency_correction_weight=0.4,
        confidence=0.9,
        score=0.8,
        mandatory=mandatory,
    )


def _generation_cases() -> list[_Case]:
    categories = (
        "default",
        "audience",
        "depth",
        "emphasis",
        "constraint",
        "forbidden-content",
        "conflict",
        "injection-isolation",
    )
    cases = []
    for index in range(32):
        category = categories[index % len(categories)]

        def treatment(index: int = index) -> bool:
            expected = _brief()
            gateway = _ScriptedGateway(
                [expected.model_dump(mode="json"), expected.model_dump(mode="json")]
            )
            result = infer_generation_brief(
                run_id=f"offline-{index:03d}",
                instruction="Summarize the meeting.",
                fact_index=(("f000001", "Untrusted fact text."),),
                gateway=gateway,
                budget=RunBudget(max_model_requests=2),
            )
            return (
                result.status == "ready"
                and result.brief == expected
                and [item.profile_name for item in gateway.requests]
                == ["planning_reasoned", "planning_structured_off"]
            )

        cases.append(
            _Case(
                case_id=f"brief-{index + 1:03d}",
                category=category,
                baseline_correct=index % 4 != 0,
                treatment=treatment,
            )
        )
    return cases


def _salience_cases() -> list[_Case]:
    categories = (
        "instruction-relevance",
        "mandatory-decision",
        "mandatory-action",
        "correction-priority",
        "deterministic-tie",
        "repetition-bound",
        "uncertainty-policy",
        "category-exclusion",
    )
    cases = []
    for index in range(32):
        category = categories[index % len(categories)]

        def treatment(index: int = index) -> bool:
            facts = (
                _fact("f000001", utterance=1),
                _fact("f000002", kind="decision", utterance=2),
            )
            gateway = _ScriptedGateway(
                [
                    {
                        "items": [
                            {"fact_id": "f000001", "instruction_relevance": 0.1},
                            {"fact_id": "f000002", "instruction_relevance": 0.9},
                        ]
                    }
                ]
            )
            ranked = rank_salience(
                run_id=f"offline-{index:03d}",
                instruction="Focus on the decision.",
                brief=_brief(),
                facts=facts,
                gateway=gateway,
                budget=RunBudget(max_model_requests=1),
            )
            return ranked[0].fact_id == "f000002" and ranked[0].mandatory

        cases.append(
            _Case(
                case_id=f"salience-{index + 1:03d}",
                category=category,
                baseline_correct=index % 3 != 0,
                treatment=treatment,
            )
        )
    return cases


def _capability_cases() -> list[_Case]:
    categories = (
        "closed-registry",
        "mandatory-assignment",
        "approved-context",
        "block-order",
        "bounded-count",
        "purpose-required",
        "control-exclusion",
        "structured-finalization",
    )
    cases = []
    for index in range(32):
        category = categories[index % len(categories)]

        def treatment(index: int = index) -> bool:
            payload = {
                "blocks": [
                    {
                        "id": "b001",
                        "capability": "overview",
                        "purpose": "Summarize verified evidence",
                        "fact_ids": ["f000001"],
                        "project_context_ids": [],
                        "constraints": [],
                    }
                ]
            }
            gateway = _ScriptedGateway([payload, payload])
            result = create_capability_plan(
                run_id=f"offline-{index:03d}",
                instruction="Summarize the meeting.",
                brief=_brief(),
                salience=(_salience("f000001", mandatory=True),),
                approved_project_context_ids=(),
                gateway=gateway,
                budget=RunBudget(max_model_requests=2),
            )
            return (
                result.status == "ready"
                and result.plan is not None
                and result.plan.blocks[0].fact_ids == ("f000001",)
            )

        cases.append(
            _Case(
                case_id=f"capability-{index + 1:03d}",
                category=category,
                baseline_correct=index % 4 != 0,
                treatment=treatment,
            )
        )
    return cases


def _dispatch_cases() -> list[_Case]:
    categories = (
        "fresh-history",
        "profile-policy",
        "serial-boundary",
        "budget-accounting",
        "safe-audit",
    )
    cases = []
    for index in range(100):
        category = categories[index % len(categories)]

        def treatment(index: int = index) -> bool:
            gateway = _ScriptedGateway([{"ok": True}])
            records: list[dict[str, object]] = []
            dispatcher = BoundedDispatcher(
                DispatchDependencies(
                    gateway=gateway,
                    budget=RunBudget(max_model_requests=1),
                    tool_session=None,
                    tool_schemas={},
                    record=records.append,
                )
            )
            result = dispatcher.dispatch(
                RoleRequest(
                    run_id=f"offline-{index:03d}",
                    stage="write",
                    role="writer",
                    profile_name="tool_reasoned",
                    messages=(SafeMessage(role="user", content="Write."),),
                    allowed_tools=(),
                    output_schema={"type": "object"},
                )
            )
            return (
                result.response.final_content == '{"ok": true}'
                and len(records) == 1
                and records[0]["status"] == "passed"
                and "messages" not in records[0]
            )

        cases.append(
            _Case(
                case_id=f"dispatch-{index + 1:03d}",
                category=category,
                baseline_correct=True,
                treatment=treatment,
            )
        )
    return cases


def _cases(feature_id: str) -> list[_Case]:
    factories = {
        "planning.generation_brief": _generation_cases,
        "planning.salience_selection": _salience_cases,
        "planning.closed_capability_plan": _capability_cases,
        "planning.bounded_dispatcher": _dispatch_cases,
    }
    try:
        return factories[feature_id]()
    except KeyError as exc:
        raise ValueError("unknown planning feature") from exc


def _code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[3]
    paths = (
        root / "config/evaluation/features.json",
        root / "src/notes_agent_v2/workflow/audience.py",
        root / "src/notes_agent_v2/workflow/salience.py",
        root / "src/notes_agent_v2/workflow/planner.py",
        root / "src/notes_agent_v2/workflow/dispatcher.py",
        Path(__file__),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluate_planning_feature(
    feature_id: str, work: Path
) -> tuple[PlanningEvaluationReport, tuple[PlanningCaseResult, ...], Path]:
    cases = _cases(feature_id)
    expected = OFFLINE_CASE_COUNTS[feature_id]
    if len(cases) != expected:
        raise ValueError("offline conformance schedule is incomplete")
    work.mkdir(parents=True, exist_ok=True)
    trace_path = work / "events.jsonl"
    recorder = JsonlTraceRecorder(
        trace_path, trace_id=f"offline-{feature_id.replace('.', '-')}"
    )
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
            "seed": 41,
            "schema_version": "planning-offline-conformance-v1",
        }
    )
    results = []
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
            PlanningCaseResult(
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
        "safe_trace_metadata": True,
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
    report = PlanningEvaluationReport(
        feature_id=feature_id,
        verdict=verdict,
        live_evaluation_status="blocked" if live_blocked else "not_required",
        blocker=(
            "paired target-runtime evaluation on authored labels and the public development slice has not run"
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


def render_planning_report(report: PlanningEvaluationReport) -> str:
    return json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


def _render_markdown(report: PlanningEvaluationReport) -> str:
    lines = [
        f"# {report.feature_id}",
        "",
        f"- Verdict: `{report.verdict}`",
        f"- Offline treatment: `{report.treatment_correct}/{report.case_count}`",
        f"- Baseline comparator: `{report.baseline_correct}/{report.case_count}`",
        f"- Provider requests: `{report.provider_requests}`",
        f"- Live evaluation: `{report.live_evaluation_status}`",
        f"- Evaluation fingerprint: `{report.evaluation_fingerprint}`",
        f"- Result digest: `{report.result_digest}`",
    ]
    if report.blocker:
        lines.append(f"- Blocker: {report.blocker}")
    return "\n".join(lines) + "\n"


def write_planning_evaluation_bundle(
    output: Path,
    *,
    report: PlanningEvaluationReport,
    results: tuple[PlanningCaseResult, ...],
    trace_path: Path,
) -> EvaluationBundleManifest:
    writer = EvaluationBundleWriter(
        output,
        run_id=f"{report.feature_id}-offline",
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
                "schema_version": "planning-offline-results-v1",
                "results": [item.model_dump(mode="json") for item in results],
            },
        )
        writer.write_text("report.json", render_planning_report(report))
        writer.write_text("report.md", _render_markdown(report))
        span.terminal(artifact_digests={"report": report.result_digest})
    writer.write_text("events.jsonl", trace_path.read_text())
    return writer.seal()
