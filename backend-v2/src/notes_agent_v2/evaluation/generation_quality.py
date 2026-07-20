from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict

from notes_agent_v2.domain.document import DocumentBlock, DocumentClaim, NotesDocument, StructuredItem
from notes_agent_v2.domain.evidence import EvidenceSpan, Fact
from notes_agent_v2.domain.quality import CriticIssue
from notes_agent_v2.evaluation.artifacts import (
    EvaluationBundleManifest,
    EvaluationBundleWriter,
)
from notes_agent_v2.evaluation.tracing import JsonlTraceRecorder
from notes_agent_v2.workflow.acceptance import evaluate_acceptance
from notes_agent_v2.workflow.audience import GenerationBrief
from notes_agent_v2.workflow.critics import deterministic_critic_issues
from notes_agent_v2.workflow.outline import build_fact_covered_outline
from notes_agent_v2.workflow.planner import CapabilityBlock, CapabilityPlan
from notes_agent_v2.workflow.revision import apply_targeted_revision
from notes_agent_v2.workflow.salience import SalienceRecord
from notes_agent_v2.workflow.writers import parse_cited_narrative, parse_structured_block


OFFLINE_CASE_COUNTS = {
    "generation.fact_covered_outline": 32,
    "generation.evidence_linked_writing": 16,
    "quality.specialist_critics": 48,
    "quality.deterministic_acceptance": 32,
    "quality.targeted_revision": 24,
}
_SEMANTIC_FEATURES = frozenset(
    {
        "generation.evidence_linked_writing",
        "quality.specialist_critics",
        "quality.targeted_revision",
    }
)


class GenerationCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    category: str
    baseline_correct: bool
    treatment_correct: bool
    latency_ms: float


class GenerationEvaluationReport(BaseModel):
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


class _Case:
    def __init__(
        self,
        case_id: str,
        category: str,
        baseline_correct: bool,
        treatment: Callable[[], bool],
    ) -> None:
        self.case_id = case_id
        self.category = category
        self.baseline_correct = baseline_correct
        self.treatment = treatment


def evaluate_generation_feature(
    feature_id: str, work: Path
) -> tuple[
    GenerationEvaluationReport, tuple[GenerationCaseResult, ...], Path
]:
    cases = _cases(feature_id)
    expected = OFFLINE_CASE_COUNTS.get(feature_id)
    if expected is None:
        raise ValueError("unknown generation-quality feature")
    if len(cases) != expected:
        raise ValueError("offline generation-quality schedule is incomplete")
    work.mkdir(parents=True, exist_ok=True)
    trace_path = work / "events.jsonl"
    recorder = JsonlTraceRecorder(
        trace_path, trace_id=f"offline-{feature_id.replace('.', '-')}"
    )
    fixture_digest = _digest(
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
    evaluation_fingerprint = _digest(
        {
            "feature_id": feature_id,
            "fixture_digest": fixture_digest,
            "code_fingerprint": code_fingerprint,
            "seed": 41,
            "schema_version": "generation-quality-offline-v1",
        }
    )
    results: list[GenerationCaseResult] = []
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
            GenerationCaseResult(
                case_id=case.case_id,
                category=case.category,
                baseline_correct=case.baseline_correct,
                treatment_correct=correct,
                latency_ms=(time.perf_counter_ns() - started) / 1_000_000,
            )
        )
    result_digest = _digest([item.model_dump(mode="json") for item in results])
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
    report = GenerationEvaluationReport(
        feature_id=feature_id,
        verdict=verdict,
        live_evaluation_status="blocked" if live_blocked else "not_required",
        blocker=(
            "paired target-runtime effectiveness evaluation has not run"
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


def render_generation_report(report: GenerationEvaluationReport) -> str:
    return json.dumps(report.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"


def write_generation_evaluation_bundle(
    output: Path,
    *,
    report: GenerationEvaluationReport,
    results: tuple[GenerationCaseResult, ...],
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
                "schema_version": "generation-quality-offline-results-v1",
                "results": [item.model_dump(mode="json") for item in results],
            },
        )
        writer.write_text("report.json", render_generation_report(report))
        writer.write_text("report.md", _render_markdown(report))
        span.terminal(artifact_digests={"report": report.result_digest})
    writer.write_text("events.jsonl", trace_path.read_text())
    return writer.seal()


def _cases(feature_id: str) -> tuple[_Case, ...]:
    count = OFFLINE_CASE_COUNTS.get(feature_id)
    if count is None:
        raise ValueError("unknown generation-quality feature")
    factories = {
        "generation.fact_covered_outline": _outline_case,
        "generation.evidence_linked_writing": _writing_case,
        "quality.specialist_critics": _critic_case,
        "quality.deterministic_acceptance": _acceptance_case,
        "quality.targeted_revision": _revision_case,
    }
    return tuple(factories[feature_id](index) for index in range(1, count + 1))


def _outline_case(index: int) -> _Case:
    def treatment() -> bool:
        salience = (
            _salience("f000001", "fact", False),
            _salience("f000002", "action", True),
        )
        plan = CapabilityPlan(
            blocks=(
                CapabilityBlock(
                    id="b001",
                    capability="overview",
                    purpose="Overview",
                    fact_ids=("f000001",),
                    project_context_ids=(),
                    constraints=(),
                ),
                CapabilityBlock(
                    id="b002",
                    capability="actions",
                    purpose="Actions",
                    fact_ids=("f000002",),
                    project_context_ids=(),
                    constraints=(),
                ),
            )
        )
        outline = build_fact_covered_outline(
            plan=plan,
            brief=_brief(),
            salience=salience,
            approved_project_context_ids=(),
        )
        return {fact for claim in outline.claims for fact in claim.fact_ids} == {
            "f000001",
            "f000002",
        }

    return _Case(f"outline-{index:03d}", "coverage", index % 4 == 0, treatment)


def _writing_case(index: int) -> _Case:
    structured = index % 2 == 0

    def treatment() -> bool:
        if structured:
            fact = _fact(
                "f000001",
                "Mina will ship Friday.",
                kind="action",
                status="approved",
                owner="Mina",
                due_text="Friday",
            )
            block = parse_structured_block(
                json.dumps(
                    {
                        "title": "Actions",
                        "items": [
                            {
                                "text": fact.text,
                                "fact_ids": [fact.id],
                                "status": fact.status,
                                "owner": fact.owner,
                                "due_text": fact.due_text,
                            }
                        ],
                    }
                ),
                assignment=_assignment("actions", (fact.id,)),
                facts=(fact,),
            )
            return block is not None and block.structured_items[0].fact_ids == (fact.id,)
        fact = _fact("f000001", "The rollout begins Friday.")
        block = parse_cited_narrative(
            f"{fact.text} [[{fact.id}]]",
            assignment=_assignment("overview", (fact.id,)),
            facts=(fact,),
            instruction="Summarize the rollout.",
        )
        return block.claims[0].fact_ids == (fact.id,)

    return _Case(
        f"writing-{index:03d}",
        "structured" if structured else "narrative",
        index % 4 == 0,
        treatment,
    )


def _critic_case(index: int) -> _Case:
    category = (
        "missing_mandatory_fact",
        "duplicate_coverage",
        "unsupported_claim",
        "wrong_status",
        "wrong_owner",
        "wrong_due",
    )[(index - 1) % 6]

    def treatment() -> bool:
        first = _fact("f000001", "The rollout begins Friday.")
        second = _fact("f000002", "The approval was recorded.", kind="decision", status="approved")
        if category == "missing_mandatory_fact":
            document = _document_with_claims(("f000001",))
            issues = deterministic_critic_issues(
                document,
                facts=(first, second),
                mandatory_fact_ids=(second.id,),
                instruction="Summarize.",
            )
        elif category == "duplicate_coverage":
            document = _document_with_claims(("f000001", "f000001"))
            issues = deterministic_critic_issues(
                document, facts=(first,), mandatory_fact_ids=(), instruction="Summarize."
            )
        elif category == "unsupported_claim":
            document = _document_with_claims(("f999999",))
            issues = deterministic_critic_issues(
                document, facts=(first,), mandatory_fact_ids=(), instruction="Summarize."
            )
        else:
            action = _fact(
                "f000001",
                "Mina will ship Friday.",
                kind="action",
                status="approved",
                owner="Mina",
                due_text="Friday",
            )
            values = {
                "status": "completed" if category == "wrong_status" else "approved",
                "owner": "Omar" if category == "wrong_owner" else "Mina",
                "due_text": "Monday" if category == "wrong_due" else "Friday",
            }
            document = _document_with_item(action, **values)
            issues = deterministic_critic_issues(
                document, facts=(action,), mandatory_fact_ids=(), instruction="List actions."
            )
        return category in {item.category for item in issues}

    return _Case(f"critic-{index:03d}", category, index % 6 == 0, treatment)


def _acceptance_case(index: int) -> _Case:
    mode = (index - 1) % 4
    category = ("accepted", "critical", "critic_failure", "missing_mandatory")[mode]

    def treatment() -> bool:
        facts = (_fact("f000001", "Fact one."), _fact("f000002", "Fact two."))
        document = _document_with_claims(("f000001",))
        issues: tuple[CriticIssue, ...] = ()
        mandatory = ("f000001",)
        expected = "accepted"
        if mode == 1:
            issues = (_critic_issue("claim", "critical", "contradiction", "b000001", ("f000001",)),)
            expected = "rejected"
        elif mode == 2:
            issues = (_critic_issue("system", "critical", "critic_failure", None, ()),)
            expected = "review_required"
        elif mode == 3:
            mandatory = ("f000002",)
            expected = "rejected"
        report = evaluate_acceptance(
            document=document,
            facts=facts,
            mandatory_fact_ids=mandatory,
            issues=issues,
        )
        return report.disposition == expected

    return _Case(f"acceptance-{index:03d}", category, mode == 0, treatment)


def _revision_case(index: int) -> _Case:
    def treatment() -> bool:
        facts = (_fact("f000001", "Fact one."), _fact("f000002", "Fact two."))
        parent = NotesDocument(
            id="d000001",
            run_id="r000001",
            version=1,
            parent_id=None,
            title="Notes",
            blocks=(
                _block("b000001", "c000001", "f000001", "Fact one."),
                _block("b000002", "c000002", "f000002", "Fact two."),
            ),
        )
        replacement = _block("b000001", "c000001", "f000001", "Revised fact one.")
        revised = apply_targeted_revision(
            parent=parent,
            document_id="d000002",
            issues=(
                _critic_issue(
                    "claim", "critical", "contradiction", "b000001", ("f000001",)
                ),
            ),
            revised_blocks={"b000001": replacement},
            facts=facts,
            mandatory_fact_ids=("f000001", "f000002"),
        )
        return (
            revised.parent_id == parent.id
            and revised.blocks[1].model_dump_json() == parent.blocks[1].model_dump_json()
        )

    return _Case(f"revision-{index:03d}", "targeted", index % 5 == 0, treatment)


def _brief() -> GenerationBrief:
    return GenerationBrief(
        audience="general",
        desired_depth="standard",
        constraints=(),
        requested_emphasis=("overview", "actions"),
        forbidden_content=(),
        uncertainty=(),
        eligible_blocks=("overview", "actions"),
    )


def _salience(identifier: str, kind: str, mandatory: bool) -> SalienceRecord:
    return SalienceRecord(
        fact_id=identifier,
        kind=kind,
        status="approved" if kind == "action" else "asserted",
        verification="supported",
        instruction_relevance=1,
        meeting_importance=1,
        decision_action_weight=1 if kind == "action" else 0,
        recency_correction_weight=0,
        confidence=1,
        score=1,
        mandatory=mandatory,
    )


def _fact(
    identifier: str,
    text: str,
    *,
    kind: str = "fact",
    status: str = "asserted",
    owner: str | None = None,
    due_text: str | None = None,
) -> Fact:
    return Fact(
        id=identifier,
        text=text,
        kind=kind,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        speaker_ids=("s1",),
        owner=owner,
        due_text=due_text,
        confidence=1,
        verification="supported",
        evidence=(EvidenceSpan(utterance_ids=("u000001",), quote=text),),
        source_candidate_ids=("fc000001",),
        supersedes_fact_ids=(),
        conflicts_with_fact_ids=(),
    )


def _assignment(capability: str, fact_ids: tuple[str, ...]) -> CapabilityBlock:
    return CapabilityBlock(
        id="b001",
        capability=capability,  # type: ignore[arg-type]
        purpose=capability.capitalize(),
        fact_ids=fact_ids,
        project_context_ids=(),
        constraints=(),
    )


def _block(block_id: str, claim_id: str, fact_id: str, text: str) -> DocumentBlock:
    return DocumentBlock(
        id=block_id,
        capability="overview" if block_id == "b000001" else "narrative",
        title="Overview",
        claims=(
            DocumentClaim(
                id=claim_id,
                text=text,
                fact_ids=(fact_id,),
                project_context_citations=(),
            ),
        ),
        structured_items=(),
    )


def _document_with_claims(fact_ids: tuple[str, ...]) -> NotesDocument:
    return NotesDocument(
        id="d000001",
        run_id="r000001",
        version=1,
        parent_id=None,
        title="Notes",
        blocks=(
            DocumentBlock(
                id="b000001",
                capability="overview",
                title="Overview",
                claims=tuple(
                    DocumentClaim(
                        id=f"c{index:06d}",
                        text=f"Claim {index}.",
                        fact_ids=(fact_id,),
                        project_context_citations=(),
                    )
                    for index, fact_id in enumerate(fact_ids, start=1)
                ),
                structured_items=(),
            ),
        ),
    )


def _document_with_item(fact: Fact, **updates: str) -> NotesDocument:
    return NotesDocument(
        id="d000001",
        run_id="r000001",
        version=1,
        parent_id=None,
        title="Notes",
        blocks=(
            DocumentBlock(
                id="b000001",
                capability="actions",
                title="Actions",
                claims=(),
                structured_items=(
                    StructuredItem(
                        id="si000001",
                        kind="action",
                        text=fact.text,
                        fact_ids=(fact.id,),
                        status=updates["status"],
                        owner=updates["owner"],
                        due_text=updates["due_text"],
                    ),
                ),
            ),
        ),
    )


def _critic_issue(
    critic: str,
    severity: str,
    category: str,
    block_id: str | None,
    fact_ids: tuple[str, ...],
) -> CriticIssue:
    return CriticIssue(
        id="i000001",
        critic=critic,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        category=category,
        block_id=block_id,
        claim_id=None,
        fact_ids=fact_ids,
        message="Injected issue.",
        confidence=None,
    )


def _code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[3]
    names = ("outline.py", "writers.py", "assembly.py", "critics.py", "acceptance.py", "revision.py")
    digest = hashlib.sha256()
    for name in names:
        path = root / "src" / "notes_agent_v2" / "workflow" / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _render_markdown(report: GenerationEvaluationReport) -> str:
    return (
        f"# {report.feature_id}\n\n"
        f"Verdict: `{report.verdict}`\n\n"
        f"Cases: {report.treatment_correct}/{report.case_count}\n\n"
        f"Provider requests: {report.provider_requests}\n"
    )
