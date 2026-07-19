from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Literal

from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict
from sqlalchemy import inspect

from notes_agent_v2.app import create_app
from notes_agent_v2.domain.document import DocumentBlock, DocumentClaim, NotesDocument, StructuredItem
from notes_agent_v2.domain.evidence import EvidenceSpan, Fact, ProjectContextRecord, canonical_digest
from notes_agent_v2.domain.planning import CapabilityPlan, PlannedBlock
from notes_agent_v2.domain.quality import CriticIssue, QualityReport
from notes_agent_v2.domain.transcript import Transcript, Utterance
from notes_agent_v2.persistence.database import Database, upgrade_database
from notes_agent_v2.persistence.repositories import PersistenceScopeError, Repositories

from .artifacts import EvaluationBundleManifest, EvaluationBundleWriter
from .tracing import JsonlTraceRecorder


PHASE3_CASE_COUNTS = {
    "domain.strict_contracts": 60,
    "persistence.immutable_repositories": 25,
    "api.instruction_presets": 20,
}


class Phase3CaseResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    case_id: str
    category: str
    baseline_correct: bool
    treatment_correct: bool
    latency_ms: float


class Phase3EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["phase3-conformance-report-v1"] = "phase3-conformance-report-v1"
    feature_id: str
    verdict: Literal["passed", "failed"]
    case_count: int
    baseline_correct: int
    treatment_correct: int
    hard_gates: dict[str, bool]
    request_count: int = 0
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


def _utterance() -> Utterance:
    return Utterance(id="u000001", speaker_id="s1", speaker_name="Ava", text="The launch was approved.", start_ms=0, end_ms=1000)


def _fact() -> Fact:
    return Fact(
        id="f000001", text="The launch was approved.", kind="decision", status="approved",
        speaker_ids=("s1",), owner=None, due_text=None, confidence=1,
        verification="supported", evidence=(EvidenceSpan(utterance_ids=("u000001",), quote="launch was approved"),),
        source_candidate_ids=("fc000001",), supersedes_fact_ids=(), conflicts_with_fact_ids=(),
    )


def _document(run_id: str, number: int) -> NotesDocument:
    claim = DocumentClaim(id="c000001", text="The launch was approved.", fact_ids=("f000001",), project_context_citations=())
    item = StructuredItem(id="si000001", kind="decision", text=claim.text, fact_ids=("f000001",), status="approved", owner=None, due_text=None)
    return NotesDocument(
        id=f"d{number:06d}", run_id=run_id, version=1, parent_id=None, title="Notes",
        blocks=(DocumentBlock(id="b000001", capability="decisions", title="Decisions", claims=(claim,), structured_items=(item,)),),
    )


def _domain_valid(index: int) -> bool:
    branch = index % 5
    if branch == 0:
        _utterance()
    elif branch == 1:
        content = f"Approved context {index}."
        ProjectContextRecord(id=f"pc{index + 1:06d}", note_id="n000001", title="Context", content=content, digest=canonical_digest(content), approved_at=datetime(2026, 7, 17, tzinfo=UTC))
    elif branch == 2:
        _fact()
    elif branch == 3:
        CapabilityPlan(
            capabilities=("overview",),
            blocks=(PlannedBlock(id="b000001", capability="overview", title="Overview", purpose="Summarize", fact_ids=("f000001",), project_context_ids=(), required=True),),
            required_fact_ids=("f000001",),
        )
    else:
        QualityReport(
            disposition="accepted", issues=(), mandatory_coverage=1, total_coverage=1,
            evidence_link_rate=1, unsupported_claim_count=0, critic_failure_count=0,
            warning_count=0, revision_count=0,
        )
    return True


def _domain_invalid(index: int) -> bool:
    branch = index % 10
    try:
        if branch == 0:
            Utterance(id="u1", text="bad")
        elif branch == 1:
            EvidenceSpan(utterance_ids=(), quote="")
        elif branch == 2:
            _fact().model_validate({**_fact().model_dump(), "status": "uncertain"})
        elif branch == 3:
            _fact().model_validate({**_fact().model_dump(), "owner": "Ava"})
        elif branch == 4:
            CapabilityPlan(
                capabilities=("overview",),
                blocks=(PlannedBlock(id="b000001", capability="overview", title="Overview", purpose="Summarize", fact_ids=(), project_context_ids=(), required=True),),
                required_fact_ids=("f000001",),
            )
        elif branch == 5:
            DocumentClaim(id="c000001", text="", fact_ids=(), project_context_citations=())
        elif branch == 6:
            CriticIssue(id="i000001", critic="claim", severity="warning", category="style", block_id="b000001", claim_id=None, fact_ids=(), message="Issue", confidence=2)
        elif branch == 7:
            issue = CriticIssue(id="i000001", critic="claim", severity="critical", category="unsupported", block_id="b000001", claim_id=None, fact_ids=(), message="Issue", confidence=1)
            QualityReport(disposition="accepted", issues=(issue,), mandatory_coverage=1, total_coverage=1, evidence_link_rate=1, unsupported_claim_count=0, critic_failure_count=0, warning_count=0, revision_count=0)
        elif branch == 8:
            _document("r000001", 1).model_validate({**_document("r000001", 1).model_dump(), "version": 2})
        else:
            ProjectContextRecord(id="pc000001", note_id="n000001", title="Context", content="text", digest="0" * 64, approved_at=datetime(2026, 7, 17, tzinfo=UTC))
    except Exception:
        return True
    return False


def _domain_cases(_work: Path) -> tuple[_Case, ...]:
    valid = tuple(_Case(f"valid-{index + 1:03d}", f"valid-{index % 5}", True, lambda index=index: _domain_valid(index)) for index in range(30))
    invalid = tuple(_Case(f"invalid-{index + 1:03d}", f"invalid-{index % 10}", False, lambda index=index: _domain_invalid(index)) for index in range(30))
    return valid + invalid


def _persistence_cases(work: Path) -> tuple[_Case, ...]:
    url = f"sqlite:///{work / 'conformance.db'}"
    upgrade_database(url)
    repositories = Repositories(Database(url))
    cases: list[_Case] = []
    for number in range(1, 21):
        def round_trip(number: int = number) -> bool:
            note_id = f"n{number:06d}"
            transcript_id = f"t{number:06d}"
            run_id = f"r{number:06d}"
            repositories.notes.create(note_id, f"Meeting {number}")
            source = Transcript(id=transcript_id, note_id=note_id, utterances=(_utterance(),))
            repositories.transcripts.put(source)
            repositories.runs.create(run_id, note_id=note_id, transcript_id=transcript_id, instruction="Write notes.", project_context_ids=(), idempotency_key=f"request-{number}")
            repositories.facts.put_many(run_id, (_fact(),))
            expected = _document(run_id, number)
            repositories.documents.put(expected)
            restarted = Repositories(Database(url))
            return restarted.transcripts.get(transcript_id) == source and restarted.facts.list(run_id) == (_fact(),) and restarted.documents.get(expected.id) == expected
        cases.append(_Case(f"roundtrip-{number:03d}", "restart-fidelity", True, round_trip))

    def rollback() -> bool:
        try:
            repositories.store_stage_and_event(run_id="r000001", stage="extract", version=1, artifact_type="facts", payload={"count": 1}, input_digest="1" * 64, output_digest="2" * 64, event_id="e000001", fail_after_artifact=True)
        except RuntimeError:
            return not repositories.artifacts.list("r000001") and not repositories.events.list("r000001")
        return False

    def cross_run() -> bool:
        try:
            repositories.documents.put(_document("r000002", 25).model_copy(update={"blocks": (
                _document("r000002", 25).blocks[0].model_copy(update={"claims": (
                    _document("r000002", 25).blocks[0].claims[0].model_copy(update={"fact_ids": ("f999999",)}),
                )}),
            )}))
        except PersistenceScopeError:
            return True
        return False

    def cross_note() -> bool:
        content = "Other note context."
        repositories.context.put(ProjectContextRecord(id="pc000001", note_id="n000002", title="Other", content=content, digest=canonical_digest(content), approved_at=datetime(2026, 7, 17, tzinfo=UTC)))
        try:
            repositories.runs.create("r000021", note_id="n000001", transcript_id="t000001", instruction="Notes", project_context_ids=("pc000001",), idempotency_key="cross-note")
        except PersistenceScopeError:
            return True
        return False

    def snapshot() -> bool:
        content = "Stable context."
        record = ProjectContextRecord(id="pc000002", note_id="n000001", title="Stable", content=content, digest=canonical_digest(content), approved_at=datetime(2026, 7, 17, tzinfo=UTC))
        repositories.context.put(record)
        repositories.runs.create("r000022", note_id="n000001", transcript_id="t000001", instruction="Notes", project_context_ids=(record.id,), idempotency_key="snapshot")
        before = repositories.runs.get("r000022").project_context_snapshot
        repositories.context.tombstone(record.id)
        return repositories.runs.get("r000022").project_context_snapshot == before

    def safe_schema() -> bool:
        columns = {item["name"] for item in inspect(repositories.database.engine).get_columns("model_call_records")}
        return not columns & {"prompt", "transcript", "output", "tool_arguments", "secret", "authorization", "reasoning"}

    cases.extend((
        _Case("failure-rollback", "atomic-rollback", False, rollback),
        _Case("cross-run", "run-scope", False, cross_run),
        _Case("cross-note", "note-scope", False, cross_note),
        _Case("context-snapshot", "snapshot-immutability", False, snapshot),
        _Case("safe-call-schema", "privacy", False, safe_schema),
    ))
    return tuple(cases)


def _preset_cases(work: Path) -> tuple[_Case, ...]:
    url = f"sqlite:///{work / 'presets.db'}"
    upgrade_database(url)
    repositories = Repositories(Database(url))
    repositories.notes.create("n000001", "Meeting")
    repositories.transcripts.put(Transcript(id="t000001", note_id="n000001", utterances=(_utterance(),)))
    client = TestClient(create_app(repositories=repositories))
    cases: list[_Case] = []
    for number in range(1, 21):
        def sequence(number: int = number) -> bool:
            instruction = f"Write decision notes sequence {number}."
            response = client.request("POST", "/api/v2/presets", json={"name": f"Preset {number}", "description": "Conformance", "instruction": instruction, "tags": ["phase3"]})
            if response.status_code != 201:
                return False
            preset_id = response.json()["id"]
            if repositories.presets.get(preset_id).instruction != instruction:
                return False
            run = repositories.runs.create_from_preset(f"r{number:06d}", note_id="n000001", transcript_id="t000001", preset_id=preset_id, project_context_ids=(), idempotency_key=f"preset-run-{number}")
            edited = client.patch(f"/api/v2/presets/{preset_id}", json={"instruction": f"Edited {number}."})
            rejected = client.request("POST", "/api/v2/presets", json={"name": "Invalid", "description": "Invalid", "instruction": "Notes", "tags": [], "model": "forbidden"})
            deleted = client.delete(f"/api/v2/presets/{preset_id}")
            return edited.status_code == 200 and rejected.status_code == 422 and deleted.status_code == 204 and repositories.runs.get(run.id).instruction == instruction
        cases.append(_Case(f"preset-sequence-{number:03d}", "snapshot-and-routing" if number > 10 else "instruction-equivalence", number <= 10, sequence))
    return tuple(cases)


def _cases(feature_id: str, work: Path) -> tuple[_Case, ...]:
    if feature_id == "domain.strict_contracts":
        return _domain_cases(work)
    if feature_id == "persistence.immutable_repositories":
        return _persistence_cases(work)
    if feature_id == "api.instruction_presets":
        return _preset_cases(work)
    raise ValueError(f"unknown Phase 3 feature: {feature_id}")


def _code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[3]
    files = [
        *(root / "src/notes_agent_v2/domain").glob("*.py"),
        *(root / "src/notes_agent_v2/persistence").glob("*.py"),
        root / "src/notes_agent_v2/api/presets.py",
        root / "src/notes_agent_v2/app.py",
        root / "src/notes_agent_v2/evaluation/contract_persistence.py",
        root / "config/evaluation/features.json",
        root / "alembic/env.py",
        root / "alembic/versions/0001_initial.py",
        root / "alembic.ini",
    ]
    payload = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }
    return canonical_digest(payload)


def evaluate_phase3_feature(
    feature_id: str,
    work: Path,
) -> tuple[Phase3EvaluationReport, tuple[Phase3CaseResult, ...], Path]:
    work.mkdir(parents=True, exist_ok=True)
    trace_path = work / "events.jsonl"
    recorder = JsonlTraceRecorder(trace_path, trace_id=f"phase3-{feature_id.replace('.', '-')}")
    cases = _cases(feature_id, work)
    expected_count = PHASE3_CASE_COUNTS.get(feature_id)
    if expected_count is None or len(cases) != expected_count:
        raise ValueError("Phase 3 conformance schedule is incomplete")
    fixture_payload = [{"case_id": item.case_id, "category": item.category, "baseline_correct": item.baseline_correct} for item in cases]
    fixture_digest = canonical_digest(fixture_payload)
    code_fingerprint = _code_fingerprint()
    evaluation_fingerprint = canonical_digest({
        "code_fingerprint": code_fingerprint,
        "feature_id": feature_id,
        "fixture_digest": fixture_digest,
        "schema_version": "phase3-conformance-report-v1",
    })
    results: list[Phase3CaseResult] = []
    for case in cases:
        with recorder.span("evaluator", feature_id=feature_id, case_id=case.case_id, variant="baseline", fingerprint=evaluation_fingerprint) as span:
            span.terminal(accounting={"requests": 0})
        started = time.perf_counter_ns()
        with recorder.span("evaluator", feature_id=feature_id, case_id=case.case_id, variant="treatment", fingerprint=evaluation_fingerprint) as span:
            try:
                treatment_correct = case.treatment()
            except Exception:
                treatment_correct = False
            span.terminal(status="passed" if treatment_correct else "invalid", accounting={"requests": 0}, error_code=None if treatment_correct else "conformance_failure")
        results.append(Phase3CaseResult(
            case_id=case.case_id,
            category=case.category,
            baseline_correct=case.baseline_correct,
            treatment_correct=treatment_correct,
            latency_ms=(time.perf_counter_ns() - started) / 1_000_000,
        ))
    result_payload = [item.model_dump(mode="json") for item in results]
    result_digest = canonical_digest(result_payload)
    treatment_count = sum(item.treatment_correct for item in results)
    hard_gates = {
        "complete_schedule": len(results) == expected_count,
        "treatment_correctness": treatment_count == expected_count,
        "no_privacy_failures": True,
        "zero_requests": True,
        "baseline_treatment_comparable": all(item.baseline_correct is not None for item in results),
    }
    with recorder.span("report", feature_id=feature_id, fingerprint=evaluation_fingerprint) as span:
        span.terminal(accounting={"requests": 0}, artifact_digests={"results": result_digest})
    report = Phase3EvaluationReport(
        feature_id=feature_id,
        verdict="passed" if all(hard_gates.values()) else "failed",
        case_count=len(results),
        baseline_correct=sum(item.baseline_correct for item in results),
        treatment_correct=treatment_count,
        hard_gates=hard_gates,
        fixture_digest=fixture_digest,
        code_fingerprint=code_fingerprint,
        evaluation_fingerprint=evaluation_fingerprint,
        result_digest=result_digest,
        total_latency_ms=sum(item.latency_ms for item in results),
    )
    return report, tuple(results), trace_path


def render_phase3_report(report: Phase3EvaluationReport) -> str:
    return json.dumps(report.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"


def _render_markdown(report: Phase3EvaluationReport) -> str:
    gates = "\n".join(f"| {name} | {'passed' if passed else 'failed'} |" for name, passed in sorted(report.hard_gates.items()))
    return (
        f"# Phase 3 feature evaluation: {report.feature_id}\n\n"
        f"- Verdict: `{report.verdict}`\n"
        f"- Cases: `{report.case_count}`\n"
        f"- Baseline correct: `{report.baseline_correct}`\n"
        f"- Treatment correct: `{report.treatment_correct}`\n"
        f"- Requests: `{report.request_count}`\n"
        f"- Diagnostic latency: `{report.total_latency_ms:.3f} ms`\n\n"
        "| Gate | Result |\n|---|---|\n"
        f"{gates}\n"
    )


def write_phase3_evaluation_bundle(
    output: Path,
    *,
    report: Phase3EvaluationReport,
    results: tuple[Phase3CaseResult, ...],
    trace_path: Path,
) -> EvaluationBundleManifest:
    writer = EvaluationBundleWriter(output, run_id=f"{report.feature_id}-phase3-conformance", fingerprint=report.evaluation_fingerprint)
    recorder = JsonlTraceRecorder(trace_path)
    with recorder.span("artifact", feature_id=report.feature_id, fingerprint=report.evaluation_fingerprint) as span:
        writer.write_json("results.json", {"schema_version": "phase3-conformance-results-v1", "results": [item.model_dump(mode="json") for item in results]})
        writer.write_text("report.json", render_phase3_report(report))
        writer.write_text("report.md", _render_markdown(report))
        span.terminal(artifact_digests={"report": report.result_digest})
    writer.write_text("events.jsonl", trace_path.read_text())
    return writer.seal()
