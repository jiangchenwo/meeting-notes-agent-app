from __future__ import annotations

from datetime import UTC, datetime
import json

import pytest

from notes_agent_v2.domain.evidence import (
    EvidenceSpan,
    Fact,
    ProjectContextRecord,
    canonical_digest,
)
from notes_agent_v2.runtime.contracts import NormalizedToolCall
from notes_agent_v2.runtime.tools import ToolAuthorizationError, ToolPolicy
from notes_agent_v2.workflow.evidence_tools import build_evidence_tool_session
from notes_agent_v2.workflow.preflight import normalize_transcript


class Reader:
    def __init__(self) -> None:
        self.reads = 0
        self.utterances = normalize_transcript(
            "First fact.\nAlice owns the action.\nPrivate adjacent text.", None
        )
        self.facts = (
            Fact(
                id="f000001",
                text="First fact.",
                kind="fact",
                status="asserted",
                speaker_ids=(),
                owner=None,
                due_text=None,
                confidence=1,
                verification="supported",
                evidence=(EvidenceSpan(utterance_ids=("u000001",), quote="First fact."),),
                source_candidate_ids=("fc000001",),
                supersedes_fact_ids=(),
                conflicts_with_fact_ids=(),
            ),
            Fact(
                id="f000002",
                text="Alice owns the action.",
                kind="action",
                status="asserted",
                speaker_ids=(),
                owner="Alice",
                due_text=None,
                confidence=1,
                verification="supported",
                evidence=(EvidenceSpan(utterance_ids=("u000002",), quote="Alice owns the action."),),
                source_candidate_ids=("fc000002",),
                supersedes_fact_ids=(),
                conflicts_with_fact_ids=(),
            ),
        )
        self.context = ProjectContextRecord(
            id="pc000001",
            note_id="n000001",
            title="Glossary",
            content="API means application programming interface.",
            digest=canonical_digest("API means application programming interface."),
            approved_at=datetime(2026, 7, 17, tzinfo=UTC),
        )

    def list_facts(self, run_id):
        assert run_id == "r1"
        self.reads += 1
        return self.facts

    def get_utterances(self, run_id):
        assert run_id == "r1"
        self.reads += 1
        return self.utterances

    def get_generation_constraints(self, run_id):
        assert run_id == "r1"
        self.reads += 1
        return {"audience": "engineering", "forbidden_content": ["secrets"]}

    def get_claim_sources(self, run_id, claim_id):
        assert (run_id, claim_id) == ("r1", "cl000001")
        self.reads += 1
        return ("f000001",)

    def list_project_context(self, run_id):
        assert run_id == "r1"
        self.reads += 1
        return (self.context,)


def _policy(**updates) -> ToolPolicy:
    values = {
        "run_id": "r1",
        "stage": "write",
        "allowed_tools": frozenset(
            {
                "get_fact_details",
                "get_transcript_window",
                "search_verified_facts",
                "get_generation_constraints",
                "get_claim_sources",
                "get_project_context",
            }
        ),
        "allowed_entity_ids": frozenset(
            {"f000001", "u000001", "u000002", "cl000001", "pc000001"}
        ),
        "max_rounds": 1,
        "max_calls": 20,
        "max_result_tokens": 2_000,
    }
    values.update(updates)
    return ToolPolicy(**values)


def _call(name: str, **arguments) -> NormalizedToolCall:
    return NormalizedToolCall(call_id=f"call-{name}", name=name, arguments=arguments)


def test_closed_registry_returns_only_scoped_deterministic_safe_fields() -> None:
    reader = Reader()
    audits = []
    session = build_evidence_tool_session(
        reader=reader,
        policy=_policy(),
        count_tokens=lambda value: len(value.split()),
        audit=audits.append,
    )

    fact = json.loads(session.execute(_call("get_fact_details", fact_id="f000001"), run_id="r1", stage="write", round_number=1).content)
    window = json.loads(session.execute(_call("get_transcript_window", utterance_id="u000001", before=0, after=1), run_id="r1", stage="write", round_number=1).content)
    search = json.loads(session.execute(_call("search_verified_facts", query="first", limit=5), run_id="r1", stage="write", round_number=1).content)
    constraints = json.loads(session.execute(_call("get_generation_constraints"), run_id="r1", stage="write", round_number=1).content)
    claims = json.loads(session.execute(_call("get_claim_sources", claim_id="cl000001"), run_id="r1", stage="write", round_number=1).content)
    context = json.loads(session.execute(_call("get_project_context", context_id="pc000001"), run_id="r1", stage="write", round_number=1).content)

    assert fact["id"] == "f000001"
    assert set(fact) == {"id", "text", "kind", "status", "owner", "due_text", "verification", "evidence", "supersedes_fact_ids", "conflicts_with_fact_ids"}
    assert [item["id"] for item in window] == ["u000001", "u000002"]
    assert [item["id"] for item in search] == ["f000001"]
    assert constraints["audience"] == "engineering"
    assert claims == {"claim_id": "cl000001", "fact_ids": ["f000001"]}
    assert set(context) == {"id", "title", "content", "digest"}
    assert all(audit.run_id == "r1" and audit.stage == "write" for audit in audits)
    serialized_audits = json.dumps([item.model_dump(mode="json") for item in audits])
    assert "First fact" not in serialized_audits
    assert "arguments" not in serialized_audits


@pytest.mark.parametrize(
    "call, run_id, stage, error",
    [
        (_call("missing"), "r1", "write", "authorized"),
        (_call("get_fact_details", fact_id="f000001"), "other", "write", "run scope"),
        (_call("get_fact_details", fact_id="f000001"), "r1", "critic", "stage scope"),
        (_call("get_fact_details", fact_id="f000002"), "r1", "write", "entity"),
        (_call("get_transcript_window", utterance_id="u000003", before=0, after=0), "r1", "write", "entity"),
        (_call("get_project_context", context_id="pc999999"), "r1", "write", "entity"),
        (_call("get_fact_details", fact_id="f000001", update=True), "r1", "write", "argument"),
    ],
)
def test_registry_rejects_scope_widening_and_write_shaped_arguments(
    call: NormalizedToolCall, run_id: str, stage: str, error: str
) -> None:
    session = build_evidence_tool_session(
        reader=Reader(), policy=_policy(), count_tokens=lambda value: len(value.split())
    )
    with pytest.raises(ToolAuthorizationError, match=error):
        session.execute(call, run_id=run_id, stage=stage, round_number=1)


def test_window_cannot_include_an_unauthorized_adjacent_utterance() -> None:
    session = build_evidence_tool_session(
        reader=Reader(), policy=_policy(), count_tokens=lambda value: len(value.split())
    )
    with pytest.raises(ToolAuthorizationError, match="window"):
        session.execute(
            _call("get_transcript_window", utterance_id="u000002", before=0, after=1),
            run_id="r1",
            stage="write",
            round_number=1,
        )
    assert session.calls == 1


def test_denied_authorization_attempt_is_counted_and_audited() -> None:
    audits = []
    session = build_evidence_tool_session(
        reader=Reader(),
        policy=_policy(),
        count_tokens=lambda value: len(value.split()),
        audit=audits.append,
    )
    with pytest.raises(ToolAuthorizationError, match="entity"):
        session.execute(
            _call("get_fact_details", fact_id="f999999"),
            run_id="r1",
            stage="write",
            round_number=1,
        )
    assert session.calls == 1
    assert len(audits) == 1
    assert audits[0].status == "rejected"


def test_cache_hits_count_and_audit_without_repeating_repository_read() -> None:
    reader = Reader()
    audits = []
    session = build_evidence_tool_session(
        reader=reader,
        policy=_policy(),
        count_tokens=lambda value: len(value.split()),
        audit=audits.append,
    )
    call = _call("get_fact_details", fact_id="f000001")
    first = session.execute(call, run_id="r1", stage="write", round_number=1)
    second = session.execute(call, run_id="r1", stage="write", round_number=1)

    assert first.content == second.content
    assert session.calls == 2
    assert reader.reads == 1
    assert [item.cache_hit for item in audits] == [False, True]


def test_result_token_call_and_round_limits_fail_closed() -> None:
    oversized = build_evidence_tool_session(
        reader=Reader(),
        policy=_policy(max_result_tokens=1),
        count_tokens=lambda value: len(value.split()),
    )
    with pytest.raises(ToolAuthorizationError, match="result token"):
        oversized.execute(_call("get_fact_details", fact_id="f000001"), run_id="r1", stage="write", round_number=1)

    limited = build_evidence_tool_session(
        reader=Reader(), policy=_policy(max_calls=1), count_tokens=lambda value: 1
    )
    limited.execute(_call("get_fact_details", fact_id="f000001"), run_id="r1", stage="write", round_number=1)
    with pytest.raises(ToolAuthorizationError, match="call limit"):
        limited.execute(_call("get_fact_details", fact_id="f000001"), run_id="r1", stage="write", round_number=1)
    with pytest.raises(ToolAuthorizationError, match="round"):
        build_evidence_tool_session(reader=Reader(), policy=_policy(), count_tokens=lambda value: 1).execute(
            _call("get_fact_details", fact_id="f000001"), run_id="r1", stage="write", round_number=2
        )
