from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from notes_agent_v2.domain.document import DocumentBlock, DocumentClaim, NotesDocument, StructuredItem
from notes_agent_v2.domain.evidence import EvidenceSpan, Fact
from notes_agent_v2.workflow.critics import (
    critic_tool_policy,
    deterministic_critic_issues,
    run_specialist_critics,
)


def _fact(
    identifier: str,
    text: str,
    *,
    kind: str = "fact",
    status: str = "asserted",
    verification: str = "supported",
    owner: str | None = None,
    due_text: str | None = None,
) -> Fact:
    return Fact(
        id=identifier,
        text=text,
        kind=kind,
        status=status,
        speaker_ids=("s1",),
        owner=owner,
        due_text=due_text,
        confidence=1,
        verification=verification,
        evidence=(EvidenceSpan(utterance_ids=("u000001",), quote=text),),
        source_candidate_ids=("fc000001",),
        supersedes_fact_ids=(),
        conflicts_with_fact_ids=(),
    )


def _document(
    *,
    fact_id: str = "f000001",
    structured: StructuredItem | None = None,
) -> NotesDocument:
    return NotesDocument(
        id="d000001",
        run_id="r000001",
        version=1,
        parent_id=None,
        title="Notes",
        blocks=(
            DocumentBlock(
                id="b000001",
                capability="actions" if structured else "overview",
                title="Actions" if structured else "Overview",
                claims=(
                    ()
                    if structured
                    else (
                        DocumentClaim(
                            id="c000001",
                            text="The rollout begins Friday.",
                            fact_ids=(fact_id,),
                            project_context_citations=(),
                        ),
                    )
                ),
                structured_items=(structured,) if structured else (),
            ),
        ),
    )


def test_deterministic_critics_emit_concrete_missing_unknown_and_duplicate_issues() -> None:
    fact = _fact("f000001", "The rollout begins Friday.")
    missing = _fact("f000002", "The approval was recorded.", kind="decision", status="approved")
    duplicate_document = _document().model_copy(
        update={
            "blocks": (
                _document().blocks[0].model_copy(
                    update={
                        "claims": (
                            _document().blocks[0].claims[0],
                            _document().blocks[0].claims[0].model_copy(update={"id": "c000002"}),
                        )
                    }
                ),
            )
        }
    )
    issues = deterministic_critic_issues(
        duplicate_document,
        facts=(fact, missing),
        mandatory_fact_ids=(missing.id,),
        instruction="Summarize the meeting.",
    )
    assert {(issue.critic, issue.category, issue.severity) for issue in issues} == {
        ("coverage", "missing_mandatory_fact", "critical"),
        ("coverage", "duplicate_coverage", "warning"),
    }
    unknown = deterministic_critic_issues(
        _document(fact_id="f999999"),
        facts=(fact,),
        mandatory_fact_ids=(),
        instruction="Summarize the meeting.",
    )
    assert unknown[0].category == "unsupported_claim"
    assert unknown[0].claim_id == "c000001"


def test_deterministic_structured_critic_preserves_source_semantics() -> None:
    fact = _fact(
        "f000001",
        "Mina will ship the export Friday.",
        kind="action",
        status="approved",
        owner="Mina",
        due_text="Friday",
    )
    wrong = StructuredItem(
        id="si000001",
        kind="action",
        text=fact.text,
        fact_ids=(fact.id,),
        status="completed",
        owner="Omar",
        due_text="Monday",
    )
    issues = deterministic_critic_issues(
        _document(structured=wrong),
        facts=(fact,),
        mandatory_fact_ids=(),
        instruction="List actions.",
    )
    assert {issue.category for issue in issues} == {
        "wrong_status",
        "wrong_owner",
        "wrong_due",
    }
    assert all(issue.block_id == "b000001" for issue in issues)


@pytest.mark.parametrize(
    ("critic", "tools", "limits"),
    [
        ("claim", {"get_claim_sources", "get_fact_details", "get_project_context", "get_transcript_window"}, (2, 3, 3072)),
        ("coverage", {"search_verified_facts", "get_fact_details", "get_generation_constraints"}, (1, 3, 3072)),
        ("structured", {"get_claim_sources", "get_fact_details"}, (1, 2, 2048)),
        ("audience", {"get_generation_constraints"}, (1, 1, 1024)),
    ],
)
def test_specialist_tool_policies_are_closed(critic, tools, limits) -> None:
    policy = critic_tool_policy(
        critic,
        run_id="r000001",
        allowed_entity_ids=("f000001", "c000001", "u000001"),
    )
    assert policy.allowed_tools == frozenset(tools)
    assert (policy.max_rounds, policy.max_calls, policy.max_result_tokens) == limits


def test_specialists_receive_fresh_history_and_preserve_deterministic_issues() -> None:
    fact = _fact("f000001", "The rollout begins Friday.")

    class Dispatcher:
        def __init__(self, critic: str) -> None:
            self.critic = critic
            self.requests = []

        def dispatch(self, request, *, validate):
            self.requests.append(request)
            payload = {
                "issues": (
                    [
                        {
                            "id": "i000099",
                            "critic": self.critic,
                            "severity": "warning",
                            "category": "redundancy" if self.critic == "audience" else "semantic_check",
                            "block_id": "b000001",
                            "claim_id": "c000001" if self.critic == "claim" else None,
                            "fact_ids": [fact.id],
                            "message": "Concrete issue",
                            "confidence": 0.8,
                        }
                    ]
                    if self.critic in {"claim", "audience"}
                    else []
                )
            }
            content = json.dumps(payload)
            assert validate(content)
            return SimpleNamespace(response=SimpleNamespace(final_content=content))

    dispatchers = {name: Dispatcher(name) for name in ("claim", "coverage", "structured", "audience")}
    issues = run_specialist_critics(
        document=_document(fact_id="f999999"),
        facts=(fact,),
        mandatory_fact_ids=(),
        instruction="Summarize the meeting.",
        dispatchers=dispatchers,
        allowed_entity_ids=(fact.id, "c000001", "u000001"),
    )
    assert issues[0].critic == "claim"
    assert issues[0].category == "unsupported_claim"
    assert all(len(dispatcher.requests) == 1 for dispatcher in dispatchers.values())
    assert all(len(dispatcher.requests[0].messages) == 2 for dispatcher in dispatchers.values())
    assert all("Concrete issue" not in dispatcher.requests[0].messages[1].content for dispatcher in dispatchers.values())


@pytest.mark.parametrize(
    "response",
    [
        RuntimeError("timeout"),
        "not-json",
        json.dumps(
            {
                "issues": [
                    {
                        "id": "i000001",
                        "critic": "claim",
                        "severity": "critical",
                        "category": "unknown_category",
                        "block_id": "b000001",
                        "claim_id": "c000001",
                        "fact_ids": ["f000001"],
                        "message": "bad",
                        "confidence": 0.5,
                    }
                ]
            }
        ),
    ],
)
def test_any_specialist_failure_becomes_critical_system_issue(response) -> None:
    fact = _fact("f000001", "The rollout begins Friday.")

    class Dispatcher:
        def dispatch(self, request, *, validate):
            if isinstance(response, Exception):
                raise response
            return SimpleNamespace(response=SimpleNamespace(final_content=response))

    issues = run_specialist_critics(
        document=_document(),
        facts=(fact,),
        mandatory_fact_ids=(),
        instruction="Summarize the meeting.",
        dispatchers={"claim": Dispatcher()},
        allowed_entity_ids=(fact.id, "c000001", "u000001"),
    )
    failure = issues[-1]
    assert (failure.critic, failure.category, failure.severity) == (
        "system",
        "critic_failure",
        "critical",
    )


def test_missing_specialists_each_fail_closed_as_distinct_issues() -> None:
    issues = run_specialist_critics(
        document=_document(),
        facts=(_fact("f000001", "The rollout begins Friday."),),
        mandatory_fact_ids=(),
        instruction="Summarize the meeting.",
        dispatchers={},
        allowed_entity_ids=("f000001", "c000001", "u000001"),
    )
    failures = [item for item in issues if item.category == "critic_failure"]
    assert len(failures) == 4
    assert {item.message for item in failures} == {
        "The claim critic is unavailable.",
        "The coverage critic is unavailable.",
        "The structured critic is unavailable.",
        "The audience critic is unavailable.",
    }
