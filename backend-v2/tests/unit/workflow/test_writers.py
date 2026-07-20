from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from notes_agent_v2.domain.evidence import EvidenceSpan, Fact
from notes_agent_v2.workflow.planner import CapabilityBlock
from notes_agent_v2.workflow.writers import (
    WriterPolicyError,
    parse_cited_narrative,
    parse_structured_block,
    write_blocks_serially,
    writer_tool_policy,
)


def _fact(
    identifier: str,
    text: str,
    *,
    kind: str = "fact",
    status: str = "asserted",
    owner: str | None = None,
    due_text: str | None = None,
    verification: str = "supported",
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


def _block(capability: str, fact_ids: tuple[str, ...]) -> CapabilityBlock:
    return CapabilityBlock(
        id="b001",
        capability=capability,
        purpose="Write the requested block",
        fact_ids=fact_ids,
        project_context_ids=(),
        constraints=(),
    )


def test_structured_writer_preserves_kind_status_owner_due_and_support() -> None:
    fact = _fact(
        "f000001",
        "Mina will ship the audit export by Friday.",
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
                        "status": "approved",
                        "owner": "Mina",
                        "due_text": "Friday",
                    }
                ],
            }
        ),
        assignment=_block("actions", (fact.id,)),
        facts=(fact,),
    )
    item = block.structured_items[0]
    assert (item.kind, item.status, item.owner, item.due_text, item.fact_ids) == (
        "action",
        "approved",
        "Mina",
        "Friday",
        (fact.id,),
    )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("status", "completed", "status_mismatch"),
        ("owner", "Omar", "owner_mismatch"),
        ("due_text", "Monday", "due_mismatch"),
        ("fact_ids", ["f999999"], "unknown_fact_reference"),
    ],
)
def test_structured_writer_rejects_inferred_or_unknown_fields(
    field: str, value: object, code: str
) -> None:
    fact = _fact(
        "f000001",
        "Mina will ship the audit export by Friday.",
        kind="action",
        status="approved",
        owner="Mina",
        due_text="Friday",
    )
    item = {
        "text": fact.text,
        "fact_ids": [fact.id],
        "status": "approved",
        "owner": "Mina",
        "due_text": "Friday",
    }
    item[field] = value
    with pytest.raises(WriterPolicyError, match=code):
        parse_structured_block(
            json.dumps({"title": "Actions", "items": [item]}),
            assignment=_block("actions", (fact.id,)),
            facts=(fact,),
        )


def test_empty_optional_structured_category_yields_no_block() -> None:
    assert (
        parse_structured_block(
            '{"title":"Risks","items":[]}',
            assignment=_block("risks", ()),
            facts=(),
        )
        is None
    )


def test_narrative_writer_parses_markers_and_requires_every_factual_line() -> None:
    fact = _fact("f000001", "The rollout begins Friday.")
    block = parse_cited_narrative(
        "## Overview\nThe rollout begins Friday. [[f000001]]",
        assignment=_block("overview", (fact.id,)),
        facts=(fact,),
        instruction="Summarize the rollout.",
    )
    assert block.claims[0].text == "The rollout begins Friday."
    assert block.claims[0].fact_ids == (fact.id,)
    with pytest.raises(WriterPolicyError, match="uncited_factual_content"):
        parse_cited_narrative(
            "## Overview\nThe rollout begins Friday.",
            assignment=_block("overview", (fact.id,)),
            facts=(fact,),
            instruction="Summarize the rollout.",
        )


def test_narrative_rejects_unassigned_uncertain_and_unsupported_entities() -> None:
    assigned = _fact("f000001", "The rollout begins Friday.")
    other = _fact("f000002", "The budget is 50 dollars.")
    uncertain = _fact(
        "f000003", "The budget may be 70 dollars.", status="uncertain", verification="uncertain"
    )
    for text, assignment_ids, facts, code in (
        ("Budget is 50 dollars. [[f000002]]", (assigned.id,), (assigned, other), "unassigned_fact_reference"),
        ("Budget may be 70 dollars. [[f000003]]", (uncertain.id,), (assigned, uncertain), "uncertain_fact_reference"),
        ("The rollout costs 99 dollars. [[f000001]]", (assigned.id,), (assigned,), "unsupported_entity"),
    ):
        with pytest.raises(WriterPolicyError, match=code):
            parse_cited_narrative(
                text,
                assignment=_block("overview", assignment_ids),
                facts=facts,
                instruction="Summarize the rollout.",
            )


def test_narrative_rejects_duplicate_claim_ids_and_output_limit() -> None:
    fact = _fact("f000001", "The rollout begins Friday.")
    with pytest.raises(WriterPolicyError, match="duplicate_claim_id"):
        parse_cited_narrative(
            "[c000001] The rollout begins Friday. [[f000001]]\n"
            "[c000001] The rollout begins Friday. [[f000001]]",
            assignment=_block("overview", (fact.id,)),
            facts=(fact,),
            instruction="Summarize the rollout.",
        )
    with pytest.raises(WriterPolicyError, match="output_limit_exceeded"):
        parse_cited_narrative(
            "The rollout begins Friday. [[f000001]]",
            assignment=_block("overview", (fact.id,)),
            facts=(fact,),
            instruction="Summarize the rollout.",
            output_tokens=101,
            output_limit=100,
        )


def test_writer_tool_policy_is_exact_and_scoped() -> None:
    policy = writer_tool_policy(
        run_id="r000001",
        allowed_entity_ids=("f000001", "pc000001"),
    )
    assert policy.allowed_tools == frozenset(
        {"get_fact_details", "get_project_context", "get_generation_constraints"}
    )
    assert (policy.max_rounds, policy.max_calls, policy.max_result_tokens) == (2, 2, 2048)
    assert policy.allowed_entity_ids == frozenset({"f000001", "pc000001"})


def test_writers_dispatch_serial_isolated_calls_with_owned_profiles() -> None:
    overview_fact = _fact("f000001", "The rollout begins Friday.")
    action_fact = _fact(
        "f000002",
        "Mina will ship the audit export by Friday.",
        kind="action",
        status="approved",
        owner="Mina",
        due_text="Friday",
    )

    class Dispatcher:
        def __init__(self) -> None:
            self.requests = []

        def dispatch(self, request, *, validate):
            self.requests.append(request)
            content = (
                "## Overview\nThe rollout begins Friday. [[f000001]]"
                if request.stage == "write_narrative"
                else json.dumps(
                    {
                        "title": "Actions",
                        "items": [
                            {
                                "text": action_fact.text,
                                "fact_ids": [action_fact.id],
                                "status": action_fact.status,
                                "owner": action_fact.owner,
                                "due_text": action_fact.due_text,
                            }
                        ],
                    }
                )
            )
            assert validate(content)
            return SimpleNamespace(
                response=SimpleNamespace(
                    final_content=content,
                    usage=SimpleNamespace(output_tokens=32),
                )
            )

    dispatcher = Dispatcher()
    assignments = (
        _block("overview", (overview_fact.id,)),
        _block("actions", (action_fact.id,)).model_copy(update={"id": "b002"}),
    )
    results = write_blocks_serially(
        run_id="r000001",
        instruction="Summarize the rollout and actions.",
        assignments=assignments,
        facts=(overview_fact, action_fact),
        project_context=(),
        dispatcher=dispatcher,
    )
    assert [request.stage for request in dispatcher.requests] == [
        "write_narrative",
        "write_structured",
    ]
    assert [request.profile_name for request in dispatcher.requests] == [
        "narrative_reasoned",
        "structured_off",
    ]
    assert all(request.allowed_tools == () for request in dispatcher.requests)
    assert all(len(request.messages) == 2 for request in dispatcher.requests)
    assert [result.status for result in results] == ["ready", "ready"]
