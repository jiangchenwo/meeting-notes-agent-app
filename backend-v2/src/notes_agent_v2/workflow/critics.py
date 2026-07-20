from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from notes_agent_v2.domain.document import NotesDocument
from notes_agent_v2.domain.evidence import Fact
from notes_agent_v2.domain.quality import CriticIssue
from notes_agent_v2.runtime.tools import ToolPolicy
from notes_agent_v2.workflow.dispatcher import RoleRequest, SafeMessage


Specialist = Literal["claim", "coverage", "structured", "audience"]

_POLICIES = {
    "claim": (
        frozenset(
            {
                "get_claim_sources",
                "get_fact_details",
                "get_project_context",
                "get_transcript_window",
            }
        ),
        2,
        3,
        3072,
    ),
    "coverage": (
        frozenset(
            {"search_verified_facts", "get_fact_details", "get_generation_constraints"}
        ),
        1,
        3,
        3072,
    ),
    "structured": (
        frozenset({"get_claim_sources", "get_fact_details"}),
        1,
        2,
        2048,
    ),
    "audience": (frozenset({"get_generation_constraints"}), 1, 1, 1024),
}
_CATEGORIES = {
    "claim": frozenset(
        {
            "semantic_check",
            "unsupported_detail",
            "contradiction",
            "wrong_speaker",
            "wrong_number",
            "wrong_date",
            "wrong_status",
            "ignored_correction",
        }
    ),
    "coverage": frozenset(
        {"semantic_check", "missing_mandatory_fact", "duplicate_coverage"}
    ),
    "structured": frozenset(
        {
            "semantic_check",
            "wrong_owner",
            "wrong_due",
            "wrong_status",
            "proposal_as_decision",
            "duplicate_item",
        }
    ),
    "audience": frozenset(
        {
            "redundancy",
            "instruction_violation",
            "depth_violation",
            "format_violation",
            "order_failure",
        }
    ),
}


class CriticDispatcher(Protocol):
    def dispatch(self, request: RoleRequest, *, validate): ...


class _CriticOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    issues: tuple[CriticIssue, ...]


def critic_tool_policy(
    critic: Specialist,
    *,
    run_id: str,
    allowed_entity_ids: Sequence[str],
) -> ToolPolicy:
    tools, rounds, calls, tokens = _POLICIES[critic]
    return ToolPolicy(
        run_id=run_id,
        stage="critic",
        allowed_tools=tools,
        allowed_entity_ids=frozenset(allowed_entity_ids),
        max_rounds=rounds,
        max_calls=calls,
        max_result_tokens=tokens,
    )


def deterministic_critic_issues(
    document: NotesDocument,
    *,
    facts: Sequence[Fact],
    mandatory_fact_ids: Sequence[str],
    instruction: str,
) -> tuple[CriticIssue, ...]:
    del instruction
    fact_by_id = {item.id: item for item in facts}
    issues: list[CriticIssue] = []
    covered: list[str] = []
    for block in document.blocks:
        for claim in block.claims:
            covered.extend(claim.fact_ids)
            if any(
                identifier not in fact_by_id
                or fact_by_id[identifier].verification != "supported"
                for identifier in claim.fact_ids
            ):
                issues.append(
                    _issue(
                        critic="claim",
                        severity="critical",
                        category="unsupported_claim",
                        block_id=block.id,
                        claim_id=claim.id,
                        fact_ids=claim.fact_ids,
                        message="Claim support is unknown or unverified.",
                    )
                )
        for item in block.structured_items:
            covered.extend(item.fact_ids)
            sources = [fact_by_id.get(identifier) for identifier in item.fact_ids]
            if any(source is None or source.verification != "supported" for source in sources):
                issues.append(
                    _issue(
                        critic="structured",
                        severity="critical",
                        category="unsupported_claim",
                        block_id=block.id,
                        claim_id=None,
                        fact_ids=item.fact_ids,
                        message="Structured item support is unknown or unverified.",
                    )
                )
                continue
            typed_sources = [source for source in sources if source is not None]
            checks = (
                (item.status not in {source.status for source in typed_sources}, "wrong_status"),
                (item.owner not in {source.owner for source in typed_sources}, "wrong_owner"),
                (item.due_text not in {source.due_text for source in typed_sources}, "wrong_due"),
            )
            for failed, category in checks:
                if failed:
                    issues.append(
                        _issue(
                            critic="structured",
                            severity="critical",
                            category=category,
                            block_id=block.id,
                            claim_id=None,
                            fact_ids=item.fact_ids,
                            message=f"Structured item has {category.replace('_', ' ')}.",
                        )
                    )
    covered_set = set(covered)
    for identifier in mandatory_fact_ids:
        if identifier not in covered_set:
            issues.append(
                _issue(
                    critic="coverage",
                    severity="critical",
                    category="missing_mandatory_fact",
                    block_id=None,
                    claim_id=None,
                    fact_ids=(identifier,),
                    message="A mandatory fact is absent from the document.",
                )
            )
    for identifier, count in Counter(covered).items():
        if count > 1:
            issues.append(
                _issue(
                    critic="coverage",
                    severity="warning",
                    category="duplicate_coverage",
                    block_id=None,
                    claim_id=None,
                    fact_ids=(identifier,),
                    message="A fact is covered more than once.",
                )
            )
    return _renumber(_deduplicate(issues))


def run_specialist_critics(
    *,
    document: NotesDocument,
    facts: Sequence[Fact],
    mandatory_fact_ids: Sequence[str],
    instruction: str,
    dispatchers: Mapping[Specialist, CriticDispatcher],
    allowed_entity_ids: Sequence[str],
) -> tuple[CriticIssue, ...]:
    issues = list(
        deterministic_critic_issues(
            document,
            facts=facts,
            mandatory_fact_ids=mandatory_fact_ids,
            instruction=instruction,
        )
    )
    block_ids = {item.id for item in document.blocks}
    claim_ids = {claim.id for block in document.blocks for claim in block.claims}
    fact_ids = {item.id for item in facts}
    authoritative = json.dumps(
        {
            "document": document.model_dump(mode="json"),
            "facts": [item.model_dump(mode="json") for item in facts],
            "mandatory_fact_ids": list(mandatory_fact_ids),
            "instruction": instruction,
        },
        sort_keys=True,
    )
    for critic in ("claim", "coverage", "structured", "audience"):
        dispatcher = dispatchers.get(critic)  # type: ignore[arg-type]
        if dispatcher is None:
            issues.append(
                _issue(
                    critic="system",
                    severity="critical",
                    category="critic_failure",
                    block_id=None,
                    claim_id=None,
                    fact_ids=(),
                    message=f"The {critic} critic is unavailable.",
                )
            )
            continue
        policy = critic_tool_policy(
            critic,  # type: ignore[arg-type]
            run_id=document.run_id,
            allowed_entity_ids=allowed_entity_ids,
        )

        def validate(content: str, specialist: str = critic) -> bool:
            try:
                parsed = _CriticOutput.model_validate_json(content)
                _validate_model_issues(
                    parsed.issues,
                    specialist=specialist,
                    block_ids=block_ids,
                    claim_ids=claim_ids,
                    fact_ids=fact_ids,
                )
            except Exception:
                return False
            return True

        request = RoleRequest(
            run_id=document.run_id,
            stage="critic",
            role="critic",
            profile_name="critic_structured_off",
            messages=(
                SafeMessage(
                    role="system",
                    content=(
                        f"Act only as the {critic} error detector. Return concrete issue "
                        "instances in the required schema. Do not score or accept the draft."
                    ),
                ),
                SafeMessage(role="user", content=authoritative),
            ),
            allowed_tools=tuple(sorted(policy.allowed_tools)),
            allowed_entity_ids=tuple(sorted(policy.allowed_entity_ids)),
            max_tool_rounds=policy.max_rounds,
            max_tool_calls=policy.max_calls,
            max_tool_result_tokens=policy.max_result_tokens,
            output_schema=_CriticOutput.model_json_schema(),
        )
        try:
            result = dispatcher.dispatch(request, validate=validate)
            parsed = _CriticOutput.model_validate_json(result.response.final_content)
            _validate_model_issues(
                parsed.issues,
                specialist=critic,
                block_ids=block_ids,
                claim_ids=claim_ids,
                fact_ids=fact_ids,
            )
            issues.extend(parsed.issues)
        except Exception:
            issues.append(
                _issue(
                    critic="system",
                    severity="critical",
                    category="critic_failure",
                    block_id=None,
                    claim_id=None,
                    fact_ids=(),
                    message=f"The {critic} critic failed.",
                )
            )
    return _renumber(_deduplicate(issues))


def _validate_model_issues(
    issues: Sequence[CriticIssue],
    *,
    specialist: str,
    block_ids: set[str],
    claim_ids: set[str],
    fact_ids: set[str],
) -> None:
    for issue in issues:
        if issue.critic != specialist:
            raise ValueError("critic identity mismatch")
        if issue.category not in _CATEGORIES[specialist]:
            raise ValueError("unknown critic category")
        if issue.confidence is None:
            raise ValueError("model critic confidence is required")
        if issue.block_id is not None and issue.block_id not in block_ids:
            raise ValueError("critic block target is unknown")
        if issue.claim_id is not None and issue.claim_id not in claim_ids:
            raise ValueError("critic claim target is unknown")
        if any(identifier not in fact_ids for identifier in issue.fact_ids):
            raise ValueError("critic fact target is unknown")


def _issue(
    *,
    critic: str,
    severity: str,
    category: str,
    block_id: str | None,
    claim_id: str | None,
    fact_ids: Sequence[str],
    message: str,
) -> CriticIssue:
    return CriticIssue(
        id="i000001",
        critic=critic,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        category=category,
        block_id=block_id,
        claim_id=claim_id,
        fact_ids=tuple(fact_ids),
        message=message,
        confidence=None,
    )


def _deduplicate(issues: Sequence[CriticIssue]) -> tuple[CriticIssue, ...]:
    unique: dict[tuple[object, ...], CriticIssue] = {}
    for issue in issues:
        key = (
            issue.critic,
            issue.severity,
            issue.category,
            issue.block_id,
            issue.claim_id,
            issue.fact_ids,
            issue.message,
        )
        unique.setdefault(key, issue)
    return tuple(unique.values())


def _renumber(issues: Sequence[CriticIssue]) -> tuple[CriticIssue, ...]:
    return tuple(
        issue.model_copy(update={"id": f"i{index:06d}"})
        for index, issue in enumerate(issues, start=1)
    )
