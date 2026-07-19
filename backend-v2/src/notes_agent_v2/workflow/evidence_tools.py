from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Protocol

from notes_agent_v2.domain.evidence import Fact, ProjectContextRecord
from notes_agent_v2.domain.transcript import Utterance
from notes_agent_v2.runtime.tools import (
    ToolAuditRecord,
    ToolAuthorizationError,
    ToolDefinition,
    ToolPolicy,
    ToolSession,
)


CLOSED_EVIDENCE_TOOLS = frozenset(
    {
        "get_fact_details",
        "get_transcript_window",
        "search_verified_facts",
        "get_generation_constraints",
        "get_claim_sources",
        "get_project_context",
    }
)


class EvidenceReader(Protocol):
    """Read-only repository boundary; implementations expose no session or path."""

    def list_facts(self, run_id: str) -> tuple[Fact, ...]: ...

    def get_utterances(self, run_id: str) -> tuple[Utterance, ...]: ...

    def get_generation_constraints(self, run_id: str) -> Mapping[str, object]: ...

    def get_claim_sources(self, run_id: str, claim_id: str) -> tuple[str, ...]: ...

    def list_project_context(self, run_id: str) -> tuple[ProjectContextRecord, ...]: ...


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _integer(
    arguments: dict[str, object], name: str, *, minimum: int, maximum: int
) -> int:
    value = arguments.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolAuthorizationError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ToolAuthorizationError(f"{name} is outside the allowed range")
    return value


def _fact_payload(fact: Fact) -> dict[str, object]:
    return {
        "id": fact.id,
        "text": fact.text,
        "kind": fact.kind,
        "status": fact.status,
        "owner": fact.owner,
        "due_text": fact.due_text,
        "verification": fact.verification,
        "evidence": [item.model_dump(mode="json") for item in fact.evidence],
        "supersedes_fact_ids": list(fact.supersedes_fact_ids),
        "conflicts_with_fact_ids": list(fact.conflicts_with_fact_ids),
    }


def build_evidence_tool_session(
    *,
    reader: EvidenceReader,
    policy: ToolPolicy,
    count_tokens: Callable[[str], int],
    audit: Callable[[ToolAuditRecord], None] = lambda _record: None,
) -> ToolSession:
    """Bind the closed evidence registry to one immutable authorization scope."""

    if not policy.allowed_tools.issubset(CLOSED_EVIDENCE_TOOLS):
        raise ToolAuthorizationError("tool policy contains an unknown evidence tool")
    run_id = policy.run_id
    allowed_ids = policy.allowed_entity_ids

    def get_fact_details(arguments: dict[str, object]) -> str:
        fact_id = arguments["fact_id"]
        facts = {item.id: item for item in reader.list_facts(run_id)}
        fact = facts.get(str(fact_id))
        if fact is None:
            raise ToolAuthorizationError("fact is unavailable in this run")
        return _json(_fact_payload(fact))

    def get_transcript_window(arguments: dict[str, object]) -> str:
        utterance_id = str(arguments["utterance_id"])
        before = _integer(arguments, "before", minimum=0, maximum=4)
        after = _integer(arguments, "after", minimum=0, maximum=4)
        utterances = reader.get_utterances(run_id)
        order = {item.id: index for index, item in enumerate(utterances)}
        if utterance_id not in order:
            raise ToolAuthorizationError("utterance is unavailable in this run")
        center = order[utterance_id]
        window = utterances[max(0, center - before) : center + after + 1]
        if any(item.id not in allowed_ids for item in window):
            raise ToolAuthorizationError("transcript window exceeds authorized scope")
        return _json(
            [
                {
                    "id": item.id,
                    "speaker_id": item.speaker_id,
                    "speaker_name": item.speaker_name,
                    "text": item.text,
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                }
                for item in window
            ]
        )

    def search_verified_facts(arguments: dict[str, object]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolAuthorizationError("search query must not be blank")
        limit = _integer(arguments, "limit", minimum=1, maximum=20)
        terms = tuple(query.casefold().split())
        matches = [
            item
            for item in reader.list_facts(run_id)
            if item.id in allowed_ids
            and item.verification == "supported"
            and all(term in item.text.casefold() for term in terms)
        ][:limit]
        return _json([_fact_payload(item) for item in matches])

    def get_generation_constraints(arguments: dict[str, object]) -> str:
        del arguments
        allowed_fields = {
            "audience",
            "depth",
            "emphasis",
            "forbidden_content",
            "required_sections",
            "instruction",
        }
        constraints = reader.get_generation_constraints(run_id)
        return _json(
            {key: value for key, value in constraints.items() if key in allowed_fields}
        )

    def get_claim_sources(arguments: dict[str, object]) -> str:
        claim_id = str(arguments["claim_id"])
        fact_ids = reader.get_claim_sources(run_id, claim_id)
        if any(item not in allowed_ids for item in fact_ids):
            raise ToolAuthorizationError("claim sources exceed authorized scope")
        return _json({"claim_id": claim_id, "fact_ids": list(fact_ids)})

    def get_project_context(arguments: dict[str, object]) -> str:
        context_id = str(arguments["context_id"])
        records = {item.id: item for item in reader.list_project_context(run_id)}
        record = records.get(context_id)
        if record is None:
            raise ToolAuthorizationError("project context is unavailable in this run")
        return _json(
            {
                "id": record.id,
                "title": record.title,
                "content": record.content,
                "digest": record.digest,
            }
        )

    definitions = {
        "get_fact_details": ToolDefinition(
            name="get_fact_details",
            allowed_arguments=frozenset({"fact_id"}),
            entity_fields=("fact_id",),
            handler=get_fact_details,
        ),
        "get_transcript_window": ToolDefinition(
            name="get_transcript_window",
            allowed_arguments=frozenset({"utterance_id", "before", "after"}),
            entity_fields=("utterance_id",),
            handler=get_transcript_window,
        ),
        "search_verified_facts": ToolDefinition(
            name="search_verified_facts",
            allowed_arguments=frozenset({"query", "limit"}),
            entity_fields=(),
            handler=search_verified_facts,
        ),
        "get_generation_constraints": ToolDefinition(
            name="get_generation_constraints",
            allowed_arguments=frozenset(),
            entity_fields=(),
            handler=get_generation_constraints,
        ),
        "get_claim_sources": ToolDefinition(
            name="get_claim_sources",
            allowed_arguments=frozenset({"claim_id"}),
            entity_fields=("claim_id",),
            handler=get_claim_sources,
        ),
        "get_project_context": ToolDefinition(
            name="get_project_context",
            allowed_arguments=frozenset({"context_id"}),
            entity_fields=("context_id",),
            handler=get_project_context,
        ),
    }
    return ToolSession(
        policy=policy,
        definitions=definitions,
        count_tokens=count_tokens,
        audit=audit,
    )
