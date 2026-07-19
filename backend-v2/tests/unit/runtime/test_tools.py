from __future__ import annotations

import pytest

from notes_agent_v2.runtime.contracts import NormalizedToolCall
from notes_agent_v2.runtime.tools import (
    ToolAuthorizationError,
    ToolDefinition,
    ToolPolicy,
    ToolSession,
)


def call(arguments: dict[str, object] | None = None, *, name: str = "lookup") -> NormalizedToolCall:
    return NormalizedToolCall(
        call_id="call-1", name=name, arguments=arguments or {"fact_id": "f1"}
    )


def session(counter: list[int], **policy_overrides: object) -> ToolSession:
    values = {
        "run_id": "run-1",
        "stage": "writer",
        "allowed_tools": frozenset({"lookup"}),
        "allowed_entity_ids": frozenset({"f1"}),
        "max_rounds": 1,
        "max_calls": 1,
        "max_result_tokens": 4,
    }
    values.update(policy_overrides)
    policy = ToolPolicy(**values)

    def handler(arguments: dict[str, object]) -> str:
        counter.append(1)
        return "safe result"

    return ToolSession(
        policy=policy,
        definitions={
            "lookup": ToolDefinition(
                name="lookup",
                allowed_arguments=frozenset({"fact_id"}),
                entity_fields=("fact_id",),
                handler=handler,
            )
        },
        count_tokens=lambda text: len(text.split()),
    )


def test_authorized_tool_executes_and_returns_no_arguments() -> None:
    executions: list[int] = []
    result = session(executions).execute(
        call(), run_id="run-1", stage="writer", round_number=1
    )
    assert executions == [1]
    assert result.model_dump() == {
        "call_id": "call-1",
        "name": "lookup",
        "content": "safe result",
        "result_tokens": 2,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_id": "other"},
        {"stage": "critic"},
        {"round_number": 2},
    ],
)
def test_wrong_scope_never_executes(kwargs: dict[str, object]) -> None:
    executions: list[int] = []
    values = {"run_id": "run-1", "stage": "writer", "round_number": 1}
    values.update(kwargs)
    with pytest.raises(ToolAuthorizationError):
        session(executions).execute(call(), **values)
    assert executions == []


def test_unknown_tool_and_entity_never_execute() -> None:
    executions: list[int] = []
    tool_session = session(executions)
    with pytest.raises(ToolAuthorizationError, match="tool"):
        tool_session.execute(call(name="delete"), run_id="run-1", stage="writer", round_number=1)
    tool_session = session(executions)
    with pytest.raises(ToolAuthorizationError, match="entity"):
        tool_session.execute(
            call({"fact_id": "f2"}), run_id="run-1", stage="writer", round_number=1
        )
    assert executions == []


def test_unexpected_tool_arguments_never_execute() -> None:
    executions: list[int] = []
    with pytest.raises(ToolAuthorizationError, match="argument"):
        session(executions).execute(
            call({"fact_id": "f1", "unrestricted": "value"}),
            run_id="run-1",
            stage="writer",
            round_number=1,
        )
    assert executions == []


def test_call_and_result_token_limits_fail_closed() -> None:
    executions: list[int] = []
    tool_session = session(executions, max_calls=1)
    tool_session.execute(call(), run_id="run-1", stage="writer", round_number=1)
    with pytest.raises(ToolAuthorizationError, match="call limit"):
        tool_session.execute(call(), run_id="run-1", stage="writer", round_number=1)

    executions = []
    with pytest.raises(ToolAuthorizationError, match="result token"):
        session(executions, max_result_tokens=1).execute(
            call(), run_id="run-1", stage="writer", round_number=1
        )
    assert executions == [1]
