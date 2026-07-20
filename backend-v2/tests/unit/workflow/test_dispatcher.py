import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from notes_agent_v2.runtime.budget import BudgetExceeded, RunBudget
from notes_agent_v2.runtime.gateway import GatewayError
from notes_agent_v2.workflow.dispatcher import (
    BoundedDispatcher,
    DispatchDependencies,
    DispatchError,
    RoleRequest,
    SafeMessage,
)


class Gateway:
    def __init__(self, *, error=None) -> None:
        self.calls = []
        self.error = error
        self.callback = None

    def call(self, request, *, budget, tools=None, validate=lambda value: True):
        self.calls.append((request, tools))
        if self.callback is not None:
            self.callback()
        if self.error is not None:
            raise self.error
        content = '{"ok": true}'
        assert validate(content)
        return SimpleNamespace(response=SimpleNamespace(final_content=content))


def _request(**updates):
    values = {
        "run_id": "run-1",
        "stage": "write",
        "role": "writer",
        "profile_name": "tool_reasoned",
        "messages": (
            SafeMessage(role="system", content="Use evidence."),
            SafeMessage(role="user", content="Write the note."),
        ),
        "allowed_tools": (),
        "allowed_entity_ids": (),
        "max_tool_rounds": 0,
        "max_tool_calls": 0,
        "max_tool_result_tokens": 0,
        "output_schema": {"type": "object"},
    }
    values.update(updates)
    return RoleRequest(**values)


def _dispatcher(gateway=None, records=None, budget=None):
    gateway = gateway or Gateway()
    records = records if records is not None else []
    return BoundedDispatcher(
        DispatchDependencies(
            gateway=gateway,
            budget=budget or RunBudget(max_model_requests=200),
            tool_session=SimpleNamespace(
                policy=SimpleNamespace(
                    run_id="run-1",
                    stage="write",
                    allowed_tools=frozenset({"get_fact_details"}),
                    allowed_entity_ids=frozenset(),
                    max_rounds=1,
                    max_calls=1,
                    max_result_tokens=64,
                )
            ),
            tool_schemas={"get_fact_details": {"name": "get_fact_details"}},
            record=records.append,
        )
    ), gateway, records


def test_dispatches_with_application_owned_profile_and_fresh_messages() -> None:
    dispatcher, gateway, records = _dispatcher()
    request = _request()
    result = dispatcher.dispatch(request, validate=lambda value: value.startswith("{"))
    sent, tools = gateway.calls[0]
    assert result.response.final_content == '{"ok": true}'
    assert sent.role == "writer"
    assert sent.profile_name == "tool_reasoned"
    assert sent.messages == tuple(item.model_dump() for item in request.messages)
    assert sent.messages is not request.messages
    assert tools is None
    assert records[0]["status"] == "passed"
    assert "messages" not in records[0]
    assert "content" not in json.dumps(records[0])


@pytest.mark.parametrize(
    ("role", "profile"),
    [
        ("extractor", "narrative_reasoned"),
        ("planner", "critic_structured_off"),
        ("critic", "tool_reasoned"),
        ("writer", "structured_off"),
    ],
)
def test_rejects_role_profile_mismatches(role, profile) -> None:
    dispatcher, gateway, records = _dispatcher()
    with pytest.raises(DispatchError, match="profile_not_allowed"):
        dispatcher.dispatch(_request(role=role, profile_name=profile))
    assert not gateway.calls
    assert records[-1]["error_code"] == "profile_not_allowed"


def test_rejects_unknown_fields_and_role_tool_widening() -> None:
    with pytest.raises(ValidationError):
        RoleRequest(**{**_request().model_dump(), "model": "untrusted"})
    dispatcher, gateway, _ = _dispatcher()
    with pytest.raises(DispatchError, match="tool_not_allowed"):
        dispatcher.dispatch(
            _request(
                role="planner",
                stage="capability_analysis",
                profile_name="planning_reasoned",
                allowed_tools=("get_fact_details",),
                max_tool_rounds=1,
                max_tool_calls=1,
                max_tool_result_tokens=64,
            )
        )
    assert not gateway.calls


def test_authorized_tools_are_resolved_from_closed_schemas() -> None:
    dispatcher, gateway, _ = _dispatcher()
    dispatcher.dispatch(
        _request(
            allowed_tools=("get_fact_details",),
            max_tool_rounds=1,
            max_tool_calls=1,
            max_tool_result_tokens=64,
        )
    )
    sent, tools = gateway.calls[0]
    assert sent.tools == ({"name": "get_fact_details"},)
    assert tools is not None


def test_rejects_tool_session_scope_widening() -> None:
    dispatcher, gateway, records = _dispatcher()
    with pytest.raises(DispatchError, match="tool_scope_mismatch"):
        dispatcher.dispatch(
            _request(
                allowed_tools=("get_fact_details",),
                allowed_entity_ids=("f000001",),
                max_tool_rounds=1,
                max_tool_calls=1,
                max_tool_result_tokens=64,
            )
        )
    assert not gateway.calls
    assert records[-1]["error_code"] == "tool_scope_mismatch"


def test_rejects_parallel_or_recursive_dispatch() -> None:
    dispatcher, gateway, records = _dispatcher()
    gateway.callback = lambda: dispatcher.dispatch(_request())
    with pytest.raises(DispatchError, match="parallel_dispatch"):
        dispatcher.dispatch(_request())
    assert records[-1]["error_code"] == "parallel_dispatch"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (GatewayError("provider_timeout"), "provider_timeout"),
        (BudgetExceeded("model request limit exceeded"), "budget_exceeded"),
    ],
)
def test_classifies_gateway_and_budget_failures(error, code) -> None:
    dispatcher, _, records = _dispatcher(Gateway(error=error))
    with pytest.raises(DispatchError, match=code):
        dispatcher.dispatch(_request())
    assert records[-1]["status"] == "failed"
    assert records[-1]["error_code"] == code


def test_serial_stress_has_exact_accounting() -> None:
    dispatcher, gateway, records = _dispatcher()
    for _ in range(100):
        dispatcher.dispatch(_request())
    assert len(gateway.calls) == 100
    assert len(records) == 100
    assert all(item["status"] == "passed" for item in records)


def test_dispatcher_source_has_no_execution_or_process_primitives() -> None:
    path = Path(__file__).parents[3] / "src" / "notes_agent_v2" / "workflow" / "dispatcher.py"
    tree = ast.parse(path.read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not ({"subprocess", "importlib", "multiprocessing"} & imported)
    assert not ({"eval", "exec", "compile", "__import__"} & called)
