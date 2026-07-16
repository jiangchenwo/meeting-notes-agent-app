from __future__ import annotations

import pytest
from pydantic import ValidationError

from notes_agent_v2.runtime.budget import (
    BudgetExceeded,
    RunBudget,
    RunCancelled,
)


def test_default_run_budget_is_exact_and_serial() -> None:
    budget = RunBudget()
    assert budget.max_model_requests == 64
    assert budget.max_tool_calls == 64
    assert budget.max_input_tokens == 600_000
    assert budget.max_output_tokens == 80_000
    assert budget.max_wall_seconds == 3600
    assert budget.max_revisions == 2
    assert budget.max_parallel_model_requests == 1


def test_parallel_limit_cannot_be_relaxed() -> None:
    with pytest.raises(ValidationError, match="max_parallel_model_requests"):
        RunBudget(max_parallel_model_requests=2)


def test_request_65_fails_before_provider_invocation() -> None:
    budget = RunBudget()
    for _ in range(64):
        budget.reserve_model_request(input_tokens=1, output_tokens=1, elapsed_seconds=0)
        budget.reconcile_model_request(input_tokens=1, output_tokens=1)
    with pytest.raises(BudgetExceeded, match="model request"):
        budget.reserve_model_request(input_tokens=1, output_tokens=1, elapsed_seconds=0)
    assert budget.model_requests == 64


def test_active_request_blocks_parallel_reservation() -> None:
    budget = RunBudget()
    budget.reserve_model_request(input_tokens=1, output_tokens=1, elapsed_seconds=0)
    with pytest.raises(BudgetExceeded, match="parallel"):
        budget.reserve_model_request(input_tokens=1, output_tokens=1, elapsed_seconds=0)


def test_cancellation_has_precedence_over_exhaustion() -> None:
    budget = RunBudget(max_model_requests=1)
    budget.reserve_model_request(input_tokens=1, output_tokens=1, elapsed_seconds=0)
    budget.reconcile_model_request(input_tokens=1, output_tokens=1)
    budget.cancel()
    with pytest.raises(RunCancelled):
        budget.reserve_model_request(input_tokens=1, output_tokens=1, elapsed_seconds=0)


def test_token_tool_revision_and_wall_limits_fail_closed() -> None:
    with pytest.raises(BudgetExceeded, match="input token"):
        RunBudget(max_input_tokens=2).reserve_model_request(
            input_tokens=3, output_tokens=1, elapsed_seconds=0
        )
    with pytest.raises(BudgetExceeded, match="wall"):
        RunBudget().reserve_model_request(input_tokens=1, output_tokens=1, elapsed_seconds=3601)

    tools = RunBudget(max_tool_calls=1)
    tools.reserve_tool_call()
    with pytest.raises(BudgetExceeded, match="tool call"):
        tools.reserve_tool_call()

    revisions = RunBudget(max_revisions=1)
    revisions.reserve_revision()
    with pytest.raises(BudgetExceeded, match="revision"):
        revisions.reserve_revision()


def test_run_counters_survive_serialization_and_reload() -> None:
    budget = RunBudget()
    budget.reserve_model_request(input_tokens=10, output_tokens=5, elapsed_seconds=2)
    budget.reconcile_model_request(input_tokens=8, output_tokens=4)
    budget.reserve_tool_call()
    restored = RunBudget.model_validate_json(budget.model_dump_json())

    assert restored.model_requests == 1
    assert restored.input_tokens == 8
    assert restored.output_tokens == 4
    assert restored.tool_calls == 1
    assert restored.active_model_requests == 0
