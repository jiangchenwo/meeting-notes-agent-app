from __future__ import annotations

import pytest
import importlib.util
from pathlib import Path

from notes_agent_v2.evaluation.development_qualification import (
    QualificationObservation,
    build_development_authorization,
    qualification_schedule,
)
from notes_agent_v2.evaluation.runtime_authorization import AuthorizationError
from notes_agent_v2.runtime.contracts import NormalizedToolCall


def passing_observations() -> list[QualificationObservation]:
    return [
        QualificationObservation(case_id=case.case_id, passed=True, provider_requests=2 if case.kind == "tool" else 1)
        for case in qualification_schedule()
    ]


def test_schedule_is_fixed_bounded_and_covers_required_call_classes() -> None:
    schedule = qualification_schedule()
    assert [sum(case.kind == kind for case in schedule) for kind in ("structured", "narrative", "tool", "critic_injected", "critic_clean")] == [4, 4, 3, 4, 2]
    assert sum(2 if case.kind == "tool" else 1 for case in schedule) == 20


def test_passing_observations_create_exact_qualified_authorization() -> None:
    authorization = build_development_authorization(
        passing_observations(),
        runtime_fingerprint="a" * 64,
        profile_fingerprint="b" * 64,
        prompt_fingerprint="c" * 64,
        schema_fingerprint="d" * 64,
        fixture_fingerprint="e" * 64,
        probe_requests=9,
    )
    assert authorization.status == "development_evaluation_qualified"
    assert authorization.evidence.total_requests == 29
    assert authorization.evidence.tool_calls == 3


def test_missing_failed_or_duplicate_observation_fails_closed() -> None:
    observations = passing_observations()
    with pytest.raises(AuthorizationError):
        build_development_authorization(
            observations[:-1],
            runtime_fingerprint="a" * 64,
            profile_fingerprint="b" * 64,
            prompt_fingerprint="c" * 64,
            schema_fingerprint="d" * 64,
            fixture_fingerprint="e" * 64,
            probe_requests=9,
        )
    observations[0] = observations[0].model_copy(update={"passed": False})
    with pytest.raises(AuthorizationError):
        build_development_authorization(
            observations,
            runtime_fingerprint="a" * 64,
            profile_fingerprint="b" * 64,
            prompt_fingerprint="c" * 64,
            schema_fingerprint="d" * 64,
            fixture_fingerprint="e" * 64,
            probe_requests=9,
        )


def test_qualification_tool_returns_the_exact_scalar_contract() -> None:
    script = Path(__file__).parents[3] / "scripts" / "qualify_development_runtime.py"
    spec = importlib.util.spec_from_file_location("qualify_development_runtime_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Tokenizer:
        def count_tokens(self, value: str) -> int:
            return len(value.split())

    session = module._tool_session("tool-01", Tokenizer())
    result = session.execute(
        NormalizedToolCall(
            call_id="call-1",
            name="lookup_qualification",
            arguments={"key": "tool-01"},
        ),
        run_id="development-runtime-qualification",
        stage="tool",
        round_number=1,
    )
    assert result.content == "VALUE-TOOL-01"
    assert module._tool_schema("tool-01")["properties"]["value"]["enum"] == [
        "VALUE-TOOL-01"
    ]
