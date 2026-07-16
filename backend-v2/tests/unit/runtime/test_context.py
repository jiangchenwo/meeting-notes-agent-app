from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from notes_agent_v2.runtime.context import (
    ConservativeTokenEstimator,
    ContextBudgetExceeded,
    ContextEnvelope,
    ContextLedger,
    LMStudioPromptTokenizer,
    TokenEstimate,
    TokenizerModelMismatch,
    prohibit_estimate_readiness_certification,
)


class ExactTokenizer:
    model_key = "google/gemma-4-26b-a4b-qat"
    instance_id = "loaded-1"
    exact = True

    def render_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        return "|".join(str(message["content"]) for message in messages)

    def count_tokens(self, rendered_prompt: str) -> int:
        return len(rendered_prompt.split())


def test_default_context_envelope_is_exact() -> None:
    envelope = ContextEnvelope()
    assert envelope.total_reserved == 40960
    assert envelope.total_reserved == envelope.hard_context


def test_context_envelope_must_partition_the_exact_hard_context() -> None:
    with pytest.raises(ValidationError, match="exactly equal"):
        ContextEnvelope(safety_margin=4095)


def test_rendered_prompt_and_all_partitions_are_reserved_before_invocation() -> None:
    ledger = ContextLedger()
    ledger.reserve_call(
        messages=[{"role": "user", "content": "one two three"}],
        tools=None,
        output_schema=None,
        tokenizer=ExactTokenizer(),
        model_key="google/gemma-4-26b-a4b-qat",
        instance_id="loaded-1",
    )
    assert ledger.initial_prompt_tokens == 3
    assert ledger.generation_tokens == 8192
    assert ledger.reserved_tool_capacity == 4096
    assert ledger.safety_tokens == 4096
    assert ledger.reservations == ["initial:3", "generation:8192", "tool_capacity:4096", "safety:4096"]


def test_wrong_or_inexact_tokenizer_cannot_certify_call() -> None:
    wrong = ExactTokenizer()
    wrong.instance_id = "other"
    with pytest.raises(TokenizerModelMismatch):
        ContextLedger().reserve_call(
            messages=[],
            tools=None,
            output_schema=None,
            tokenizer=wrong,
            model_key="google/gemma-4-26b-a4b-qat",
            instance_id="loaded-1",
        )


def test_repeated_tool_results_and_overflow_fail_closed() -> None:
    ledger = ContextLedger()
    ledger.reserve_call(
        messages=[],
        tools=None,
        output_schema=None,
        tokenizer=ExactTokenizer(),
        model_key=ExactTokenizer.model_key,
        instance_id=ExactTokenizer.instance_id,
    )
    ledger.reserve_tool_result(2048)
    ledger.reserve_tool_result(2048)
    assert ledger.tool_result_tokens == 4096
    with pytest.raises(ContextBudgetExceeded):
        ledger.reserve_tool_result(1)
    with pytest.raises(ContextBudgetExceeded, match="previously rejected"):
        ledger.reserve_tool_result(0)


def test_reconciliation_rejects_actual_partition_overflow() -> None:
    ledger = ContextLedger()
    ledger.reserve_call(
        messages=[],
        tools=None,
        output_schema=None,
        tokenizer=ExactTokenizer(),
        model_key=ExactTokenizer.model_key,
        instance_id=ExactTokenizer.instance_id,
    )
    with pytest.raises(ContextBudgetExceeded, match="actual output"):
        ledger.reconcile(prompt_tokens=1, output_tokens=8193)


def test_conservative_estimator_may_reject_but_never_certify() -> None:
    estimate = ConservativeTokenEstimator().estimate("abcd" * 10)
    assert estimate.exact is False
    with pytest.raises(ContextBudgetExceeded, match="exact tokenizer"):
        prohibit_estimate_readiness_certification(estimate)
    prohibit_estimate_readiness_certification(TokenEstimate(tokens=10, exact=True))


def test_lm_studio_tokenizer_binds_exact_model_and_instance() -> None:
    class Model:
        model_key = "google/gemma-4-26b-a4b-qat"
        instance_id = "loaded-1"

        def apply_prompt_template(self, messages, tools=None, output_schema=None):
            return "rendered"

        def tokenize(self, value: str):
            return [1, 2, 3]

    class Client:
        def get_model(self, model_key: str, instance_id: str):
            assert model_key == Model.model_key
            assert instance_id == Model.instance_id
            return Model()

    tokenizer = LMStudioPromptTokenizer.from_client(
        Client(), model_key=Model.model_key, instance_id=Model.instance_id
    )
    assert tokenizer.exact is True
    assert tokenizer.render_chat([]) == "rendered"
    assert tokenizer.count_tokens("rendered") == 3

    with pytest.raises(TokenizerModelMismatch):
        LMStudioPromptTokenizer.from_client(
            Client(), model_key=Model.model_key, instance_id="other"
        )
