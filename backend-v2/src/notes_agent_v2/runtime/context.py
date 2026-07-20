from __future__ import annotations

from math import ceil
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextBudgetExceeded(RuntimeError):
    pass


class TokenizerModelMismatch(RuntimeError):
    pass


class PromptTokenizer(Protocol):
    model_key: str
    instance_id: str
    exact: bool

    def render_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> str: ...

    def count_tokens(self, rendered_prompt: str) -> int: ...


def get_loaded_lm_studio_model(client: Any) -> Any:
    """Return LM Studio's current model without selecting or loading one."""

    return client.llm.model()


class ContextEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hard_context: int = Field(default=40960, gt=0)
    initial_prompt_limit: int = Field(default=24576, ge=0)
    generation_limit: int = Field(default=8192, ge=0)
    tool_result_limit: int = Field(default=4096, ge=0)
    safety_margin: int = Field(default=4096, ge=0)

    @property
    def total_reserved(self) -> int:
        return (
            self.initial_prompt_limit
            + self.generation_limit
            + self.tool_result_limit
            + self.safety_margin
        )

    @model_validator(mode="after")
    def validate_exact_partition(self) -> ContextEnvelope:
        if self.total_reserved != self.hard_context:
            raise ValueError("context partitions must be exactly equal to hard_context")
        return self


class ContextLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    envelope: ContextEnvelope = Field(default_factory=ContextEnvelope)
    initial_prompt_tokens: int = Field(default=0, ge=0)
    generation_tokens: int = Field(default=0, ge=0)
    reserved_tool_capacity: int = Field(default=0, ge=0)
    safety_tokens: int = Field(default=0, ge=0)
    tool_result_tokens: int = Field(default=0, ge=0)
    actual_prompt_tokens: int = Field(default=0, ge=0)
    actual_output_tokens: int = Field(default=0, ge=0)
    rejected: bool = False
    reservations: list[str] = Field(default_factory=list)

    def reserve_call(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        output_schema: dict[str, Any] | None,
        tokenizer: PromptTokenizer,
        model_key: str,
        instance_id: str,
    ) -> None:
        self.ensure_not_rejected()
        if (
            not tokenizer.exact
            or tokenizer.model_key != model_key
            or tokenizer.instance_id != instance_id
        ):
            raise TokenizerModelMismatch("exact tokenizer is not bound to the resolved instance")
        rendered = tokenizer.render_chat(messages, tools, output_schema)
        token_count = tokenizer.count_tokens(rendered)
        if token_count > self.envelope.initial_prompt_limit:
            self._reject("initial prompt exceeds context envelope")
        self.initial_prompt_tokens = token_count
        self.generation_tokens = self.envelope.generation_limit
        self.reserved_tool_capacity = self.envelope.tool_result_limit
        self.safety_tokens = self.envelope.safety_margin
        self.reservations.extend(
            [
                f"initial:{token_count}",
                f"generation:{self.generation_tokens}",
                f"tool_capacity:{self.reserved_tool_capacity}",
                f"safety:{self.safety_tokens}",
            ]
        )

    def reserve_tool_result(self, token_count: int) -> None:
        self.ensure_not_rejected()
        if token_count < 0 or self.tool_result_tokens + token_count > self.reserved_tool_capacity:
            self._reject("tool result tokens exceed context envelope")
        self.tool_result_tokens += token_count
        self.reservations.append(f"tool_result:{token_count}")

    def reconcile(self, *, prompt_tokens: int, output_tokens: int) -> None:
        self.ensure_not_rejected()
        if prompt_tokens > self.envelope.initial_prompt_limit:
            self._reject("actual prompt exceeds context envelope")
        if output_tokens > self.envelope.generation_limit:
            self._reject("actual output exceeds context envelope")
        self.actual_prompt_tokens = prompt_tokens
        self.actual_output_tokens = output_tokens

    def ensure_not_rejected(self) -> None:
        if self.rejected:
            raise ContextBudgetExceeded("context ledger was previously rejected")

    def _reject(self, message: str) -> None:
        self.rejected = True
        raise ContextBudgetExceeded(message)


class LMStudioPromptTokenizer:
    exact = True

    def __init__(self, client: Any, model: Any, *, model_key: str, instance_id: str) -> None:
        self.client = client
        self.model = model
        self.model_key = model_key
        self.instance_id = instance_id

    @classmethod
    def from_client(
        cls, client: Any, *, model_key: str, instance_id: str
    ) -> LMStudioPromptTokenizer:
        try:
            model = client.get_model(model_key, instance_id)
        except Exception as exc:
            raise TokenizerModelMismatch("exact LM Studio tokenizer model is unavailable") from exc
        found_key = getattr(model, "model_key", getattr(model, "key", None))
        found_instance = getattr(model, "instance_id", getattr(model, "id", None))
        if found_key != model_key or found_instance != instance_id:
            raise TokenizerModelMismatch("tokenizer is not bound to the resolved model instance")
        return cls(client, model, model_key=model_key, instance_id=instance_id)

    def render_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        return self.model.apply_prompt_template(
            messages, tools=tools, output_schema=output_schema
        )

    def count_tokens(self, rendered_prompt: str) -> int:
        return len(self.model.tokenize(rendered_prompt))

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


class LMStudioSDKPromptTokenizer:
    exact = True

    def __init__(
        self,
        model: Any,
        *,
        model_key: str,
        instance_id: str,
        loaded_context: int,
    ) -> None:
        try:
            info = model.get_info()
        except Exception as exc:
            raise TokenizerModelMismatch("loaded tokenizer model metadata is unavailable") from exc
        if (
            getattr(info, "model_key", None) != model_key
            or getattr(info, "context_length", None) != loaded_context
        ):
            raise TokenizerModelMismatch("tokenizer model identity or context drifted")
        self.model = model
        self.model_key = model_key
        self.instance_id = instance_id

    def render_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        options: dict[str, Any] = {}
        if tools:
            options["toolDefinitions"] = tools
        return self.model.apply_prompt_template({"messages": messages}, options)

    def count_tokens(self, rendered_prompt: str) -> int:
        return len(self.model.tokenize(rendered_prompt))


class TokenEstimate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tokens: int = Field(ge=0)
    exact: bool


class ConservativeTokenEstimator(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chars_per_token: int = Field(default=4, gt=0)
    safety_multiplier: float = Field(default=1.5, gt=0)

    def estimate(self, text: str) -> TokenEstimate:
        return TokenEstimate(
            tokens=ceil(len(text) / self.chars_per_token * self.safety_multiplier),
            exact=False,
        )


def prohibit_estimate_readiness_certification(estimate: TokenEstimate) -> None:
    if not estimate.exact:
        raise ContextBudgetExceeded("an exact tokenizer is required to certify readiness")
