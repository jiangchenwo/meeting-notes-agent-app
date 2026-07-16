from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .contracts import NormalizedToolCall


class ToolAuthorizationError(RuntimeError):
    pass


class ToolPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    allowed_tools: frozenset[str]
    allowed_entity_ids: frozenset[str]
    max_rounds: int = Field(ge=0)
    max_calls: int = Field(ge=0)
    max_result_tokens: int = Field(ge=0)


class ToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    allowed_arguments: frozenset[str]
    entity_fields: tuple[str, ...]
    handler: Callable[[dict[str, object]], str]


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    name: str
    content: str
    result_tokens: int = Field(ge=0)


class ToolSession:
    def __init__(
        self,
        *,
        policy: ToolPolicy,
        definitions: dict[str, ToolDefinition],
        count_tokens: Callable[[str], int],
    ) -> None:
        self.policy = policy
        self.definitions = definitions
        self.count_tokens = count_tokens
        self.calls = 0
        self.result_tokens = 0

    def execute(
        self,
        call: NormalizedToolCall,
        *,
        run_id: str,
        stage: str,
        round_number: int,
    ) -> ToolExecutionResult:
        definition = self._authorize(call, run_id=run_id, stage=stage, round_number=round_number)
        self.calls += 1
        content = definition.handler(dict(call.arguments))
        tokens = self.count_tokens(content)
        if self.result_tokens + tokens > self.policy.max_result_tokens:
            raise ToolAuthorizationError("tool result token limit exceeded")
        self.result_tokens += tokens
        return ToolExecutionResult(
            call_id=call.call_id,
            name=call.name,
            content=content,
            result_tokens=tokens,
        )

    def _authorize(
        self,
        call: NormalizedToolCall,
        *,
        run_id: str,
        stage: str,
        round_number: int,
    ) -> ToolDefinition:
        if run_id != self.policy.run_id:
            raise ToolAuthorizationError("tool run scope mismatch")
        if stage != self.policy.stage:
            raise ToolAuthorizationError("tool stage scope mismatch")
        if round_number < 1 or round_number > self.policy.max_rounds:
            raise ToolAuthorizationError("tool round limit exceeded")
        if self.calls >= self.policy.max_calls:
            raise ToolAuthorizationError("tool call limit exceeded")
        if call.name not in self.policy.allowed_tools or call.name not in self.definitions:
            raise ToolAuthorizationError("tool is not authorized")
        definition = self.definitions[call.name]
        if not set(call.arguments).issubset(definition.allowed_arguments):
            raise ToolAuthorizationError("tool argument is not authorized")
        for field in definition.entity_fields:
            value = call.arguments.get(field)
            values = value if isinstance(value, list) else [value]
            if not values or any(
                not isinstance(entity_id, str)
                or entity_id not in self.policy.allowed_entity_ids
                for entity_id in values
            ):
                raise ToolAuthorizationError("entity is not authorized")
        return definition
