from __future__ import annotations

from collections.abc import Callable
import json
from typing import Literal
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


class ToolAuditRecord(BaseModel):
    """Safe accounting metadata; never contains tool arguments or result bodies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    audit_id: str
    run_id: str
    stage: str
    call_id: str
    tool_name: str
    round_number: int = Field(ge=1)
    status: Literal["passed", "rejected", "failed"]
    cache_hit: bool
    result_tokens: int = Field(ge=0)
    error_code: str | None


class ToolSession:
    def __init__(
        self,
        *,
        policy: ToolPolicy,
        definitions: dict[str, ToolDefinition],
        count_tokens: Callable[[str], int],
        audit: Callable[[ToolAuditRecord], None] = lambda _record: None,
    ) -> None:
        self.policy = policy
        self.definitions = definitions
        self.count_tokens = count_tokens
        self.audit = audit
        self.calls = 0
        self.result_tokens = 0
        self._cache: dict[str, str] = {}
        self._audit_sequence = 0

    def execute(
        self,
        call: NormalizedToolCall,
        *,
        run_id: str,
        stage: str,
        round_number: int,
    ) -> ToolExecutionResult:
        cache_hit = False
        self.calls += 1
        try:
            definition = self._authorize(
                call, run_id=run_id, stage=stage, round_number=round_number
            )
            cache_key = json.dumps(
                {"name": call.name, "arguments": call.arguments},
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            cache_hit = cache_key in self._cache
            content = (
                self._cache[cache_key]
                if cache_hit
                else definition.handler(dict(call.arguments))
            )
            tokens = self.count_tokens(content)
            if self.result_tokens + tokens > self.policy.max_result_tokens:
                raise ToolAuthorizationError("tool result token limit exceeded")
            self.result_tokens += tokens
            if not cache_hit:
                self._cache[cache_key] = content
            result = ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                content=content,
                result_tokens=tokens,
            )
            self._record_audit(
                run_id=run_id,
                stage=stage,
                call=call,
                round_number=round_number,
                status="passed",
                cache_hit=cache_hit,
                result_tokens=tokens,
                error_code=None,
            )
            return result
        except ToolAuthorizationError as exc:
            self._record_audit(
                run_id=run_id,
                stage=stage,
                call=call,
                round_number=max(1, round_number),
                status="rejected",
                cache_hit=cache_hit,
                result_tokens=0,
                error_code=str(exc),
            )
            raise
        except Exception as exc:
            self._record_audit(
                run_id=run_id,
                stage=stage,
                call=call,
                round_number=max(1, round_number),
                status="failed",
                cache_hit=cache_hit,
                result_tokens=0,
                error_code=type(exc).__name__,
            )
            raise

    def _record_audit(
        self,
        *,
        run_id: str,
        stage: str,
        call: NormalizedToolCall,
        round_number: int,
        status: Literal["passed", "rejected", "failed"],
        cache_hit: bool,
        result_tokens: int,
        error_code: str | None,
    ) -> None:
        self._audit_sequence += 1
        self.audit(
            ToolAuditRecord(
                audit_id=f"tool-audit-{self._audit_sequence:06d}",
                run_id=run_id,
                stage=stage,
                call_id=call.call_id,
                tool_name=call.name,
                round_number=round_number,
                status=status,
                cache_hit=cache_hit,
                result_tokens=result_tokens,
                error_code=error_code,
            )
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
        if self.calls > self.policy.max_calls:
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
