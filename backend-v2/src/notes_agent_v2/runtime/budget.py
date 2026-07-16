from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BudgetExceeded(RuntimeError):
    pass


class RunCancelled(RuntimeError):
    pass


class RunBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_model_requests: int = Field(default=64, gt=0)
    max_tool_calls: int = Field(default=64, ge=0)
    max_input_tokens: int = Field(default=600_000, gt=0)
    max_output_tokens: int = Field(default=80_000, gt=0)
    max_wall_seconds: int = Field(default=3600, gt=0)
    max_revisions: int = Field(default=2, ge=0)
    max_parallel_model_requests: int = Field(default=1, gt=0)

    model_requests: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    revisions: int = Field(default=0, ge=0)
    active_model_requests: int = Field(default=0, ge=0)
    reserved_input_tokens: int = Field(default=0, ge=0)
    reserved_output_tokens: int = Field(default=0, ge=0)
    cancelled: bool = False

    @model_validator(mode="after")
    def require_serial_execution(self) -> RunBudget:
        if self.max_parallel_model_requests != 1:
            raise ValueError("max_parallel_model_requests must be exactly 1")
        return self

    def reserve_model_request(
        self, *, input_tokens: int, output_tokens: int, elapsed_seconds: int
    ) -> None:
        self._ensure_active()
        if self.active_model_requests >= self.max_parallel_model_requests:
            raise BudgetExceeded("parallel model request limit exceeded")
        if self.model_requests >= self.max_model_requests:
            raise BudgetExceeded("model request limit exceeded")
        if elapsed_seconds > self.max_wall_seconds:
            raise BudgetExceeded("wall time limit exceeded")
        if self.input_tokens + input_tokens > self.max_input_tokens:
            raise BudgetExceeded("input token limit exceeded")
        if self.output_tokens + output_tokens > self.max_output_tokens:
            raise BudgetExceeded("output token limit exceeded")
        self.model_requests += 1
        self.active_model_requests += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.reserved_input_tokens = input_tokens
        self.reserved_output_tokens = output_tokens

    def reconcile_model_request(self, *, input_tokens: int, output_tokens: int) -> None:
        if self.active_model_requests != 1:
            raise BudgetExceeded("no active model request to reconcile")
        reconciled_input = self.input_tokens - self.reserved_input_tokens + input_tokens
        reconciled_output = self.output_tokens - self.reserved_output_tokens + output_tokens
        if reconciled_input > self.max_input_tokens:
            raise BudgetExceeded("actual input token limit exceeded")
        if reconciled_output > self.max_output_tokens:
            raise BudgetExceeded("actual output token limit exceeded")
        self.input_tokens = reconciled_input
        self.output_tokens = reconciled_output
        self.active_model_requests = 0
        self.reserved_input_tokens = 0
        self.reserved_output_tokens = 0

    def reserve_tool_call(self) -> None:
        self._ensure_active()
        if self.tool_calls >= self.max_tool_calls:
            raise BudgetExceeded("tool call limit exceeded")
        self.tool_calls += 1

    def reserve_revision(self) -> None:
        self._ensure_active()
        if self.revisions >= self.max_revisions:
            raise BudgetExceeded("revision limit exceeded")
        self.revisions += 1

    def cancel(self) -> None:
        self.cancelled = True

    def _ensure_active(self) -> None:
        if self.cancelled:
            raise RunCancelled("run is cancelled")
