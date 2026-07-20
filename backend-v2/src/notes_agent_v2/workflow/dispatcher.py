from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from notes_agent_v2.runtime.budget import BudgetExceeded, RunBudget, RunCancelled
from notes_agent_v2.runtime.gateway import GatewayError, GatewayRequest
from notes_agent_v2.runtime.tools import ToolSession
from notes_agent_v2.workflow.evidence_tools import CLOSED_EVIDENCE_TOOLS


Role = Literal[
    "extractor", "verifier", "audience", "planner", "writer", "critic", "reviser"
]


class DispatchError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class SafeMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user"]
    content: str = Field(min_length=1, max_length=200_000)


class RoleRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    role: Role
    profile_name: str = Field(min_length=1)
    messages: tuple[SafeMessage, ...] = Field(min_length=1, max_length=16)
    allowed_tools: tuple[str, ...]
    allowed_entity_ids: tuple[str, ...] = ()
    max_tool_rounds: int = Field(default=0, ge=0)
    max_tool_calls: int = Field(default=0, ge=0)
    max_tool_result_tokens: int = Field(default=0, ge=0)
    output_schema: dict[str, Any] | None = None
    production: bool = False

    @model_validator(mode="after")
    def valid_request(self) -> RoleRequest:
        if len(self.allowed_tools) != len(set(self.allowed_tools)):
            raise ValueError("allowed tool names must be unique")
        if len(self.allowed_entity_ids) != len(set(self.allowed_entity_ids)):
            raise ValueError("allowed entity IDs must be unique")
        if sum(item.role == "system" for item in self.messages) > 1:
            raise ValueError("at most one system message is allowed")
        if not any(item.role == "user" for item in self.messages):
            raise ValueError("a user message is required")
        limits = (
            self.max_tool_rounds,
            self.max_tool_calls,
            self.max_tool_result_tokens,
        )
        if self.allowed_tools and any(value == 0 for value in limits):
            raise ValueError("enabled tools require positive limits")
        if not self.allowed_tools and (self.allowed_entity_ids or any(limits)):
            raise ValueError("tool scope requires an enabled tool")
        return self


class DispatchGateway(Protocol):
    def call(
        self,
        request: GatewayRequest,
        *,
        budget: RunBudget,
        tools: ToolSession | None = None,
        validate: Callable[[str], bool] = lambda _content: True,
    ): ...


@dataclass(frozen=True)
class DispatchDependencies:
    gateway: DispatchGateway
    budget: RunBudget
    tool_session: ToolSession | None
    tool_schemas: Mapping[str, dict[str, Any]]
    record: Callable[[dict[str, Any]], None]


_ROLE_PROFILES = {
    "extractor": frozenset({"structured_off"}),
    "verifier": frozenset({"structured_off"}),
    "audience": frozenset({"planning_reasoned", "planning_structured_off"}),
    "planner": frozenset({"planning_reasoned", "planning_structured_off"}),
    "writer": frozenset({"tool_reasoned"}),
    "critic": frozenset({"critic_structured_off"}),
    "reviser": frozenset({"tool_reasoned"}),
}
_STAGE_PROFILES = {
    ("extractor", "extract"): "structured_off",
    ("verifier", "verify"): "structured_off",
    ("audience", "audience_analysis"): "planning_reasoned",
    ("audience", "audience_finalization"): "planning_structured_off",
    ("planner", "salience"): "planning_structured_off",
    ("planner", "capability_analysis"): "planning_reasoned",
    ("planner", "capability_finalization"): "planning_structured_off",
    ("writer", "write"): "tool_reasoned",
    ("critic", "critic"): "critic_structured_off",
    ("reviser", "revise"): "tool_reasoned",
}
_TOOL_ROLES = {"writer", "critic", "reviser"}


class BoundedDispatcher:
    def __init__(self, dependencies: DispatchDependencies) -> None:
        self._deps = dependencies
        self._active = Lock()
        self._sequence = 0

    def dispatch(
        self,
        request: RoleRequest,
        *,
        validate: Callable[[str], bool] = lambda _content: True,
    ):
        if not self._active.acquire(blocking=False):
            self._write_record(request, status="rejected", error_code="parallel_dispatch")
            raise DispatchError("parallel_dispatch")
        try:
            self._enforce_policy(request)
            schemas = tuple(
                dict(self._deps.tool_schemas[name]) for name in request.allowed_tools
            )
            gateway_request = GatewayRequest(
                run_id=request.run_id,
                stage=request.stage,
                role=request.role,
                profile_name=request.profile_name,
                messages=tuple(item.model_dump() for item in request.messages),
                tools=schemas,
                output_schema=(
                    dict(request.output_schema)
                    if request.output_schema is not None
                    else None
                ),
                production=request.production,
            )
            result = self._deps.gateway.call(
                gateway_request,
                budget=self._deps.budget,
                tools=self._deps.tool_session if schemas else None,
                validate=validate,
            )
            self._write_record(request, status="passed", error_code=None)
            return result
        except DispatchError as exc:
            self._write_record(request, status="failed", error_code=exc.error_code)
            raise
        except GatewayError as exc:
            self._write_record(request, status="failed", error_code=exc.error_code)
            raise DispatchError(exc.error_code) from exc
        except BudgetExceeded as exc:
            self._write_record(request, status="failed", error_code="budget_exceeded")
            raise DispatchError("budget_exceeded") from exc
        except RunCancelled as exc:
            self._write_record(request, status="failed", error_code="cancelled")
            raise DispatchError("cancelled") from exc
        except Exception as exc:
            self._write_record(request, status="failed", error_code="dispatch_failure")
            raise DispatchError("dispatch_failure") from exc
        finally:
            self._active.release()

    def _enforce_policy(self, request: RoleRequest) -> None:
        if request.profile_name not in _ROLE_PROFILES[request.role]:
            raise DispatchError("profile_not_allowed")
        expected_profile = _STAGE_PROFILES.get((request.role, request.stage))
        if expected_profile is None:
            raise DispatchError("stage_not_allowed")
        if request.profile_name != expected_profile:
            raise DispatchError("profile_not_allowed")
        if request.allowed_tools and request.role not in _TOOL_ROLES:
            raise DispatchError("tool_not_allowed")
        for name in request.allowed_tools:
            if name not in CLOSED_EVIDENCE_TOOLS:
                raise DispatchError("tool_not_allowed")
            if name not in self._deps.tool_schemas:
                raise DispatchError("tool_not_configured")
        if request.allowed_tools and self._deps.tool_session is None:
            raise DispatchError("tool_session_unavailable")
        if request.allowed_tools:
            policy = self._deps.tool_session.policy
            scope_matches = (
                policy.run_id == request.run_id
                and policy.stage == request.stage
                and policy.allowed_tools == frozenset(request.allowed_tools)
                and policy.allowed_entity_ids == frozenset(request.allowed_entity_ids)
                and policy.max_rounds == request.max_tool_rounds
                and policy.max_calls == request.max_tool_calls
                and policy.max_result_tokens == request.max_tool_result_tokens
            )
            if not scope_matches:
                raise DispatchError("tool_scope_mismatch")

    def _write_record(
        self,
        request: RoleRequest,
        *,
        status: Literal["passed", "rejected", "failed"],
        error_code: str | None,
    ) -> None:
        self._sequence += 1
        self._deps.record(
            {
                "dispatch_id": f"dispatch-{self._sequence:06d}",
                "run_id": request.run_id,
                "stage": request.stage,
                "role": request.role,
                "profile_name": request.profile_name,
                "message_count": len(request.messages),
                "tool_count": len(request.allowed_tools),
                "entity_scope_count": len(request.allowed_entity_ids),
                "status": status,
                "error_code": error_code,
            }
        )
