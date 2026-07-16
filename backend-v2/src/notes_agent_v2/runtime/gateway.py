from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .adapter import (
    assert_safe_serialization,
    build_finalization_history,
    build_tool_loop_history,
    normalize_response,
)
from .budget import RunBudget
from .context import ContextEnvelope, ContextLedger, PromptTokenizer
from .contracts import ModelIdentity, NormalizedResponse, RuntimeReport, assert_runtime_ready
from .lm_studio import ExpectedModel
from .profiles import ProfileCatalog, StageProfile
from .tools import ToolAuthorizationError, ToolSession


class Provider(Protocol):
    def complete(self, **payload: Any) -> Mapping[str, Any]: ...


class RuntimeControl(Protocol):
    def resolve_instance(self, expected: ExpectedModel) -> ModelIdentity: ...


class GatewayError(RuntimeError):
    def __init__(self, error_code: str, message: str | None = None) -> None:
        super().__init__(message or error_code)
        self.error_code = error_code


class GatewayRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    role: str = Field(min_length=1)
    profile_name: str = Field(min_length=1)
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] = ()
    output_schema: dict[str, Any] | None = None
    production: bool = False


class SafeCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    run_id: str
    stage: str
    role: str
    profile_name: str
    runtime_fingerprint: str
    profile_fingerprint: str
    prompt_fingerprint: str
    finish_reason: str | None
    reasoning_observed: bool
    usage: dict[str, Any]
    tool_audit_ids: tuple[str, ...] = ()


class GatewayResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    response: NormalizedResponse
    record: SafeCallRecord


@dataclass(frozen=True)
class GatewayDependencies:
    control: RuntimeControl
    runtime_report: Callable[[], RuntimeReport]
    tokenizer: PromptTokenizer
    provider: Provider
    profiles: ProfileCatalog
    expected_model: ExpectedModel
    context_envelope: ContextEnvelope
    inference_timeout_seconds: float
    record: Callable[[dict[str, Any]], None]
    event: Callable[[str], None] = lambda _event: None


class RuntimeGateway:
    def __init__(self, dependencies: GatewayDependencies) -> None:
        self._deps = dependencies

    def call(
        self,
        request: GatewayRequest,
        *,
        budget: RunBudget,
        tools: ToolSession | None = None,
        validate: Callable[[str], bool] = lambda _content: True,
    ) -> GatewayResult:
        identity = self._deps.control.resolve_instance(self._deps.expected_model)
        report = self._deps.runtime_report()
        assert_runtime_ready(report)
        if identity != report.identity:
            raise GatewayError("runtime_drift", "runtime report does not match loaded instance")
        self._deps.event("runtime_ready")

        profile = self._deps.profiles.resolve(
            request.profile_name, production=request.production
        )
        self._deps.event("profile_resolved")

        response, raw = self._invoke_counted(
            request=request,
            profile=profile,
            messages=list(request.messages),
            tools=list(request.tools),
            output_schema=request.output_schema,
            budget=budget,
            identity=identity,
            force_structured_off=False,
        )
        self._deps.event("prompt_counted")
        self._deps.event("budgets_reserved")
        self._deps.event("provider_invoked")
        self._deps.event("response_normalized")

        tool_audit_ids: list[str] = []
        if response.tool_calls:
            if tools is None:
                raise GatewayError("unauthorized_tool", "tool calls are not enabled")
            tool_results = []
            try:
                for call in response.tool_calls:
                    budget.reserve_tool_call()
                    result = tools.execute(
                        call,
                        run_id=request.run_id,
                        stage=request.stage,
                        round_number=1,
                    )
                    tool_audit_ids.append(f"tool-{uuid4().hex}")
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.call_id,
                            "content": result.content,
                        }
                    )
            except ToolAuthorizationError as exc:
                raise GatewayError("unauthorized_tool", str(exc)) from exc
            self._deps.event("tools_authorized")

            if request.output_schema is not None:
                assistant_message = raw.get("message")
                if not isinstance(assistant_message, Mapping):
                    assistant_message = {}
                history: list[dict[str, Any]] = list(request.messages)
                for tool_result in tool_results:
                    history = build_tool_loop_history(history, assistant_message, tool_result)
                safe_history = build_finalization_history(history)
                response, raw = self._invoke_counted(
                    request=request,
                    profile=profile,
                    messages=safe_history,
                    tools=[],
                    output_schema=request.output_schema,
                    budget=budget,
                    identity=identity,
                    force_structured_off=True,
                )
                self._deps.event("structured_finalized")

        if not validate(response.final_content):
            response = self._retry_invalid_result(
                request=request,
                profile=profile,
                budget=budget,
                identity=identity,
                validate=validate,
            )
        self._deps.event("result_validated")

        record = self._build_record(request, report, profile, response, tool_audit_ids)
        payload = record.model_dump(mode="json")
        assert_safe_serialization(payload)
        self._deps.record(payload)
        self._deps.event("safe_metadata_recorded")
        self._deps.event("budgets_reconciled")
        return GatewayResult(response=response, record=record)

    def _retry_invalid_result(
        self,
        *,
        request: GatewayRequest,
        profile: StageProfile,
        budget: RunBudget,
        identity: Any,
        validate: Callable[[str], bool],
    ) -> NormalizedResponse:
        if profile.parse_retries != 1:
            raise GatewayError("invalid_output")
        retried, _ = self._invoke_counted(
            request=request,
            profile=profile,
            messages=list(request.messages),
            tools=[],
            output_schema=request.output_schema,
            budget=budget,
            identity=identity,
            force_structured_off=True,
        )
        if validate(retried.final_content):
            return retried
        if request.profile_name == "critic_structured_off":
            raise GatewayError("critic_failure")
        raise GatewayError("invalid_output")

    def _invoke_counted(
        self,
        *,
        request: GatewayRequest,
        profile: StageProfile,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        output_schema: dict[str, Any] | None,
        budget: RunBudget,
        identity: Any,
        force_structured_off: bool,
    ) -> tuple[NormalizedResponse, Mapping[str, Any]]:
        context = ContextLedger(envelope=self._deps.context_envelope)
        context.reserve_call(
            messages=messages,
            tools=tools or None,
            output_schema=output_schema,
            tokenizer=self._deps.tokenizer,
            model_key=identity.model_key,
            instance_id=identity.instance_id,
        )
        budget.reserve_model_request(
            input_tokens=context.initial_prompt_tokens,
            output_tokens=profile.max_tokens,
            elapsed_seconds=0,
        )
        try:
            raw = self._invoke_provider(
                model=identity.model_key,
                messages=messages,
                tools=tools,
                output_schema=output_schema,
                settings=profile.provider_settings(
                    force_structured_off=force_structured_off
                ),
                timeout=self._deps.inference_timeout_seconds,
            )
            response = normalize_response(raw)
            raw_usage = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
            hidden_tokens = int(raw_usage.get("reasoning_tokens") or 0)
            budgeted_output_tokens = response.usage.output_tokens + hidden_tokens
            context.reconcile(
                prompt_tokens=response.usage.prompt_tokens,
                output_tokens=budgeted_output_tokens,
            )
            budget.reconcile_model_request(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=budgeted_output_tokens,
            )
            return response, raw
        except Exception:
            if budget.active_model_requests:
                budget.active_model_requests = 0
                budget.reserved_input_tokens = 0
                budget.reserved_output_tokens = 0
            raise

    def _invoke_provider(self, **payload: Any) -> Mapping[str, Any]:
        try:
            return self._deps.provider.complete(**payload)
        except Exception as exc:
            raise GatewayError("model_invocation_failed") from exc

    @staticmethod
    def _build_record(
        request: GatewayRequest,
        report: RuntimeReport,
        profile: StageProfile,
        response: NormalizedResponse,
        tool_audit_ids: list[str],
    ) -> SafeCallRecord:
        prompt_metadata = {
            "stage": request.stage,
            "role": request.role,
            "profile": request.profile_name,
            "message_count": len(request.messages),
            "tool_names": [tool.get("name") for tool in request.tools],
            "structured": request.output_schema is not None,
        }
        canonical = json.dumps(prompt_metadata, sort_keys=True, separators=(",", ":"))
        return SafeCallRecord(
            call_id=f"call-{uuid4().hex}",
            run_id=request.run_id,
            stage=request.stage,
            role=request.role,
            profile_name=profile.name,
            runtime_fingerprint=report.fingerprint,
            profile_fingerprint=profile.fingerprint,
            prompt_fingerprint=hashlib.sha256(canonical.encode()).hexdigest(),
            finish_reason=response.finish_reason,
            reasoning_observed=response.reasoning_observed,
            usage=response.usage.model_dump(mode="json"),
            tool_audit_ids=tuple(tool_audit_ids),
        )
