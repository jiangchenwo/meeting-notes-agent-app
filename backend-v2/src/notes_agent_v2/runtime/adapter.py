from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import NormalizedResponse, NormalizedToolCall, NormalizedUsage


class ReasoningLeakError(ValueError):
    pass


class UnsafeSerializationError(ValueError):
    pass


_XML_THOUGHT = re.compile(r"^\s*<thought>.*?</thought>\s*", re.IGNORECASE | re.DOTALL)
_CHANNEL_THOUGHT = re.compile(
    r"^\s*<\|channel>thought\b.*?<\|channel>final\s*", re.IGNORECASE | re.DOTALL
)
_VISIBLE_MARKERS = (
    "<thought>",
    "</thought>",
    "<|channel>thought",
    "reasoning_content",
    "chain_of_thought",
)
_VISIBLE_LABEL = re.compile(r"(?:^|\n)\s*(?:thinking|analysis|reasoning)\s*:", re.IGNORECASE)
_PRIVATE_KEY_PARTS = (
    "reasoning",
    "thinking",
    "thought",
    "analysis",
    "raw_prompt",
    "authorization",
    "api_key",
)


def normalize_response(raw: Mapping[str, Any]) -> NormalizedResponse:
    final_parts: list[str] = []
    tool_calls: list[NormalizedToolCall] = []
    reasoning_observed = False

    parts = raw.get("parts")
    if isinstance(parts, Sequence) and not isinstance(parts, (str, bytes)):
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            part_type = part.get("type")
            if part_type in {"thinking", "reasoning"}:
                reasoning_observed = True
            elif part_type == "text":
                final_parts.append(str(part.get("content") or ""))
            elif part_type == "tool_call":
                tool_calls.append(_normalize_tool_call(part))

    message = raw.get("message")
    if isinstance(message, Mapping):
        if "reasoning_content" in message or "thinking" in message:
            reasoning_observed = True
        content, hidden = _strip_leading_hidden_reasoning(str(message.get("content") or ""))
        reasoning_observed = reasoning_observed or hidden
        if content:
            final_parts.append(content)
        message_calls = message.get("tool_calls")
        if isinstance(message_calls, Sequence) and not isinstance(message_calls, (str, bytes)):
            for call in message_calls:
                if isinstance(call, Mapping):
                    tool_calls.append(_normalize_tool_call(call))

    final_content = "\n".join(part for part in final_parts if part).strip()
    _reject_visible_reasoning(final_content)
    usage = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
    return NormalizedResponse(
        final_content=final_content,
        tool_calls=tuple(tool_calls),
        usage=_normalize_usage(usage),
        finish_reason=str(raw["finish_reason"]) if raw.get("finish_reason") is not None else None,
        reasoning_observed=reasoning_observed,
    )


def build_tool_loop_history(
    history: Sequence[Mapping[str, Any]],
    assistant_message: Mapping[str, Any],
    tool_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    copied = copy.deepcopy(list(history))
    copied.append(copy.deepcopy(dict(assistant_message)))
    copied.append(copy.deepcopy(dict(tool_result)))
    return copied


def build_finalization_history(
    history: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [_strip_private_fields(copy.deepcopy(dict(message))) for message in history]


def assert_safe_serialization(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = str(key).lower()
            if normalized_key != "reasoning_observed" and any(
                part in normalized_key for part in _PRIVATE_KEY_PARTS
            ):
                raise UnsafeSerializationError(f"private field cannot be serialized: {key}")
            assert_safe_serialization(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            assert_safe_serialization(child)


def _strip_leading_hidden_reasoning(content: str) -> tuple[str, bool]:
    for pattern in (_XML_THOUGHT, _CHANNEL_THOUGHT):
        match = pattern.match(content)
        if match:
            return content[match.end() :].strip(), True
    return content.strip(), False


def _reject_visible_reasoning(content: str) -> None:
    lowered = content.lower()
    if _VISIBLE_LABEL.search(content) or any(marker in lowered for marker in _VISIBLE_MARKERS):
        raise ReasoningLeakError("visible final content contains reasoning")


def _normalize_usage(usage: Mapping[str, Any]) -> NormalizedUsage:
    return NormalizedUsage(
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or usage.get("total_output_tokens")
            or 0
        ),
        tool_result_tokens=int(usage.get("tool_result_tokens") or 0),
        detail_available=bool(usage),
    )


def _normalize_tool_call(call: Mapping[str, Any]) -> NormalizedToolCall:
    function = call.get("function")
    source = function if isinstance(function, Mapping) else call
    arguments = source.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("tool arguments must be valid JSON") from exc
    if not isinstance(arguments, Mapping):
        raise ValueError("tool arguments must be an object")
    return NormalizedToolCall(
        call_id=str(call.get("id") or source.get("id") or ""),
        name=str(source.get("name") or ""),
        arguments=dict(arguments),
    )


def _strip_private_fields(message: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in message.items():
        if any(part in key.lower() for part in _PRIVATE_KEY_PARTS):
            continue
        if key == "content" and isinstance(value, str):
            value, _ = _strip_leading_hidden_reasoning(value)
            _reject_visible_reasoning(value)
        result[key] = _strip_private_value(value)
    return result


def _strip_private_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _strip_private_fields(dict(value))
    if isinstance(value, list):
        return [_strip_private_value(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_strip_private_value(child) for child in value)
    return value
