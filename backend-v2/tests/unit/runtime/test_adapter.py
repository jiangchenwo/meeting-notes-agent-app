from __future__ import annotations

import pytest

from notes_agent_v2.runtime.adapter import (
    ReasoningLeakError,
    UnsafeSerializationError,
    assert_safe_serialization,
    build_finalization_history,
    build_tool_loop_history,
    normalize_response,
)


@pytest.mark.parametrize(
    ("raw", "content", "reasoning_observed"),
    [
        (
            {
                "parts": [
                    {"type": "thinking", "content": "private"},
                    {"type": "text", "content": "answer"},
                ]
            },
            "answer",
            True,
        ),
        (
            {"message": {"reasoning_content": "private", "content": "answer"}},
            "answer",
            True,
        ),
        (
            {"message": {"content": "<thought>private</thought>answer"}},
            "answer",
            True,
        ),
        (
            {"message": {"content": "<|channel>thought private<|channel>final answer"}},
            "answer",
            True,
        ),
        (
            {"parts": [{"type": "thinking", "content": ""}, {"type": "text", "content": "ok"}]},
            "ok",
            True,
        ),
        (
            {"message": {"content": "Result:\n```json\n{\"ok\": true}\n```"}},
            "Result:\n```json\n{\"ok\": true}\n```",
            False,
        ),
    ],
)
def test_response_shapes_discard_reasoning(
    raw: dict[str, object], content: str, reasoning_observed: bool
) -> None:
    result = normalize_response(raw)
    assert result.final_content == content
    assert result.reasoning_observed is reasoning_observed


def test_tool_calls_and_usage_are_normalized() -> None:
    result = normalize_response(
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "lookup", "arguments": "{\"id\": \"f1\"}"},
                    },
                    {"id": "c2", "name": "search", "arguments": {"query": "alpha"}},
                ],
            },
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "reasoning_tokens": 2,
            },
            "finish_reason": "tool_calls",
        }
    )
    assert [(call.call_id, call.name) for call in result.tool_calls] == [
        ("c1", "lookup"),
        ("c2", "search"),
    ]
    assert result.tool_calls[0].arguments == {"id": "f1"}
    assert result.usage.total_tokens == 15
    assert result.finish_reason == "tool_calls"


def test_multi_tool_parts_are_normalized_in_order() -> None:
    result = normalize_response(
        {
            "parts": [
                {"type": "tool_call", "id": "a", "name": "one", "arguments": {}},
                {"type": "tool_call", "id": "b", "name": "two", "arguments": {}},
            ]
        }
    )
    assert [call.name for call in result.tool_calls] == ["one", "two"]


@pytest.mark.parametrize(
    "content",
    [
        "Thinking: private answer",
        "answer <thought>private</thought>",
        "answer <|channel>thought private",
        "reasoning_content: private",
        "Analysis: hidden chain",
    ],
)
def test_visible_reasoning_is_rejected(content: str) -> None:
    with pytest.raises(ReasoningLeakError):
        normalize_response({"message": {"content": content}})


def test_hidden_reasoning_is_available_only_during_tool_round_trip() -> None:
    history = [{"role": "user", "content": "use a tool"}]
    assistant = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "private plan",
        "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"id": "f1"}}],
    }
    tool_result = {"role": "tool", "tool_call_id": "c1", "content": "safe result"}

    loop_history = build_tool_loop_history(history, assistant, tool_result)
    finalization = build_finalization_history(loop_history)

    assert loop_history[1]["reasoning_content"] == "private plan"
    assert "reasoning_content" not in finalization[1]
    assert "private plan" not in str(finalization)
    assert history == [{"role": "user", "content": "use a tool"}]


def test_finalization_recursively_removes_nested_reasoning_fields() -> None:
    finalization = build_finalization_history(
        [
            {
                "role": "assistant",
                "content": "safe",
                "metadata": {"analysis": "private", "safe_id": "x"},
            }
        ]
    )
    assert finalization == [
        {"role": "assistant", "content": "safe", "metadata": {"safe_id": "x"}}
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"reasoning": "x"},
        {"outer": [{"thinking_trace": "x"}]},
        {"thought": "x"},
        {"analysis": "x"},
        {"raw_prompt": "x"},
        {"Authorization": "Bearer x"},
        {"api_key": "x"},
    ],
)
def test_safe_serialization_recursively_rejects_private_keys(payload: object) -> None:
    with pytest.raises(UnsafeSerializationError):
        assert_safe_serialization(payload)


def test_safe_serialization_accepts_normalized_response() -> None:
    payload = normalize_response({"message": {"content": "safe"}}).model_dump(mode="json")
    assert_safe_serialization(payload)
