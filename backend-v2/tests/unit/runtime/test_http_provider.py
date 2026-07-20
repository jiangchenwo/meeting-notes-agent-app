from __future__ import annotations

import httpx
import pytest

from notes_agent_v2.runtime.http_provider import OpenAICompatibleRuntimeProvider, RuntimeProviderError


def test_provider_maps_structured_request_and_returns_safe_gateway_shape() -> None:
    captured: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "model-key",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"status":"ok"}',
                            "reasoning_content": "hidden",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 17,
                    "completion_tokens_details": {"reasoning_tokens": 13},
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handle))
    provider = OpenAICompatibleRuntimeProvider(
        base_url="http://runtime.test/v1", api_token="secret", client=client
    )
    result = provider.complete(
        model="model-key",
        messages=[{"role": "user", "content": "return status"}],
        tools=[],
        output_schema={"type": "object"},
        settings={"reasoning_effort": "none", "max_tokens": 128},
        timeout=3,
    )

    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "runtime_output", "strict": True, "schema": {"type": "object"}},
    }
    assert result["message"]["content"] == '{"status":"ok"}'
    assert "reasoning_content" not in result["message"]
    assert result["usage"] == {"prompt_tokens": 11, "output_tokens": 4, "reasoning_tokens": 13}


def test_provider_errors_are_typed_without_response_or_authorization_text() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="PRIVATE PROVIDER BODY")

    provider = OpenAICompatibleRuntimeProvider(
        base_url="http://runtime.test/v1",
        api_token="secret",
        client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    with pytest.raises(RuntimeProviderError, match="provider request failed") as caught:
        provider.complete(
            model="model-key",
            messages=[],
            tools=[],
            output_schema=None,
            settings={},
            timeout=3,
        )
    assert "PRIVATE" not in str(caught.value)
    assert "secret" not in str(caught.value)
