from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx


class RuntimeProviderError(RuntimeError):
    pass


class OpenAICompatibleRuntimeProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_token: str | None,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        self._client = client

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        output_schema: Mapping[str, Any] | None,
        settings: Mapping[str, object],
        timeout: float,
    ) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [dict(message) for message in messages],
            **settings,
        }
        if tools:
            payload["tools"] = [dict(tool) for tool in tools]
        if output_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "runtime_output",
                    "strict": True,
                    "schema": dict(output_schema),
                },
            }
        try:
            if self._client is None:
                with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
                    response = client.post(
                        f"{self._base_url}/chat/completions",
                        headers=self._headers,
                        json=payload,
                    )
            else:
                response = self._client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                    timeout=timeout,
                )
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            raw_message = choice["message"]
            message = {
                key: value
                for key, value in raw_message.items()
                if key not in {"reasoning_content", "thinking"}
            }
            raw_usage = data.get("usage") or {}
            details = raw_usage.get("completion_tokens_details") or {}
            reasoning_tokens = int(details.get("reasoning_tokens") or 0)
            completion_tokens = int(raw_usage.get("completion_tokens") or 0)
            return {
                "message": message,
                "finish_reason": choice.get("finish_reason"),
                "usage": {
                    "prompt_tokens": int(raw_usage.get("prompt_tokens") or 0),
                    "output_tokens": max(0, completion_tokens - reasoning_tokens),
                    "reasoning_tokens": reasoning_tokens,
                },
            }
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeProviderError("provider request failed") from exc
