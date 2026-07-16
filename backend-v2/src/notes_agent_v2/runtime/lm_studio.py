from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contracts import ModelIdentity


DEFAULT_MODEL_KEY = "google/gemma-4-26b-a4b-qat"


class LMStudioControlError(RuntimeError):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class ExpectedModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_key: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    format: str = Field(min_length=1)
    quantization_name: str = Field(default="4bit", min_length=1)
    bits_per_weight: int = Field(gt=0)
    loaded_context: int = Field(gt=0)
    instance_id: str | None = None

    @field_validator("model_key")
    @classmethod
    def reject_placeholder_model_key(cls, value: str) -> str:
        if value.strip() != value or value in {"your-model-name", "placeholder"}:
            raise ValueError("model_key must identify a configured model")
        return value


EXPECTED_MODEL = ExpectedModel(
    model_key=DEFAULT_MODEL_KEY,
    architecture="gemma4",
    format="mlx",
    bits_per_weight=4,
    loaded_context=40960,
)


def derive_server_base(openai_base_url: str) -> str:
    parsed = urlsplit(openai_base_url.rstrip("/"))
    if parsed.path != "/v1" and not parsed.path.endswith("/v1"):
        raise ValueError("expected an OpenAI-compatible URL ending in /v1")
    if parsed.path == "/api/v1" or parsed.path.endswith("/api/v1"):
        raise ValueError("expected an inference URL, not the native control URL")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path[:-3], "", "")).rstrip("/")


class LMStudioControlClient:
    def __init__(
        self,
        server_base_url: str,
        http_client: httpx.Client | None = None,
        *,
        api_token: str | None = None,
        timeout_seconds: float = 10,
        static_models_response: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        self.server_base_url = server_base_url.rstrip("/")
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds)
        )
        self._api_token = api_token
        self._static_models_response = static_models_response

    @classmethod
    def from_models_response(
        cls, response: dict[str, Any] | list[Any]
    ) -> LMStudioControlClient:
        return cls("http://fixture.invalid", static_models_response=response)

    def list_models(self) -> dict[str, Any]:
        if self._static_models_response is not None:
            return _normalize_models_response(self._static_models_response)
        headers = (
            {"Authorization": f"Bearer {self._api_token}"} if self._api_token else {}
        )
        try:
            response = self._client.get(
                f"{self.server_base_url}/api/v1/models", headers=headers
            )
        except httpx.TimeoutException as exc:
            raise LMStudioControlError("timeout", "LM Studio model listing timed out") from exc
        except httpx.ConnectError as exc:
            raise LMStudioControlError("connection_refused", "LM Studio is unreachable") from exc
        except httpx.HTTPError as exc:
            raise LMStudioControlError("transport_error", "LM Studio request failed") from exc
        if response.status_code == 401:
            raise LMStudioControlError("unauthorized", "LM Studio rejected authorization")
        if response.status_code == 404:
            raise LMStudioControlError("models_endpoint_not_found", "model endpoint not found")
        if response.status_code >= 400:
            raise LMStudioControlError("http_error", "LM Studio model listing failed")
        try:
            return _normalize_models_response(response.json())
        except ValueError as exc:
            raise LMStudioControlError("invalid_json", "LM Studio returned invalid JSON") from exc

    def resolve_instance(self, expected: ExpectedModel = EXPECTED_MODEL) -> ModelIdentity:
        records = _extract_models(self.list_models())
        matches = [record for record in records if _model_key(record) == expected.model_key]
        if not matches:
            raise LMStudioControlError("model_not_loaded", "approved model is not loaded")
        candidates = [
            candidate
            for record in matches
            for candidate in _loaded_candidates(record)
        ]
        if expected.instance_id:
            candidates = [c for c in candidates if c["instance_id"] == expected.instance_id]
        if not candidates:
            raise LMStudioControlError("model_not_loaded", "approved model is not loaded")
        if len(candidates) != 1:
            raise LMStudioControlError(
                "ambiguous_loaded_model", "multiple approved model instances are loaded"
            )
        candidate = candidates[0]
        _validate_candidate(candidate, expected)
        return ModelIdentity(**candidate)


def _normalize_models_response(payload: dict[str, Any] | list[Any]) -> dict[str, Any]:
    if isinstance(payload, list):
        return {"data": payload}
    if not isinstance(payload, dict):
        raise LMStudioControlError("invalid_models_response", "invalid models response")
    records = payload.get("data", payload.get("models"))
    if not isinstance(records, list):
        raise LMStudioControlError("invalid_models_response", "invalid models response")
    return {"data": records}


def _extract_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload["data"]
    if not all(isinstance(record, dict) for record in records):
        raise LMStudioControlError("invalid_models_response", "invalid models response")
    return records


def _model_key(record: dict[str, Any]) -> str | None:
    key = record.get("key") or record.get("model_key") or record.get("id")
    return key if isinstance(key, str) else None


def _loaded_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("type", "llm") != "llm":
        raise LMStudioControlError("not_llm", "loaded model is not an LLM")
    instances = record.get("instances", record.get("loaded_instances"))
    if not isinstance(instances, list) or not instances:
        return []
    candidates: list[dict[str, Any]] = []
    for instance in instances:
        if not isinstance(instance, dict) or not bool(instance.get("loaded", True)):
            continue
        if instance.get("id") in {"your-instance-id", "placeholder"}:
            raise LMStudioControlError(
                "placeholder_instance", "loaded model instance ID is a placeholder"
            )
        try:
            candidates.append(_candidate(record, instance))
        except (KeyError, TypeError, ValueError) as exc:
            raise LMStudioControlError("invalid_models_response", "missing model metadata") from exc
    return candidates


def _required(value: Any, name: str) -> Any:
    if value is None or value == "":
        raise KeyError(name)
    return value


def _candidate(record: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    quantization = record.get("quantization")
    if not isinstance(quantization, dict):
        quantization = {}
    config = instance.get("config") if isinstance(instance.get("config"), dict) else {}
    return {
        "model_key": _required(_model_key(record), "model_key"),
        "display_name": _required(
            record.get("display_name") or record.get("displayName") or record.get("name"),
            "display_name",
        ),
        "instance_id": _required(instance.get("id") or instance.get("instance_id"), "instance_id"),
        "architecture": _required(record.get("architecture"), "architecture"),
        "format": _required(record.get("format"), "format"),
        "quantization_name": _required(
            quantization.get("name") or record.get("quantization_name"), "quantization_name"
        ),
        "bits_per_weight": int(
            _required(
                quantization.get("bits_per_weight")
                or quantization.get("bitsPerWeight")
                or record.get("bits_per_weight"),
                "bits_per_weight",
            )
        ),
        "loaded_context": int(
            _required(
                instance.get("context_length")
                or instance.get("contextLength")
                or config.get("context_length")
                or config.get("contextLength"),
                "loaded_context",
            )
        ),
        "maximum_context": int(
            _required(
                record.get("max_context_length")
                or record.get("maximum_context")
                or record.get("maxContextLength"),
                "maximum_context",
            )
        ),
        "selected_variant": str(
            record.get("selected_variant") or record.get("variant") or _model_key(record)
        ),
    }


def _validate_candidate(candidate: dict[str, Any], expected: ExpectedModel) -> None:
    checks = (
        ("architecture", expected.architecture, "architecture_mismatch"),
        ("format", expected.format, "format_mismatch"),
        ("quantization_name", expected.quantization_name, "quantization_mismatch"),
        ("bits_per_weight", expected.bits_per_weight, "quantization_mismatch"),
        ("loaded_context", expected.loaded_context, "context_mismatch"),
    )
    for field, required, error_code in checks:
        if candidate[field] != required:
            raise LMStudioControlError(error_code, f"loaded model {field} differs")
