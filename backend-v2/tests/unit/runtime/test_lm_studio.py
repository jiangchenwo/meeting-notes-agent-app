from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from notes_agent_v2.runtime.lm_studio import (
    EXPECTED_MODEL,
    ExpectedModel,
    LMStudioControlClient,
    LMStudioControlError,
    derive_server_base,
)


def model_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "key": "google/gemma-4-26b-a4b-qat",
        "display_name": "Gemma 4 26B A4B QAT",
        "type": "llm",
        "architecture": "gemma4",
        "format": "mlx",
        "quantization": {"name": "4bit", "bits_per_weight": 4},
        "max_context_length": 131072,
        "selected_variant": "26b-a4b-qat",
        "instances": [{"id": "loaded-1", "loaded": True, "context_length": 40960}],
    }
    record.update(overrides)
    return record


def resolve(*records: dict[str, object]):
    return LMStudioControlClient.from_models_response({"data": list(records)}).resolve_instance(
        EXPECTED_MODEL
    )


def test_exact_loaded_instance_is_accepted() -> None:
    found = resolve(model_record())
    assert found.model_key == "google/gemma-4-26b-a4b-qat"
    assert found.architecture == "gemma4"
    assert found.format == "mlx"
    assert found.bits_per_weight == 4
    assert found.loaded_context == 40960


@pytest.mark.parametrize(
    ("record", "error_code"),
    [
        (model_record(key="other/model"), "model_not_loaded"),
        (model_record(instances=[]), "model_not_loaded"),
        (model_record(type="embedding"), "not_llm"),
        (
            model_record(instances=[{"id": "your-instance-id", "loaded": True, "context_length": 40960}]),
            "placeholder_instance",
        ),
        (model_record(architecture="gemma3"), "architecture_mismatch"),
        (model_record(format="gguf"), "format_mismatch"),
        (
            model_record(quantization={"name": "8bit", "bits_per_weight": 8}),
            "quantization_mismatch",
        ),
        (
            model_record(instances=[{"id": "loaded-1", "loaded": True, "context_length": 32768}]),
            "context_mismatch",
        ),
    ],
)
def test_runtime_drift_is_rejected(record: dict[str, object], error_code: str) -> None:
    with pytest.raises(LMStudioControlError) as exc_info:
        resolve(record)
    assert exc_info.value.error_code == error_code


def test_ambiguous_loaded_instances_are_rejected() -> None:
    record = model_record(
        instances=[
            {"id": "one", "loaded": True, "context_length": 40960},
            {"id": "two", "loaded": True, "context_length": 40960},
        ]
    )
    with pytest.raises(LMStudioControlError) as exc_info:
        resolve(record)
    assert exc_info.value.error_code == "ambiguous_loaded_model"


@pytest.mark.parametrize("value", ["", "your-model-name", " model "])
def test_placeholder_or_inexact_model_keys_are_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        ExpectedModel(
            model_key=value,
            architecture="gemma4",
            format="mlx",
            bits_per_weight=4,
            loaded_context=40960,
        )


def test_http_errors_are_typed_and_authorization_is_not_exposed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer private-token"
        return httpx.Response(401)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    control = LMStudioControlClient("http://localhost:1234", client, api_token="private-token")
    with pytest.raises(LMStudioControlError) as exc_info:
        control.list_models()
    assert exc_info.value.error_code == "unauthorized"
    assert "private-token" not in str(exc_info.value)


def test_timeout_is_typed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    control = LMStudioControlClient(
        "http://localhost:1234", httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(LMStudioControlError) as exc_info:
        control.list_models()
    assert exc_info.value.error_code == "timeout"


def test_control_client_uses_configured_timeout() -> None:
    control = LMStudioControlClient("http://localhost:1234", timeout_seconds=12.5)
    assert control._client.timeout.read == 12.5


def test_server_base_is_derived_only_from_openai_v1_url() -> None:
    assert derive_server_base("http://localhost:1234/v1") == "http://localhost:1234"
    with pytest.raises(ValueError):
        derive_server_base("http://localhost:1234/api/v1")
