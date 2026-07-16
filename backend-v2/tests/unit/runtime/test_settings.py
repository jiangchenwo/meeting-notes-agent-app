from __future__ import annotations

import json
from pathlib import Path

import pytest

from notes_agent_v2.runtime.settings import (
    DEFAULT_CONFIG_PATH,
    RuntimeConfigurationError,
    load_runtime_settings,
)


def write_config(path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "schema_version": "runtime-settings-v1",
        "provider": "lm_studio_openai",
        "inference_base_url": "http://localhost:1234/v1",
        "control_base_url": "http://localhost:1234",
        "control_timeout_seconds": 10,
        "inference_timeout_seconds": 240,
        "profiles_path": "profiles.json",
        "model": {
            "model_key": "acme/future-model",
            "instance_id": "future-1",
            "architecture": "future_arch",
            "format": "mlx",
            "quantization_name": "4bit",
            "bits_per_weight": 4,
            "loaded_context": 32768,
        },
        "context": {
            "hard_context": 32768,
            "initial_prompt_limit": 18432,
            "generation_limit": 8192,
            "tool_result_limit": 3072,
            "safety_margin": 3072,
        },
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload))
    return path


def test_json_configures_model_connection_context_and_profiles(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "runtime.json")
    settings = load_runtime_settings(config_path, environ={})

    assert settings.provider == "lm_studio_openai"
    assert str(settings.inference_base_url) == "http://localhost:1234/v1"
    assert str(settings.control_base_url) == "http://localhost:1234/"
    assert settings.model.model_key == "acme/future-model"
    assert settings.model.architecture == "future_arch"
    assert settings.model.loaded_context == 32768
    assert settings.context.hard_context == 32768
    assert settings.profiles_path == (tmp_path / "profiles.json").resolve()


def test_environment_overrides_json_and_keeps_token_out_of_serialization(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path / "runtime.json")
    settings = load_runtime_settings(
        config_path,
        environ={
            "NOTES_RUNTIME_MODEL_KEY": "override/model",
            "NOTES_RUNTIME_MODEL_INSTANCE_ID": "override-1",
            "NOTES_RUNTIME_INFERENCE_BASE_URL": "https://example.test/v1",
            "NOTES_RUNTIME_CONTROL_TIMEOUT_SECONDS": "12.5",
            "NOTES_RUNTIME_API_TOKEN": "private-token",
        },
    )

    assert settings.model.model_key == "override/model"
    assert settings.model.instance_id == "override-1"
    assert str(settings.inference_base_url) == "https://example.test/v1"
    assert settings.control_timeout_seconds == 12.5
    assert settings.api_token == "private-token"
    assert "private-token" not in str(settings)
    assert "api_token" not in settings.model_dump(mode="json")


def test_config_file_path_can_come_from_environment(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "runtime.json")
    settings = load_runtime_settings(
        environ={"NOTES_RUNTIME_CONFIG_FILE": str(config_path)}
    )
    assert settings.model.model_key == "acme/future-model"


def test_dotenv_overrides_json_and_process_environment_wins(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "runtime.json")
    env_path = tmp_path / ".env"
    env_path.write_text(
        "NOTES_RUNTIME_MODEL_KEY=dotenv/model\n"
        "NOTES_RUNTIME_MODEL_INSTANCE_ID=dotenv-1\n"
        "NOTES_RUNTIME_API_TOKEN=dotenv-secret\n"
    )
    settings = load_runtime_settings(
        config_path,
        env_file=env_path,
        environ={"NOTES_RUNTIME_MODEL_KEY": "process/model"},
    )

    assert settings.model.model_key == "process/model"
    assert settings.model.instance_id == "dotenv-1"
    assert settings.api_token == "dotenv-secret"
    assert "dotenv-secret" not in str(settings.model_dump(mode="json"))


def test_dotenv_can_select_json_configuration(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "runtime.json")
    env_path = tmp_path / ".env"
    env_path.write_text(f"NOTES_RUNTIME_CONFIG_FILE={config_path}\n")

    settings = load_runtime_settings(env_file=env_path, environ={})
    assert settings.model.model_key == "acme/future-model"


def test_secret_in_json_is_rejected(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "runtime.json", api_token="must-not-be-here")
    with pytest.raises(RuntimeConfigurationError) as exc_info:
        load_runtime_settings(config_path, environ={})
    assert exc_info.value.error_code == "invalid_configuration"


def test_unsupported_provider_fails_closed(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "runtime.json", provider="future_api")
    with pytest.raises(RuntimeConfigurationError) as exc_info:
        load_runtime_settings(config_path, environ={})
    assert exc_info.value.error_code == "unsupported_provider"


def test_model_context_and_context_envelope_must_match(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "runtime.json",
        context={
            "hard_context": 40960,
            "initial_prompt_limit": 24576,
            "generation_limit": 8192,
            "tool_result_limit": 4096,
            "safety_margin": 4096,
        },
    )
    with pytest.raises(RuntimeConfigurationError, match="loaded context"):
        load_runtime_settings(config_path, environ={})


def test_missing_file_is_typed() -> None:
    with pytest.raises(RuntimeConfigurationError) as exc_info:
        load_runtime_settings(Path("missing-runtime.json"), environ={})
    assert exc_info.value.error_code == "configuration_not_found"


def test_checked_in_runtime_configuration_is_valid() -> None:
    settings = load_runtime_settings(DEFAULT_CONFIG_PATH, environ={})
    assert settings.model.model_key == "google/gemma-4-26b-a4b-qat"
    assert settings.context.hard_context == 40960
    assert settings.profiles_path.name == "profiles.json"
