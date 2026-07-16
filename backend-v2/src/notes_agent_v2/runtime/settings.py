from __future__ import annotations

from collections.abc import Mapping
import copy
import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, ValidationError, model_validator

from .context import ContextEnvelope
from .lm_studio import ExpectedModel


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "runtime.json"


class RuntimeConfigurationError(RuntimeError):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runtime-settings-v1"] = "runtime-settings-v1"
    provider: Literal["lm_studio_openai"] = "lm_studio_openai"
    inference_base_url: AnyHttpUrl = "http://localhost:1234/v1"
    control_base_url: AnyHttpUrl = "http://localhost:1234"
    control_timeout_seconds: float = Field(default=10, gt=0, le=600)
    inference_timeout_seconds: float = Field(default=240, gt=0, le=3600)
    profiles_path: Path = Path("profiles.json")
    model: ExpectedModel = Field(
        default_factory=lambda: ExpectedModel(
            model_key="google/gemma-4-26b-a4b-qat",
            architecture="gemma4",
            format="mlx",
            quantization_name="4bit",
            bits_per_weight=4,
            loaded_context=40960,
        )
    )
    context: ContextEnvelope = Field(default_factory=ContextEnvelope)
    api_token: str | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def require_matching_context(self) -> RuntimeSettings:
        if self.model.loaded_context != self.context.hard_context:
            raise ValueError("model loaded context must equal the context envelope")
        return self


_ENV_PATHS: dict[str, tuple[str, ...]] = {
    "NOTES_RUNTIME_PROVIDER": ("provider",),
    "NOTES_RUNTIME_INFERENCE_BASE_URL": ("inference_base_url",),
    "NOTES_RUNTIME_CONTROL_BASE_URL": ("control_base_url",),
    "NOTES_RUNTIME_CONTROL_TIMEOUT_SECONDS": ("control_timeout_seconds",),
    "NOTES_RUNTIME_INFERENCE_TIMEOUT_SECONDS": ("inference_timeout_seconds",),
    "NOTES_RUNTIME_PROFILES_PATH": ("profiles_path",),
    "NOTES_RUNTIME_MODEL_KEY": ("model", "model_key"),
    "NOTES_RUNTIME_MODEL_INSTANCE_ID": ("model", "instance_id"),
    "NOTES_RUNTIME_MODEL_ARCHITECTURE": ("model", "architecture"),
    "NOTES_RUNTIME_MODEL_FORMAT": ("model", "format"),
    "NOTES_RUNTIME_MODEL_QUANTIZATION_NAME": ("model", "quantization_name"),
    "NOTES_RUNTIME_MODEL_BITS_PER_WEIGHT": ("model", "bits_per_weight"),
    "NOTES_RUNTIME_MODEL_LOADED_CONTEXT": ("model", "loaded_context"),
    "NOTES_RUNTIME_CONTEXT_HARD_CONTEXT": ("context", "hard_context"),
    "NOTES_RUNTIME_CONTEXT_INITIAL_PROMPT_LIMIT": ("context", "initial_prompt_limit"),
    "NOTES_RUNTIME_CONTEXT_GENERATION_LIMIT": ("context", "generation_limit"),
    "NOTES_RUNTIME_CONTEXT_TOOL_RESULT_LIMIT": ("context", "tool_result_limit"),
    "NOTES_RUNTIME_CONTEXT_SAFETY_MARGIN": ("context", "safety_margin"),
    "NOTES_RUNTIME_API_TOKEN": ("api_token",),
}


def load_runtime_settings(
    json_path: Path | None = None,
    *,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> RuntimeSettings:
    process_environment = dict(os.environ if environ is None else environ)
    selected_env_file = env_file
    if selected_env_file is None and process_environment.get("NOTES_RUNTIME_ENV_FILE"):
        selected_env_file = Path(process_environment["NOTES_RUNTIME_ENV_FILE"])
    environment: dict[str, str] = {}
    if selected_env_file is not None:
        parsed = dotenv_values(selected_env_file.expanduser().resolve())
        environment.update(
            {key: value for key, value in parsed.items() if isinstance(value, str)}
        )
    environment.update(process_environment)
    selected_path = json_path or Path(
        environment.get("NOTES_RUNTIME_CONFIG_FILE", str(DEFAULT_CONFIG_PATH))
    )
    selected_path = selected_path.expanduser().resolve()
    try:
        payload = json.loads(selected_path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeConfigurationError(
            "configuration_not_found", "runtime configuration file was not found"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeConfigurationError(
            "invalid_configuration", "runtime configuration file is invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeConfigurationError(
            "invalid_configuration", "runtime configuration must be a JSON object"
        )
    if "api_token" in payload:
        raise RuntimeConfigurationError(
            "invalid_configuration", "API tokens are environment-only"
        )

    values = copy.deepcopy(payload)
    for variable, field_path in _ENV_PATHS.items():
        if variable in environment:
            _set_nested(values, field_path, environment[variable])
    if values.get("provider", "lm_studio_openai") != "lm_studio_openai":
        raise RuntimeConfigurationError(
            "unsupported_provider", "configured runtime provider is not supported"
        )
    profiles_path = Path(str(values.get("profiles_path", "profiles.json"))).expanduser()
    if not profiles_path.is_absolute():
        profiles_path = selected_path.parent / profiles_path
    values["profiles_path"] = profiles_path.resolve()
    try:
        return RuntimeSettings.model_validate(values)
    except ValidationError as exc:
        message = "runtime configuration failed validation"
        if "model loaded context must equal" in str(exc):
            message = "runtime model loaded context does not match context envelope"
        raise RuntimeConfigurationError(
            "invalid_configuration", message
        ) from exc


def _set_nested(values: dict[str, Any], path: tuple[str, ...], value: str) -> None:
    target = values
    for part in path[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[path[-1]] = value
