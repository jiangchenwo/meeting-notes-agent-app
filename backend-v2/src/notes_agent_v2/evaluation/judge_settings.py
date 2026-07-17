from __future__ import annotations

from collections.abc import Mapping
import copy
import json
import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, ValidationError, model_validator


DEFAULT_JUDGE_CONFIG = Path(__file__).resolve().parents[3] / "config" / "evaluation" / "judge.json"


class JudgeConfigurationError(RuntimeError):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class JudgeSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["judge-settings-v1"] = "judge-settings-v1"
    provider: Literal["disabled", "openai_compatible"] = "disabled"
    model: str | None = None
    base_url: AnyHttpUrl | None = None
    timeout_seconds: float = Field(default=30, gt=0, le=600)
    min_interval_seconds: float = Field(default=0, ge=0, le=60)
    max_cost_usd: float = Field(default=0, ge=0)
    input_cost_per_million: float = Field(default=0, ge=0)
    output_cost_per_million: float = Field(default=0, ge=0)
    temperature: float = Field(default=0, ge=0, le=2)
    rubric: str = Field(default="issues-v1", min_length=1)
    api_token: str | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def remote_is_complete(self) -> JudgeSettings:
        if self.provider != "disabled" and (not self.model or self.base_url is None or not self.api_token or self.max_cost_usd <= 0):
            raise ValueError("remote judge requires model, base URL, token, and positive budget")
        return self


ENV_FIELDS = {
    "NOTES_EVAL_JUDGE_PROVIDER": "provider", "NOTES_EVAL_JUDGE_MODEL": "model",
    "NOTES_EVAL_JUDGE_BASE_URL": "base_url", "NOTES_EVAL_JUDGE_TIMEOUT_SECONDS": "timeout_seconds",
    "NOTES_EVAL_JUDGE_MAX_COST_USD": "max_cost_usd", "NOTES_EVAL_JUDGE_API_TOKEN": "api_token",
    "NOTES_EVAL_JUDGE_INPUT_COST_PER_MILLION": "input_cost_per_million",
    "NOTES_EVAL_JUDGE_OUTPUT_COST_PER_MILLION": "output_cost_per_million",
    "NOTES_EVAL_JUDGE_MIN_INTERVAL_SECONDS": "min_interval_seconds",
}


def load_judge_settings(json_path: Path | None = None, *, env_file: Path | None = None, environ: Mapping[str, str] | None = None) -> JudgeSettings:
    process = dict(os.environ if environ is None else environ)
    selected_env = env_file or (Path(process["NOTES_EVAL_JUDGE_ENV_FILE"]) if process.get("NOTES_EVAL_JUDGE_ENV_FILE") else None)
    environment = {key: value for key, value in (dotenv_values(selected_env).items() if selected_env else []) if isinstance(value, str)}
    environment.update(process)
    path = (json_path or Path(environment.get("NOTES_EVAL_JUDGE_CONFIG_FILE", DEFAULT_JUDGE_CONFIG))).expanduser().resolve()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise JudgeConfigurationError("invalid_judge_configuration", "judge configuration is missing or invalid") from exc
    if not isinstance(payload, dict):
        raise JudgeConfigurationError("invalid_judge_configuration", "judge configuration must be an object")
    if "api_token" in payload:
        raise JudgeConfigurationError("invalid_judge_configuration", "judge API tokens are environment-only")
    values = copy.deepcopy(payload)
    for variable, field in ENV_FIELDS.items():
        if variable in environment:
            values[field] = environment[variable]
    if values.get("provider", "disabled") not in {"disabled", "openai_compatible"}:
        raise JudgeConfigurationError("unsupported_judge_provider", "configured judge provider is unsupported")
    try:
        return JudgeSettings.model_validate(values)
    except ValidationError as exc:
        message = "judge configuration failed validation"
        if "remote judge requires" in str(exc):
            message = "remote judge requires model, base URL, token, and positive budget"
        raise JudgeConfigurationError("invalid_judge_configuration", message) from exc
