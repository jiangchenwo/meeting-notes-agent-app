"""Model construction from runtime lm_config."""
from pydantic_ai.output import NativeOutput, PromptedOutput

from agents.llm import build_model, build_model_settings, wrap_output
from agents.outputs import SummaryOutput


def test_build_model_profile_fits_lm_studio(cfg):
    model = build_model(cfg)
    # LM Studio rejects response_format json_object; json_schema must stay on
    # for "native" mode.
    assert model.profile["supports_json_schema_output"] is True
    assert model.profile["supports_json_object_output"] is False
    # Default-mode consumers (LLMJudge) must never use tool-call output.
    assert model.profile["default_structured_output_mode"] == "native"
    assert model.model_name == "test-model"


def test_build_model_default_mode_follows_output_mode(cfg):
    cfg["output_mode"] = "prompted"
    assert build_model(cfg).profile["default_structured_output_mode"] == "prompted"


def test_build_model_defaults_model_id_when_unset(cfg):
    cfg["model"] = ""
    assert build_model(cfg).model_name == "local-model"


def test_build_model_settings_from_cfg(cfg):
    settings = build_model_settings(cfg)
    assert settings["max_tokens"] == 1024
    assert settings["temperature"] == 0.2


def test_wrap_output_mode_switch(cfg):
    assert isinstance(wrap_output(SummaryOutput, cfg), NativeOutput)  # default
    cfg["output_mode"] = "prompted"
    assert isinstance(wrap_output(SummaryOutput, cfg), PromptedOutput)
    cfg["output_mode"] = "native"
    assert isinstance(wrap_output(SummaryOutput, cfg), NativeOutput)
