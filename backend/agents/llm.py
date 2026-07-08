"""Per-run model construction from runtime lm_config.

Models are built from `lm_config.load()` at workflow start — never at import
time — so Settings-page changes apply to the next run without a restart.
"""
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import NativeOutput, PromptedOutput
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

DEFAULT_TEMPERATURE = 0.2
# Hang protection, not a performance knob: a single call on a ~40k-char
# meeting (11k-token prompt) legitimately takes 3+ minutes on modest local
# hardware. At 180s those calls timed out right at the boundary — and the
# client's 2 automatic retries turned each failure into 9 wasted minutes.
LLM_TIMEOUT_SECONDS = 480.0


def _local_endpoint_profile(cfg: dict):
    def apply(profile: dict) -> dict:
        return {
            **dict(profile),
            # LM Studio accepts response_format json_schema or text, but rejects
            # json_object — so "prompted" mode must send no response_format at all.
            "supports_json_object_output": False,
            # Consumers that don't wrap output explicitly (e.g. pydantic-evals
            # LLMJudge) must follow the configured mode, never tool calls —
            # small local models can't be trusted with tool calling.
            "default_structured_output_mode": (
                "prompted" if cfg.get("output_mode", "native") == "prompted" else "native"
            ),
        }

    return apply


def build_model(cfg: dict) -> OpenAIChatModel:
    return OpenAIChatModel(
        # LM Studio routes unknown model ids to the loaded model, so an unset
        # config still works with whatever is loaded.
        cfg.get("model") or "local-model",
        provider=OpenAIProvider(
            base_url=cfg["base_url"].rstrip("/"),
            api_key="lm-studio",  # LM Studio ignores the key; the client requires one
        ),
        profile=_local_endpoint_profile(cfg),
    )


def build_model_settings(cfg: dict, temperature: float = DEFAULT_TEMPERATURE) -> ModelSettings:
    return ModelSettings(
        temperature=temperature,
        max_tokens=int(cfg.get("max_response_tokens", 2048)),
        timeout=LLM_TIMEOUT_SECONDS,
    )


def wrap_output(output_model: type[BaseModel], cfg: dict) -> NativeOutput | PromptedOutput:
    """Structured-output mode for local models.

    "native" uses response_format=json_schema (grammar-enforced in LM Studio);
    "prompted" injects the schema into the prompt — the universal fallback for
    engines that reject json_schema. Never tool-call based: small local models
    can't be trusted with tool calling.
    """
    if cfg.get("output_mode", "native") == "prompted":
        return PromptedOutput(output_model)
    return NativeOutput(output_model)
