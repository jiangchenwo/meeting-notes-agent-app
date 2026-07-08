from config_store import load_json, save_json

_CONFIG_NAME = "lm_config.json"


DEFAULT_SYSTEM_PROMPT = (
    "You are a professional meeting notes assistant. "
    "Generate clear, structured notes from meeting transcripts.\n\n"
    "Format your output using Markdown:\n"
    "- Use **bold** for key decisions, names, and important terms\n"
    "- Use bullet lists for takeaways, items, and points\n"
    "- Use numbered lists for ordered steps or priorities\n"
    "- Use `##` headings to organize sections when appropriate\n"
    "- Use `code` formatting for technical terms, commands, or identifiers\n\n"
    "Keep the tone concise, professional, and action-oriented."
)

# Defaults target the Docker deployment, where LM Studio runs on the host and is
# reached via host.docker.internal. Everything here is editable from the
# frontend Settings page and persisted via config_store.
DEFAULTS = {
    "base_url": "http://host.docker.internal:1234/v1",
    "model": "",
    "max_tokens": 40960,
    "max_response_tokens": 2048,
    "global_system_prompt": DEFAULT_SYSTEM_PROMPT,
    # Structured-output mode for the agent pipeline: "native" (response_format
    # json_schema, grammar-enforced in LM Studio) or "prompted" (schema in the
    # prompt — fallback for engines that reject json_schema).
    "output_mode": "native",
}


def load() -> dict:
    return load_json(_CONFIG_NAME, DEFAULTS)


def save(data: dict) -> None:
    save_json(_CONFIG_NAME, data)
