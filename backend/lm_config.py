import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "lm_config.json")


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


def load() -> dict:
    defaults = {
        "base_url": os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        "model": os.getenv("LM_STUDIO_MODEL", ""),
        "max_tokens": int(os.getenv("LM_STUDIO_MAX_TOKENS", "4096")),
        "max_response_tokens": int(os.getenv("LM_STUDIO_MAX_RESPONSE_TOKENS", "2048")),
        "global_system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
    if os.path.isfile(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH) as f:
                saved = json.load(f)
            defaults.update(saved)
        except Exception:
            pass
    return defaults


def save(data: dict) -> None:
    with open(_CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)
