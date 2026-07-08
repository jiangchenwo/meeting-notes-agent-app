from config_store import load_json, save_json

_CONFIG_NAME = "telemetry_config.json"

# Off by default: local-first means no network chatter unless the user turns
# tracing on in Settings. Endpoint is a local Arize Phoenix instance
# (`uvx arize-phoenix serve`), which receives OTLP/HTTP at /v1/traces.
DEFAULTS = {
    "enabled": False,
    "endpoint": "http://localhost:6006",
    # Include prompts/completions in spans. Everything stays on-machine, but
    # this can be turned off for sensitive transcripts.
    "capture_content": True,
}


def load() -> dict:
    return load_json(_CONFIG_NAME, DEFAULTS)


def save(data: dict) -> None:
    save_json(_CONFIG_NAME, data)
