from config_store import load_json, save_json

_CONFIG_NAME = "asr_config.json"

# Default targets the Docker deployment, where the host-native ASR service is
# reached via host.docker.internal. Editable from the frontend Settings page and
# persisted via config_store.
DEFAULTS = {
    "base_url": "http://host.docker.internal:9000",
}


def load() -> dict:
    return load_json(_CONFIG_NAME, DEFAULTS)


def save(data: dict) -> None:
    save_json(_CONFIG_NAME, data)
