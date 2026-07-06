"""Durable JSON config storage.

App configuration (LM Studio + ASR connection settings) is edited from the
frontend and must survive container recreation. In Docker, CONFIG_DIR points at
the mounted `/data` volume (see Dockerfile); for local, non-container
development it defaults to the backend directory.
"""
import json
import os

# Where editable config JSON files live. Set to /data in the container image so
# writes land on the persistent volume alongside notes.db and uploads.
CONFIG_DIR = os.getenv("CONFIG_DIR", os.path.dirname(os.path.abspath(__file__)))


def config_path(name: str) -> str:
    return os.path.join(CONFIG_DIR, name)


def load_json(name: str, defaults: dict) -> dict:
    """Return defaults merged with any saved values for `name`."""
    merged = dict(defaults)
    path = config_path(name)
    if os.path.isfile(path):
        try:
            with open(path) as f:
                merged.update(json.load(f))
        except Exception:
            pass
    return merged


def save_json(name: str, data: dict) -> None:
    """Persist `data` for `name`, writing atomically to avoid partial files."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    path = config_path(name)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
