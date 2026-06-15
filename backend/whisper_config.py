import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "whisper_config.json")

WHISPER_MODELS = [
    {"name": "tiny", "size": "~75 MB", "speed": "~10x realtime", "quality": "Lower"},
    {"name": "base", "size": "~142 MB", "speed": "~7x realtime", "quality": "Good"},
    {"name": "small", "size": "~466 MB", "speed": "~4x realtime", "quality": "Better"},
    {"name": "medium", "size": "~1.5 GB", "speed": "~2x realtime", "quality": "High"},
    {"name": "large-v3", "size": "~2.9 GB", "speed": "~1x realtime", "quality": "Best"},
]


def load() -> dict:
    defaults = {
        "binary_path": os.getenv("WHISPER_BINARY_PATH", ""),
        "model": os.getenv("WHISPER_MODEL", "base"),
        "model_path": os.getenv("WHISPER_MODEL_PATH", ""),
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
