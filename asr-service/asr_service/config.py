import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_repo: str
    device: str
    hf_offline: bool
    port: int


def load_settings() -> Settings:
    return Settings(
        model_repo=os.getenv("ASR_MODEL_REPO", "mlx-community/whisper-large-v3-turbo"),
        device=os.getenv("ASR_DEVICE", "mps"),
        hf_offline=os.getenv("HF_HUB_OFFLINE", "1") == "1",
        port=int(os.getenv("ASR_PORT", "9000")),
    )
