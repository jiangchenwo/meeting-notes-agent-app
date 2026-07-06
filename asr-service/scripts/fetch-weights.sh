#!/usr/bin/env bash
# Run ONCE by the distributor (needs an HF token with pyannote license accepted).
# Downloads the pyannote diarization + MLX whisper models into the local HF cache
# so end users can run fully offline (HF_HUB_OFFLINE=1).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${HF_TOKEN:?Set HF_TOKEN (accept the pyannote/speaker-diarization-3.1 license first)}"
export HF_HUB_OFFLINE=0

uv run python - <<'PY'
import os, torch
from pyannote.audio import Pipeline
import mlx_whisper

token = os.environ["HF_TOKEN"]
Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
# Prime the MLX whisper model download.
model = os.getenv("ASR_MODEL_REPO", "mlx-community/whisper-large-v3-turbo")
mlx_whisper.load_models.load_model(model)
print("Models cached to the local HF cache.")
PY
