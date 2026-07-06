# asr-service

Host-native ASR + speaker diarization for Meeting Notes Agent. Runs MLX-Whisper
(Metal) and pyannote.audio (MPS) and exposes an HTTP API the app calls at the
service URL set on the app's **Settings** page (default
`http://host.docker.internal:9000`). Cannot be containerized on Mac — Docker has
no Metal access.

## Run (end user)

    uv sync
    ./start.sh            # serves on :9000

Models are expected to be pre-cached; the server runs with `HF_HUB_OFFLINE=1`.

## Prepare models (distributor, one-time)

    HF_TOKEN=hf_xxx ./scripts/fetch-weights.sh

Requires a HuggingFace token that has accepted the
`pyannote/speaker-diarization-3.1` license.

## Config (env)

| Var | Default |
|-----|---------|
| `ASR_MODEL_REPO` | `mlx-community/whisper-large-v3-turbo` |
| `ASR_DEVICE` | `mps` |
| `ASR_PORT` | `9000` |
| `HF_HUB_OFFLINE` | `1` |
