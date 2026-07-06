#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASR_PORT="${ASR_PORT:-9000}"
uv run uvicorn asr_service.app:app --host 0.0.0.0 --port "$ASR_PORT"
