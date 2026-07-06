# syntax=docker/dockerfile:1
FROM python:3.12-slim AS runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# ffmpeg -> ffprobe (audio duration probe)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-group test

COPY . .

# Infra paths only: connection settings (ASR / LM Studio) are managed at runtime
# from the frontend Settings page and persisted under CONFIG_DIR on the volume.
# The single /data volume is split into subdirs (db/, uploads/, config/) — the
# backend migrates any pre-existing flat files into them on startup.
ENV PATH="/app/.venv/bin:$PATH" \
    DATABASE_URL="sqlite:////data/db/notes.db" \
    UPLOAD_DIR="/data/uploads" \
    CONFIG_DIR="/data/config"

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
