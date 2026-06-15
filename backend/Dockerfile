# syntax=docker/dockerfile:1

# ---------- Stage 1: build whisper.cpp (CPU) + fetch a model ----------
FROM debian:bookworm-slim AS whisper
ARG WHISPER_CPP_VERSION=v1.7.4
ARG WHISPER_MODEL=base
# OFF = portable across all CPUs of an arch (no -march=native, avoids SIGILL on
# older/different CPUs). Set ON for a faster build tuned to the build machine.
ARG GGML_NATIVE=OFF
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch ${WHISPER_CPP_VERSION} \
        https://github.com/ggerganov/whisper.cpp .
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=${GGML_NATIVE} \
    && cmake --build build -j --config Release
RUN sh ./models/download-ggml-model.sh ${WHISPER_MODEL}
# Collect just the runtime artifacts (binary + shared libs + model).
RUN mkdir -p /opt/whisper/bin /opt/whisper/models \
    && cp build/bin/whisper-cli /opt/whisper/bin/ \
    && find build -name '*.so*' -exec cp {} /opt/whisper/bin/ \; \
    && cp models/ggml-${WHISPER_MODEL}.bin /opt/whisper/models/

# ---------- Stage 2: backend runtime ----------
FROM python:3.12-slim AS runtime
ARG WHISPER_MODEL=base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# ffmpeg -> ffprobe (audio duration probe); libgomp1 -> whisper.cpp OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Dependency layer (cached): install prod deps only, skip the heavy test group.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-group test

# App source. eval/, tests/, and the Testing Lab are excluded via .dockerignore.
COPY . .

# whisper.cpp binary + model from the builder stage.
COPY --from=whisper /opt/whisper /opt/whisper

ENV PATH="/app/.venv/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/whisper/bin" \
    ENABLE_LAB="false" \
    WHISPER_BINARY_PATH="/opt/whisper/bin" \
    WHISPER_MODEL="${WHISPER_MODEL}" \
    WHISPER_MODEL_PATH="/opt/whisper/models/ggml-${WHISPER_MODEL}.bin" \
    DATABASE_URL="sqlite:////data/notes.db" \
    UPLOAD_DIR="/data/uploads"

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
