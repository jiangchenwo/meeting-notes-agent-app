# Meeting Notes Agent — image build & publish.
# Local development uses `docker compose` directly; these targets add a portable,
# multi-architecture build for distributing the app to other machines.

REGISTRY      ?=
TAG           ?= latest
PLATFORMS     ?= linux/amd64,linux/arm64

BACKEND_IMAGE  ?= $(if $(REGISTRY),$(REGISTRY)/,)meeting-notes-agent-backend:$(TAG)
FRONTEND_IMAGE ?= $(if $(REGISTRY),$(REGISTRY)/,)meeting-notes-agent-frontend:$(TAG)

# Exported so `docker compose` (build/pull/up below) uses the same image refs.
export BACKEND_IMAGE
export FRONTEND_IMAGE

.PHONY: help build up down logs pull buildx buildx-backend buildx-frontend

help:
	@echo "Local (current architecture):"
	@echo "  make build    Build both images via docker compose"
	@echo "  make up       Start the app at http://localhost:8080"
	@echo "  make down     Stop the app"
	@echo "  make logs     Tail logs"
	@echo ""
	@echo "Publish portable multi-arch images (needs a registry you can push to):"
	@echo "  make buildx REGISTRY=ghcr.io/you TAG=1.0"
	@echo ""
	@echo "Run prebuilt images on a target host:"
	@echo "  make pull up REGISTRY=ghcr.io/you TAG=1.0"
	@echo ""
	@echo "Note: transcription/diarization (asr-service) is host-native and not"
	@echo "covered by these targets — set it up separately on each host, see"
	@echo "asr-service/README.md."

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

pull:
	docker compose pull

# --- Portable multi-architecture publish ---
buildx: buildx-backend buildx-frontend

buildx-backend:
	@test -n "$(REGISTRY)" || { echo "Set REGISTRY, e.g. make buildx REGISTRY=ghcr.io/you"; exit 1; }
	docker buildx build --platform $(PLATFORMS) \
		-t $(BACKEND_IMAGE) ./backend --push

buildx-frontend:
	@test -n "$(REGISTRY)" || { echo "Set REGISTRY, e.g. make buildx REGISTRY=ghcr.io/you"; exit 1; }
	docker buildx build --platform $(PLATFORMS) \
		-t $(FRONTEND_IMAGE) ./frontend --push
