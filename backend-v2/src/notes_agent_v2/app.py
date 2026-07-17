from fastapi import FastAPI

from .api.presets import create_preset_router
from .persistence.repositories import Repositories


def create_app(*, repositories: Repositories | None = None) -> FastAPI:
    app = FastAPI(title="Meeting Notes Agent V2")

    @app.get("/api/v2/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "notes-agent-v2"}

    app.include_router(create_preset_router(repositories))
    return app
