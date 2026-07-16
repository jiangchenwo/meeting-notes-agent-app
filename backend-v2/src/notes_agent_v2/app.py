from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="Meeting Notes Agent V2")

    @app.get("/api/v2/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "notes-agent-v2"}

    return app
