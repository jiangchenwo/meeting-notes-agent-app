from fastapi.testclient import TestClient

from notes_agent_v2.app import create_app


def test_health_route_and_openapi_title() -> None:
    app = create_app()

    response = TestClient(app).get("/api/v2/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "notes-agent-v2"}
    assert app.openapi()["info"]["title"] == "Meeting Notes Agent V2"
