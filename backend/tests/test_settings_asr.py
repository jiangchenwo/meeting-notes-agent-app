import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from routers import settings


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(settings.router)
    return TestClient(app)


def _patch_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real = httpx.Client

    def fake(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real(transport=transport)

    monkeypatch.setattr("routers.settings.httpx.Client", fake)


def test_asr_status_connected(client, monkeypatch):
    def handler(request):
        assert request.url.path.endswith("/health")
        return httpx.Response(200, json={"status": "ok", "models_loaded": True})

    _patch_client(monkeypatch, handler)
    r = client.get("/api/settings/asr/status")
    assert r.status_code == 200
    body = r.json()
    assert body["connected"] is True
    assert body["models_loaded"] is True
    assert body["base_url"]


def test_asr_status_unreachable(client, monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused")

    _patch_client(monkeypatch, handler)
    r = client.get("/api/settings/asr/status")
    assert r.status_code == 200
    body = r.json()
    assert body["connected"] is False
    assert body["models_loaded"] is False
