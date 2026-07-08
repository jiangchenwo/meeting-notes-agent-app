"""Template workflow_config write-path validation (Phase 6 customization)."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, get_db
from routers import domains


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    app = FastAPI()
    app.include_router(domains.router)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    engine.dispose()


VALID_CONFIG = json.dumps({
    "steps": ["Summarizer", "DecisionLogger"],
    "critique_steps": ["Summarizer"],
    "max_retries": 1,
})


def test_create_template_with_valid_workflow_config(client):
    resp = client.post("/api/templates", json={"name": "Custom", "workflow_config": VALID_CONFIG})
    assert resp.status_code == 201
    assert resp.json()["workflow_config"] == VALID_CONFIG


def test_create_template_rejects_unknown_agent(client):
    bad = json.dumps({"steps": ["Summarizer", "NotAnAgent"]})
    resp = client.post("/api/templates", json={"name": "Bad", "workflow_config": bad})
    assert resp.status_code == 422


def test_create_template_rejects_invalid_json(client):
    resp = client.post("/api/templates", json={"name": "Bad", "workflow_config": "{not json"})
    assert resp.status_code == 422


def test_create_template_rejects_critique_of_unrun_step(client):
    bad = json.dumps({"steps": ["Summarizer"], "critique_steps": ["DecisionLogger"]})
    resp = client.post("/api/templates", json={"name": "Bad", "workflow_config": bad})
    assert resp.status_code == 422


def test_update_template_validates_and_clears(client):
    tid = client.post("/api/templates", json={"name": "T"}).json()["id"]

    resp = client.patch(f"/api/templates/{tid}", json={"workflow_config": VALID_CONFIG})
    assert resp.status_code == 200
    assert resp.json()["workflow_config"] == VALID_CONFIG

    resp = client.patch(f"/api/templates/{tid}", json={"workflow_config": json.dumps({"steps": []})})
    assert resp.status_code == 422

    # Blank string clears the override
    resp = client.patch(f"/api/templates/{tid}", json={"workflow_config": "  "})
    assert resp.status_code == 200
    assert resp.json()["workflow_config"] is None
