import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from database import Base, engine, SessionLocal
from models import Project
from routers import projects


@pytest.fixture
def client():
    # Ensure the (new) project_speakers table exists, like the real app startup does.
    Base.metadata.create_all(bind=engine)
    app = FastAPI()
    app.include_router(projects.router)
    return TestClient(app)


@pytest.fixture
def project_id():
    db = SessionLocal()
    p = Project(name="Speaker Test Project")
    db.add(p)
    db.commit()
    pid = p.id
    db.close()
    yield pid
    db = SessionLocal()
    proj = db.get(Project, pid)
    if proj:
        db.delete(proj)
        db.commit()
    db.close()


def test_create_list_rename_delete_speaker(client, project_id):
    # create
    r = client.post(f"/api/projects/{project_id}/speakers", json={"name": "Alice"})
    assert r.status_code == 201
    sid = r.json()["id"]
    assert r.json()["name"] == "Alice"

    # list
    r = client.get(f"/api/projects/{project_id}/speakers")
    assert r.status_code == 200
    assert any(s["id"] == sid for s in r.json())

    # rename
    r = client.patch(f"/api/projects/{project_id}/speakers/{sid}", json={"name": "Alice Cooper"})
    assert r.status_code == 200
    assert r.json()["name"] == "Alice Cooper"

    # delete
    r = client.delete(f"/api/projects/{project_id}/speakers/{sid}")
    assert r.status_code == 200
    r = client.get(f"/api/projects/{project_id}/speakers")
    assert all(s["id"] != sid for s in r.json())


def test_create_speaker_is_case_insensitive_deduped(client, project_id):
    r1 = client.post(f"/api/projects/{project_id}/speakers", json={"name": "Bob"})
    r2 = client.post(f"/api/projects/{project_id}/speakers", json={"name": "bob"})
    assert r1.json()["id"] == r2.json()["id"]


def test_create_speaker_unknown_project_404(client):
    r = client.post("/api/projects/99999999/speakers", json={"name": "Nobody"})
    assert r.status_code == 404
