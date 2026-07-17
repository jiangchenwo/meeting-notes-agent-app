from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from notes_agent_v2.app import create_app
from notes_agent_v2.domain.transcript import Transcript, Utterance
from notes_agent_v2.persistence.database import Database, upgrade_database
from notes_agent_v2.persistence.repositories import Repositories


@pytest.fixture
def repositories(tmp_path: Path) -> Repositories:
    url = f"sqlite:///{tmp_path / 'notes.db'}"
    upgrade_database(url)
    return Repositories(Database(url))


@pytest.fixture
def client(repositories: Repositories) -> TestClient:
    return TestClient(create_app(repositories=repositories))


def test_instruction_preset_crud_and_tag_search(client: TestClient) -> None:
    created = client.post("/api/v2/presets", json={
        "name": "Executive update",
        "description": "Short status-oriented notes.",
        "instruction": "Summarize decisions, owners, and risks for executives.",
        "tags": ["executive", "status"],
    })
    assert created.status_code == 201
    preset = created.json()
    assert preset["id"] == "p000001"
    assert set(preset) == {"id", "name", "description", "instruction", "tags"}
    assert client.get("/api/v2/presets/p000001").json() == preset
    assert client.get("/api/v2/presets", params={"tag": "executive"}).json() == [preset]
    assert client.get("/api/v2/presets", params={"tag": "engineering"}).json() == []

    updated = client.patch("/api/v2/presets/p000001", json={
        "description": "Updated description.",
        "instruction": "Write a brief executive decision and risk update.",
    })
    assert updated.status_code == 200
    assert updated.json()["instruction"] == "Write a brief executive decision and risk update."
    assert client.delete("/api/v2/presets/p000001").status_code == 204
    assert client.get("/api/v2/presets/p000001").status_code == 404


@pytest.mark.parametrize("field,value", [
    ("domain_id", "finance"),
    ("workflow_config", {"stages": ["writer"]}),
    ("stage_prompts", {"writer": "do this"}),
    ("model", "another-model"),
    ("profile", "reasoned"),
    ("tools", ["search"]),
    ("retry", 5),
])
def test_preset_rejects_execution_routing_fields(client: TestClient, field: str, value: object) -> None:
    payload = {
        "name": "Invalid",
        "description": "Must fail.",
        "instruction": "Write notes.",
        "tags": [],
        field: value,
    }
    response = client.post("/api/v2/presets", json=payload)
    assert response.status_code == 422


def test_preset_rejects_empty_instruction(client: TestClient) -> None:
    response = client.post("/api/v2/presets", json={
        "name": "Empty", "description": "Invalid", "instruction": "   ", "tags": [],
    })
    assert response.status_code == 422
    created = client.post("/api/v2/presets", json={
        "name": "Valid", "description": "Valid", "instruction": "Write notes.", "tags": [],
    })
    assert client.patch(f"/api/v2/presets/{created.json()['id']}", json={"instruction": None}).status_code == 422


def test_run_keeps_preset_instruction_snapshot_after_edit_and_tombstone(
    client: TestClient,
    repositories: Repositories,
) -> None:
    client.post("/api/v2/presets", json={
        "name": "Decision notes", "description": "Decision focus",
        "instruction": "Record every approved decision.", "tags": ["decision"],
    })
    repositories.notes.create("n000001", "Meeting")
    repositories.transcripts.put(Transcript(
        id="t000001", note_id="n000001",
        utterances=(Utterance(id="u000001", text="Approved.", speaker_id=None, speaker_name=None, start_ms=None, end_ms=None),),
    ))
    run = repositories.runs.create_from_preset(
        "r000001", note_id="n000001", transcript_id="t000001", preset_id="p000001",
        project_context_ids=(), idempotency_key="run-one",
    )
    client.patch("/api/v2/presets/p000001", json={"instruction": "Write a narrative."})
    client.delete("/api/v2/presets/p000001")
    assert repositories.runs.get(run.id).instruction == "Record every approved decision."
