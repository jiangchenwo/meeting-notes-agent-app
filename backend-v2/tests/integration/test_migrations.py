from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from notes_agent_v2.persistence.database import Database, upgrade_database


EXPECTED_TABLES = {
    "alembic_version",
    "notes",
    "transcripts",
    "utterances",
    "project_context_records",
    "prompt_presets",
    "generation_runs",
    "stage_artifacts",
    "facts",
    "documents",
    "critic_issues",
    "quality_reports",
    "run_events",
    "model_call_records",
}


def test_empty_sqlite_migrates_idempotently_with_required_tables(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'notes.db'}"
    upgrade_database(url)
    upgrade_database(url)
    database = Database(url)
    assert EXPECTED_TABLES <= set(inspect(database.engine).get_table_names())


def test_sqlite_enables_foreign_keys_wal_and_busy_timeout(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'notes.db'}"
    upgrade_database(url)
    database = Database(url)
    with database.engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        assert connection.execute(text("PRAGMA journal_mode")).scalar_one().lower() == "wal"
        assert connection.execute(text("PRAGMA busy_timeout")).scalar_one() >= 5000
    with pytest.raises(IntegrityError):
        with database.session() as session:
            session.execute(text("INSERT INTO transcripts (id, note_id, digest, payload_json) VALUES ('t000001', 'n999999', :digest, '{}')"), {"digest": "0" * 64})
