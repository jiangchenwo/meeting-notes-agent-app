"""End-to-end run_workflow test: in-memory SQLite + scripted FunctionModel.

Verifies the full persistence contract — WorkflowRun, WorkflowStepResult rows
(including token columns), and the Summary — without touching the dev DB or
any LLM endpoint.
"""
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import lm_config
import agents.orchestrator as orch
from database import Base
from models import NoteBlock, Summary, Transcription, WorkflowRun, WorkflowStepResult
from tests.conftest import MOCK_CFG
from tests.test_pipeline import happy_responder, scripted_model


@pytest.fixture
def db_session(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(orch, "SessionLocal", TestSession)
    monkeypatch.setattr(lm_config, "load", lambda: MOCK_CFG.copy())
    yield TestSession
    engine.dispose()


@pytest.fixture
def note_id(db_session):
    db = db_session()
    note = NoteBlock(display_name="standup", status="transcribed")
    db.add(note)
    db.commit()
    db.add(Transcription(
        note_block_id=note.id,
        full_text="Alice: We approved the $50k budget. Bob: I will send the report.",
    ))
    db.commit()
    nid = note.id
    db.close()
    return nid


def test_run_workflow_persists_run_steps_and_summary(db_session, note_id):
    with scripted_model(happy_responder):
        orch.run_workflow(note_id)

    db = db_session()
    note = db.get(NoteBlock, note_id)
    assert note.status == "done"

    run = db.query(WorkflowRun).filter_by(note_block_id=note_id).one()
    assert run.status == "done"
    assert run.finished_at is not None
    assert run.error_message is None
    assert run.total_input_tokens > 0 and run.total_output_tokens > 0
    assert run.model_name  # from the model response
    plan = json.loads(run.workflow_plan_json)
    # No domain on the note → legacy default "Project" workflow
    assert [s["agent"] for s in plan["steps"]] == [
        "Summarizer", "ActionItemExtractor", "DecisionLogger"
    ]

    steps = db.query(WorkflowStepResult).filter_by(run_id=run.id).all()
    assert [s.step_name for s in steps] == [
        "Summarizer", "ActionItemExtractor", "DecisionLogger", "Critic:Summarizer"
    ]
    assert all(s.status == "done" for s in steps)
    assert all(s.input_tokens > 0 and s.output_tokens > 0 for s in steps)
    assert all(s.model_name for s in steps)
    critic_row = steps[-1]
    assert critic_row.critique_score == 9.0
    summarizer_result = json.loads(steps[0].result_json)
    assert "_prompt" in summarizer_result  # composed prompt persisted for debugging

    summary = db.query(Summary).filter_by(note_block_id=note_id).one()
    assert "Budget of $50k" in summary.summary_text
    assert json.loads(summary.action_items_json)[0]["owner"] == "Alice"
    assert summary.confidence_score == 9.0
    assert summary.workflow_run_id == run.id
    assert summary.llm_model_used == run.model_name
    raw = json.loads(summary.raw_sections_json)
    assert "_schema_checks" in raw and "_risk_classification" in raw
    assert "## Decisions Made" in summary.suggestions_text  # Project-domain assembly
    db.close()


def test_run_workflow_rerun_updates_existing_summary(db_session, note_id):
    with scripted_model(happy_responder):
        orch.run_workflow(note_id)
        orch.run_workflow(note_id)

    db = db_session()
    assert db.query(Summary).filter_by(note_block_id=note_id).count() == 1
    assert db.query(WorkflowRun).filter_by(note_block_id=note_id).count() == 2
    summary = db.query(Summary).filter_by(note_block_id=note_id).one()
    latest_run = db.query(WorkflowRun).order_by(WorkflowRun.id.desc()).first()
    assert summary.workflow_run_id == latest_run.id
    db.close()


def test_run_workflow_marks_error_when_llm_unreachable(db_session, note_id):
    def dead_endpoint(messages, info):
        raise RuntimeError("connection refused")

    with scripted_model(dead_endpoint):
        orch.run_workflow(note_id)

    db = db_session()
    note = db.get(NoteBlock, note_id)
    assert note.status == "error"
    run = db.query(WorkflowRun).filter_by(note_block_id=note_id).one()
    assert run.status == "error"
    assert "all workflow steps failed" in run.error_message
    assert db.query(Summary).filter_by(note_block_id=note_id).count() == 0
    db.close()


def test_run_workflow_without_transcription_sets_error(db_session):
    db = db_session()
    note = NoteBlock(display_name="empty", status="pending")
    db.add(note)
    db.commit()
    nid = note.id
    db.close()

    orch.run_workflow(nid)

    db = db_session()
    assert db.get(NoteBlock, nid).status == "error"
    assert db.query(WorkflowRun).filter_by(note_block_id=nid).count() == 0
    db.close()
