import json
import httpx
import pytest
import routers.transcribe as tr
from models import NoteBlock, Transcription
from database import SessionLocal


@pytest.fixture
def note_with_audio(tmp_path):
    db = SessionLocal()
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFFfake")
    note = NoteBlock(display_name="n", audio_file_path=str(audio), status="pending")
    db.add(note)
    db.commit()
    nid = note.id
    db.close()
    yield nid


def test_run_transcription_stores_segments_and_diarized(note_with_audio, monkeypatch):
    def fake_call(audio_bytes, filename, **kwargs):
        assert kwargs["diarize"] is True
        return {
            "full_text": "hi there", "language": "en", "model_used": "m",
            "duration_ms": 3000, "diarized": True,
            "segments": [{"start": 0.0, "end": 3.0, "text": "hi there", "speaker": "Speaker 1"}],
        }
    monkeypatch.setattr(tr, "transcribe_via_asr", fake_call)

    tr._run_transcription(note_with_audio, diarize=True)

    db = SessionLocal()
    t = db.query(Transcription).filter_by(note_block_id=note_with_audio).first()
    note = db.get(NoteBlock, note_with_audio)
    assert note.status == "transcribed"
    assert t.diarized is True
    assert json.loads(t.segments_json)[0]["speaker"] == "Speaker 1"
    db.close()


def test_run_transcription_marks_error_when_asr_unreachable(note_with_audio, monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("refused", request=None)
    monkeypatch.setattr(tr, "transcribe_via_asr", boom)

    tr._run_transcription(note_with_audio, diarize=False)

    db = SessionLocal()
    note = db.get(NoteBlock, note_with_audio)
    assert note.status == "error"
    db.close()
