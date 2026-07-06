import io
from fastapi.testclient import TestClient
from asr_service.app import app, get_engine
from asr_service.types import TranscriptResult, SpeakerTurn


class FakeEngine:
    def __init__(self, turns=None, fail_diarize=False):
        self._turns = turns or []
        self._fail = fail_diarize

    def transcribe(self, audio_path, language):
        return TranscriptResult(
            text="hello world", language="en", model_used="fake",
            segments=[{"start": 0.0, "end": 2.0, "text": "hello world"}],
        )

    def diarize(self, audio_path, min_speakers, max_speakers):
        if self._fail:
            raise RuntimeError("diarization boom")
        return self._turns


def _client(engine):
    app.dependency_overrides[get_engine] = lambda: engine
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_health():
    c = _client(FakeEngine())
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_transcribe_no_diarize():
    c = _client(FakeEngine())
    r = c.post("/transcribe", files={"audio_file": ("a.wav", io.BytesIO(b"x"))},
               data={"diarize": "false"})
    body = r.json()
    assert body["full_text"] == "hello world"
    assert body["diarized"] is False
    assert body["segments"][0].get("speaker") is None
    assert body["duration_ms"] == 2000


def test_transcribe_with_diarize():
    engine = FakeEngine(turns=[SpeakerTurn(0.0, 2.0, "SPEAKER_00")])
    c = _client(engine)
    r = c.post("/transcribe", files={"audio_file": ("a.wav", io.BytesIO(b"x"))},
               data={"diarize": "true"})
    body = r.json()
    assert body["diarized"] is True
    assert body["segments"][0]["speaker"] == "Speaker 1"


def test_diarize_failure_degrades():
    engine = FakeEngine(fail_diarize=True)
    c = _client(engine)
    r = c.post("/transcribe", files={"audio_file": ("a.wav", io.BytesIO(b"x"))},
               data={"diarize": "true"})
    body = r.json()
    assert r.status_code == 200
    assert body["diarized"] is False
    assert body["full_text"] == "hello world"
