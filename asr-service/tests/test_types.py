from asr_service.types import SpeakerTurn, TranscriptResult

def test_speaker_turn_fields():
    t = SpeakerTurn(start=1.0, end=2.0, speaker="SPEAKER_00")
    assert (t.start, t.end, t.speaker) == (1.0, 2.0, "SPEAKER_00")

def test_transcript_result_fields():
    r = TranscriptResult(text="hi", language="en", model_used="m", segments=[{"start": 0.0, "end": 1.0, "text": "hi"}])
    assert r.segments[0]["text"] == "hi"
    assert r.language == "en"
