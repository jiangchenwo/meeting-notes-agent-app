import json
from transcript_format import build_speaker_transcript


def _segs(segments):
    return json.dumps(segments)


def test_groups_consecutive_same_speaker_into_one_line():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hi everyone.", "speaker": "Speaker 1"},
        {"start": 1.0, "end": 2.0, "text": "Thanks for joining.", "speaker": "Speaker 1"},
        {"start": 2.0, "end": 3.0, "text": "Glad to be here.", "speaker": "Speaker 2"},
    ]
    out = build_speaker_transcript("Hi everyone. Thanks for joining. Glad to be here.", _segs(segments))
    assert out == "Speaker 1: Hi everyone. Thanks for joining.\nSpeaker 2: Glad to be here."


def test_alternating_speakers_each_get_a_line():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Question?", "speaker": "Alice"},
        {"start": 1.0, "end": 2.0, "text": "Answer.", "speaker": "Bob"},
        {"start": 2.0, "end": 3.0, "text": "Follow up.", "speaker": "Alice"},
    ]
    out = build_speaker_transcript("x", _segs(segments))
    assert out == "Alice: Question?\nBob: Answer.\nAlice: Follow up."


def test_falls_back_to_full_text_when_no_speakers():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "no speaker here"},
        {"start": 1.0, "end": 2.0, "text": "still none"},
    ]
    assert build_speaker_transcript("no speaker here still none", _segs(segments)) == "no speaker here still none"


def test_empty_segments_returns_full_text():
    assert build_speaker_transcript("just the flat text", "[]") == "just the flat text"
    assert build_speaker_transcript(None, "[]") == ""


def test_skips_blank_segment_text():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Real line.", "speaker": "Speaker 1"},
        {"start": 1.0, "end": 2.0, "text": "   ", "speaker": "Speaker 1"},
    ]
    assert build_speaker_transcript("Real line.", _segs(segments)) == "Speaker 1: Real line."


def test_handles_malformed_segments_json():
    assert build_speaker_transcript("fallback", "not json") == "fallback"
