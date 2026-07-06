from asr_service.merge import assign_speakers
from asr_service.types import SpeakerTurn

def test_assigns_by_max_overlap():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "a"},
        {"start": 2.0, "end": 4.0, "text": "b"},
    ]
    turns = [
        SpeakerTurn(0.0, 2.1, "SPEAKER_00"),
        SpeakerTurn(2.1, 4.0, "SPEAKER_01"),
    ]
    out = assign_speakers(segments, turns)
    assert out[0]["speaker"] == "Speaker 1"
    assert out[1]["speaker"] == "Speaker 2"
    # original text preserved, input not mutated
    assert out[0]["text"] == "a"
    assert "speaker" not in segments[0]

def test_labels_follow_first_appearance_order():
    segments = [{"start": 5.0, "end": 6.0, "text": "x"}]
    turns = [
        SpeakerTurn(0.0, 1.0, "SPEAKER_09"),   # first appearance -> Speaker 1
        SpeakerTurn(5.0, 6.0, "SPEAKER_03"),   # second -> Speaker 2
    ]
    out = assign_speakers(segments, turns)
    assert out[0]["speaker"] == "Speaker 2"

def test_segment_with_no_overlap_gets_none():
    segments = [{"start": 10.0, "end": 11.0, "text": "z"}]
    turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]
    out = assign_speakers(segments, turns)
    assert out[0]["speaker"] is None

def test_empty_turns_all_none():
    segments = [{"start": 0.0, "end": 1.0, "text": "q"}]
    out = assign_speakers(segments, [])
    assert out[0]["speaker"] is None

def test_labels_sorted_by_start_not_list_order():
    # turns given OUT of chronological order: later-in-time turn listed first.
    segments = [
        {"start": 0.0, "end": 1.0, "text": "early"},
        {"start": 5.0, "end": 6.0, "text": "late"},
    ]
    turns = [
        SpeakerTurn(5.0, 6.0, "SPEAKER_LATE"),   # first in list, later in time
        SpeakerTurn(0.0, 1.0, "SPEAKER_EARLY"),  # second in list, earlier in time
    ]
    out = assign_speakers(segments, turns)
    # Labels follow start-time order: SPEAKER_EARLY (t=0) -> Speaker 1, SPEAKER_LATE (t=5) -> Speaker 2
    assert out[0]["speaker"] == "Speaker 1"  # early segment overlaps SPEAKER_EARLY
    assert out[1]["speaker"] == "Speaker 2"  # late segment overlaps SPEAKER_LATE
