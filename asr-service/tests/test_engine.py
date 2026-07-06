from asr_service.engine import _annotation_from_output, _turns_from_annotation
from asr_service.types import SpeakerTurn


class _FakeSegment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeAnnotation:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        for seg, label in self._tracks:
            yield seg, None, label


def test_turns_from_annotation():
    ann = _FakeAnnotation([
        (_FakeSegment(0.0, 1.0), "SPEAKER_00"),
        (_FakeSegment(1.0, 2.5), "SPEAKER_01"),
    ])
    turns = _turns_from_annotation(ann)
    assert turns == [
        SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
        SpeakerTurn(1.0, 2.5, "SPEAKER_01"),
    ]


class _FakeDiarizeOutput:
    """Mimics pyannote.audio >=4's DiarizeOutput wrapper."""
    def __init__(self, annotation):
        self.speaker_diarization = annotation


def test_annotation_from_output_unwraps_pyannote4_wrapper():
    ann = _FakeAnnotation([(_FakeSegment(0.0, 1.0), "SPEAKER_00")])
    wrapped = _FakeDiarizeOutput(ann)
    assert _annotation_from_output(wrapped) is ann


def test_annotation_from_output_passes_through_bare_annotation():
    ann = _FakeAnnotation([(_FakeSegment(0.0, 1.0), "SPEAKER_00")])
    assert _annotation_from_output(ann) is ann
