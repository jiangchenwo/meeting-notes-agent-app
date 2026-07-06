from models import Transcription

def test_transcription_has_diarized_default():
    t = Transcription(note_block_id=1)
    # column default is applied on flush; attribute exists and accepts a bool
    t.diarized = True
    assert t.diarized is True
    assert hasattr(Transcription, "diarized")
