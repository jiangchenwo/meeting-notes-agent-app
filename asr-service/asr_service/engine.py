import contextlib
import os
import subprocess
import tempfile
from typing import Protocol

from asr_service.config import Settings
from asr_service.types import SpeakerTurn, TranscriptResult


def _turns_from_annotation(annotation) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        turns.append(SpeakerTurn(float(segment.start), float(segment.end), str(label)))
    return turns


def _annotation_from_output(output):
    """Unwrap a pyannote diarization result to its Annotation.

    pyannote.audio >=4 returns a ``DiarizeOutput`` whose ``speaker_diarization``
    holds the Annotation; <4 returns the Annotation directly.
    """
    return getattr(output, "speaker_diarization", output)


@contextlib.contextmanager
def _pcm_wav(audio_path: str):
    """Decode audio to a 16 kHz mono PCM WAV for diarization.

    pyannote computes the expected sample count from a file's declared
    duration, but compressed formats (e.g. mp3) decode to a slightly
    different length, which raises in ``Audio.crop``. Feeding it exact PCM
    avoids that mismatch. Requires ffmpeg (already needed by mlx-whisper).
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", wav_path],
            check=True, capture_output=True,
        )
        yield wav_path
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


class Engine(Protocol):
    def transcribe(self, audio_path: str, language: str | None) -> TranscriptResult: ...
    def diarize(self, audio_path: str, min_speakers: int | None,
                max_speakers: int | None) -> list[SpeakerTurn]: ...


class MacMetalEngine:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._pipeline = None  # lazy

    def transcribe(self, audio_path: str, language: str | None) -> TranscriptResult:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self._settings.model_repo,
            language=language,
        )
        segments = [
            {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
            for s in result.get("segments", [])
            if s.get("text", "").strip()
        ]
        return TranscriptResult(
            text=result.get("text", "").strip(),
            language=result.get("language"),
            model_used=self._settings.model_repo,
            segments=segments,
        )

    def _get_pipeline(self):
        if self._pipeline is None:
            import torch
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            pipeline.to(torch.device(self._settings.device))
            self._pipeline = pipeline
        return self._pipeline

    def diarize(self, audio_path: str, min_speakers: int | None,
                max_speakers: int | None) -> list[SpeakerTurn]:
        kwargs = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
        with _pcm_wav(audio_path) as wav_path:
            output = self._get_pipeline()(wav_path, **kwargs)
        return _turns_from_annotation(_annotation_from_output(output))
