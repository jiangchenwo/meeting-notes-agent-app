from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    language: str | None
    model_used: str
    segments: list[dict]
