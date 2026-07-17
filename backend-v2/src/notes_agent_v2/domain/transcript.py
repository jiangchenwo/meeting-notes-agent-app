from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Utterance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^u[0-9]{6}$")
    speaker_id: str | None = None
    speaker_name: str | None = None
    text: str = Field(min_length=1)
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_time_range(self) -> Utterance:
        if not self.text.strip():
            raise ValueError("utterance text must not be blank")
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("utterance end must not precede start")
        return self


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^t[0-9]{6}$")
    note_id: str = Field(pattern=r"^n[0-9]{6}$")
    utterances: tuple[Utterance, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def stable_utterance_order(self) -> Transcript:
        expected = tuple(f"u{index:06d}" for index in range(1, len(self.utterances) + 1))
        actual = tuple(item.id for item in self.utterances)
        if actual != expected:
            raise ValueError("utterance IDs must be unique and monotonically increasing from u000001")
        return self
