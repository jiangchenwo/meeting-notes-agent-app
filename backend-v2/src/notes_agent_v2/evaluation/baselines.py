from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HistoricalResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    case_id: str = Field(min_length=1)
    output: str
    reason: str = Field(min_length=1)
    certified: bool = False

    @model_validator(mode="after")
    def never_certified(self) -> HistoricalResult:
        if self.certified:
            raise ValueError("historical results can never be certified")
        return self


class OneShotBaseline:
    def __init__(self, runtime_call: Callable[..., Any]) -> None:
        self.runtime_call = runtime_call

    def run(self, instruction: str, transcript: str) -> Any:
        return self.runtime_call(profile="narrative_reasoned", instruction=instruction, transcript=transcript)


class LegacyHttpBaseline:
    def __init__(self, endpoint: str, request: Callable[[str, dict[str, Any]], Any]) -> None:
        self.endpoint, self.request = endpoint, request

    def run(self, instruction: str, transcript: str) -> Any:
        return self.request(self.endpoint, {"instruction": instruction, "transcript": transcript})
