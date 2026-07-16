from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProfileError(RuntimeError):
    pass


class StageProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    reasoning: Literal["reasoned", "off"]
    output_mode: Literal["text", "structured"]
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(gt=0)
    max_tokens: int = Field(gt=0, le=8192)
    parse_retries: int = Field(ge=0, le=1)
    max_tool_rounds: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)
    max_tool_result_tokens: int = Field(ge=0, le=4096)
    status: Literal["candidate", "certified"]
    fingerprint: str = ""

    @model_validator(mode="after")
    def validate_policy(self) -> StageProfile:
        if self.output_mode == "structured" and self.reasoning != "off":
            raise ValueError("structured profiles must disable reasoning")
        payload = self.model_dump(mode="json", exclude={"fingerprint"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "fingerprint", hashlib.sha256(canonical.encode()).hexdigest())
        return self

    def provider_settings(self, *, force_structured_off: bool = False) -> dict[str, object]:
        settings: dict[str, object] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning == "off" or force_structured_off:
            settings["reasoning_effort"] = "none"
        return settings


class ProfileFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["runtime-profiles-v1"]
    profiles: tuple[StageProfile, ...]


class ProfileCatalog:
    def __init__(self, profile_file: ProfileFile) -> None:
        self.schema_version = profile_file.schema_version
        self._profiles = {profile.name: profile for profile in profile_file.profiles}
        if len(self._profiles) != len(profile_file.profiles):
            raise ProfileError("profile names must be unique")

    @classmethod
    def from_path(cls, path: Path) -> ProfileCatalog:
        return cls(ProfileFile.model_validate_json(path.read_text()))

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._profiles)

    def resolve(self, name: str, *, production: bool = False) -> StageProfile:
        try:
            profile = self._profiles[name]
        except KeyError as exc:
            raise ProfileError(f"unknown runtime profile: {name}") from exc
        if production and profile.status != "certified":
            raise ProfileError(f"profile {name} is not production eligible")
        return profile
