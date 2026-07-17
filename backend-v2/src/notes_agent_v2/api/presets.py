from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from notes_agent_v2.persistence.repositories import PromptPreset, Repositories


class PresetCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    tags: tuple[str, ...] = ()

    @field_validator("name", "description", "instruction")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class PresetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1)
    description: str | None = Field(default=None, min_length=1)
    instruction: str | None = Field(default=None, min_length=1)
    tags: tuple[str, ...] | None = None

    @field_validator("name", "description", "instruction")
    @classmethod
    def not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("value must not be blank")
        return value

    @model_validator(mode="after")
    def explicit_null_is_invalid(self) -> PresetUpdate:
        if any(name in self.model_fields_set and getattr(self, name) is None for name in type(self).model_fields):
            raise ValueError("preset updates cannot set fields to null")
        return self


def create_preset_router(repositories: Repositories | None) -> APIRouter:
    router = APIRouter(prefix="/api/v2/presets", tags=["presets"])

    def require_repositories() -> Repositories:
        if repositories is None:
            raise HTTPException(status_code=503, detail="persistence is not configured")
        return repositories

    @router.api_route("", methods=["POST"], response_model=PromptPreset, status_code=status.HTTP_201_CREATED)
    def create_preset(payload: PresetCreate) -> PromptPreset:
        try:
            return require_repositories().presets.create(**payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("", response_model=list[PromptPreset])
    def list_presets(tag: str | None = Query(default=None)) -> tuple[PromptPreset, ...]:
        return require_repositories().presets.list(tag=tag)

    @router.get("/{preset_id}", response_model=PromptPreset)
    def get_preset(preset_id: str) -> PromptPreset:
        try:
            return require_repositories().presets.get(preset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="preset not found") from exc

    @router.patch("/{preset_id}", response_model=PromptPreset)
    def update_preset(preset_id: str, payload: PresetUpdate) -> PromptPreset:
        try:
            return require_repositories().presets.update(preset_id, **payload.model_dump(exclude_unset=True))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="preset not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/{preset_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_preset(preset_id: str) -> Response:
        try:
            require_repositories().presets.tombstone(preset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="preset not found") from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
