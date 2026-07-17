from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BundleError(RuntimeError):
    pass


class EvaluationBundleManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["evaluation-bundle-v1"] = "evaluation-bundle-v1"
    run_id: str = Field(min_length=1)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: dict[str, str]
    sealed: bool = True
    bundle_digest: str = ""

    @model_validator(mode="after")
    def derive_digest(self) -> EvaluationBundleManifest:
        payload = self.model_dump(mode="json", exclude={"bundle_digest"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "bundle_digest", hashlib.sha256(canonical.encode()).hexdigest())
        return self


class EvaluationBundleWriter:
    def __init__(self, target: Path, *, run_id: str, fingerprint: str) -> None:
        self.target = target
        self.run_id = run_id
        self.fingerprint = fingerprint
        self.staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
        if target.exists():
            raise BundleError("bundle target already exists")
        self.staging.mkdir(parents=True)
        self.sealed = False

    def _path(self, relative: str) -> Path:
        if self.sealed:
            raise BundleError("bundle is sealed")
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or relative == "manifest.json":
            raise BundleError("unsafe or reserved bundle path")
        target = self.staging / path
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_text(self, relative: str, value: str) -> None:
        target = self._path(relative)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(value)
        os.replace(temporary, target)

    def write_json(self, relative: str, value: object) -> None:
        self.write_text(relative, json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")

    def seal(self) -> EvaluationBundleManifest:
        files = {str(path.relative_to(self.staging)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(self.staging.rglob("*")) if path.is_file()}
        manifest = EvaluationBundleManifest(run_id=self.run_id, fingerprint=self.fingerprint, files=files)
        (self.staging / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n")
        os.replace(self.staging, self.target)
        self.sealed = True
        return manifest


def verify_bundle(path: Path) -> EvaluationBundleManifest:
    try:
        manifest = EvaluationBundleManifest.model_validate_json((path / "manifest.json").read_text())
    except Exception as exc:
        raise BundleError("bundle manifest is missing or invalid") from exc
    actual_files = {str(item.relative_to(path)) for item in path.rglob("*") if item.is_file() and item.name != "manifest.json"}
    if actual_files != set(manifest.files):
        raise BundleError("bundle file set does not match manifest")
    for relative, expected in manifest.files.items():
        if hashlib.sha256((path / relative).read_bytes()).hexdigest() != expected:
            raise BundleError("bundle file digest mismatch")
    return manifest


def cleanup_stale_temporary_bundles(parent: Path) -> int:
    removed = 0
    for path in parent.glob(".*.tmp-*"):
        if path.is_dir():
            shutil.rmtree(path)
            removed += 1
    return removed
