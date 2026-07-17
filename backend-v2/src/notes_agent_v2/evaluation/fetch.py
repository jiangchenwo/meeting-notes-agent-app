from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import zipfile

import httpx


class FetchError(RuntimeError):
    pass


def fetch_archive(url: str, destination: Path, *, allow_network: bool, max_bytes: int = 2_000_000_000, timeout_seconds: float = 120, client: httpx.Client | None = None) -> str:
    if not allow_network:
        raise FetchError("benchmark network fetching is disabled")
    if destination.exists():
        raise FetchError("archive destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    size = 0
    try:
        transport = client or httpx.Client(timeout=timeout_seconds, follow_redirects=True)
        with transport.stream("GET", url) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise FetchError("benchmark download exceeded size cap")
                    digest.update(chunk)
                    handle.write(chunk)
        if size == 0:
            raise FetchError("benchmark download was empty")
        temporary.replace(destination)
        return digest.hexdigest()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def extract_archive(archive: Path, output: Path, checksum: str, *, max_bytes: int = 2_000_000_000) -> tuple[Path, ...]:
    if not archive.is_file() or archive.stat().st_size > max_bytes:
        raise FetchError("archive is missing or oversized")
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != checksum.lower():
        raise FetchError("archive checksum mismatch")
    output = output.resolve()
    if output.exists():
        raise FetchError("archive output already exists")
    extracted: list[Path] = []
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            _validate_members(members)
        output.mkdir(parents=True)
        try:
            with tarfile.open(archive) as handle:
                for member in members:
                    if member.isdir():
                        continue
                    target = output / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = handle.extractfile(member)
                    if source is None:
                        raise FetchError("archive member is unreadable")
                    with target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                    extracted.append(target)
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            raise
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            infos = handle.infolist()
            _validate_members(infos)
        output.mkdir(parents=True)
        try:
            with zipfile.ZipFile(archive) as handle:
                for info in infos:
                    if info.is_dir():
                        continue
                    target = output / info.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with handle.open(info) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                    extracted.append(target)
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            raise
    else:
        raise FetchError("unsupported archive format")
    if not extracted:
        shutil.rmtree(output, ignore_errors=True)
        raise FetchError("archive extraction was empty")
    return tuple(extracted)


def _validate_members(members: list[object]) -> None:
    for member in members:
        name = str(getattr(member, "name", getattr(member, "filename", "")))
        path = PurePosixPath(name)
        mode = int(getattr(member, "external_attr", 0)) >> 16
        is_link = bool(getattr(member, "issym", lambda: False)()) or bool(getattr(member, "islnk", lambda: False)())
        is_link = is_link or (mode & 0o170000) == 0o120000
        if not name or path.is_absolute() or ".." in path.parts or is_link:
            raise FetchError("unsafe archive member")
