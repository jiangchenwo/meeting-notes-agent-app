"""Data-storage layout for the persistent volume.

The Docker image keeps all durable state on one `/data` volume. Rather than
dumping the database, uploads, and config JSON side by side in the volume root
(the v0.1.0 layout), we split them into dedicated subdirectories:

    /data/db/notes.db        <- SQLite database (+ -wal / -shm)
    /data/uploads/           <- uploaded audio blobs (bulk binary)
    /data/config/*.json      <- runtime settings edited from the app

The paths are driven by the same env vars the code already reads
(`DATABASE_URL`, `UPLOAD_DIR`, `CONFIG_DIR`) — the Dockerfile points them at the
subdirs; unset (local dev) they keep their historical flat defaults, so this
module is a no-op outside the container.

`ensure_and_migrate()` runs once at startup *before* the DB is opened. It
creates the target directories and relocates any leftover flat files from the
old layout, so an existing volume upgrades in place without losing data.
"""
import os
import shutil


def _sqlite_file(database_url: str | None) -> str | None:
    """Filesystem path backing a `sqlite:///` URL, or None for other backends."""
    if database_url and database_url.startswith("sqlite:///"):
        path = database_url[len("sqlite:///") :]
        # In-memory / relative URLs have no directory to manage.
        return path if path and path not in (":memory:",) else None
    return None


def _relocate(src: str, dst: str) -> None:
    """Move `src` -> `dst` if `src` exists and `dst` does not (idempotent)."""
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)


def ensure_and_migrate() -> None:
    """Create the storage subdirs and migrate any legacy flat files into them.

    Safe to call every boot: directory creation and the moves are idempotent,
    and the moves only fire when a legacy file is present and the new location
    is still empty.
    """
    # --- Database: <data>/notes.db  ->  <data>/db/notes.db (+ WAL sidecars) ---
    db_path = _sqlite_file(os.getenv("DATABASE_URL"))
    if db_path:
        db_dir = os.path.dirname(db_path) or "."
        os.makedirs(db_dir, exist_ok=True)
        legacy_db = os.path.join(os.path.dirname(db_dir), os.path.basename(db_path))
        for suffix in ("", "-wal", "-shm"):
            _relocate(legacy_db + suffix, db_path + suffix)

    # --- Config JSON: <data>/*.json  ->  <data>/config/*.json ---
    config_dir = os.getenv("CONFIG_DIR")
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
        legacy_config_dir = os.path.dirname(config_dir.rstrip("/")) or "."
        for name in ("lm_config.json", "asr_config.json"):
            _relocate(os.path.join(legacy_config_dir, name), os.path.join(config_dir, name))

    # --- Uploads: ensure the directory exists (files already live here) ---
    upload_dir = os.getenv("UPLOAD_DIR")
    if upload_dir:
        os.makedirs(upload_dir, exist_ok=True)
