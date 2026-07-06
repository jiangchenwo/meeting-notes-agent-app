"""Startup migration of the legacy flat /data layout into subdirs."""
import os

import storage


def _env(monkeypatch, data_dir):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{data_dir}/db/notes.db")
    monkeypatch.setenv("UPLOAD_DIR", f"{data_dir}/uploads")
    monkeypatch.setenv("CONFIG_DIR", f"{data_dir}/config")


def test_relocates_legacy_flat_files(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "notes.db").write_text("DBDATA")
    (data / "lm_config.json").write_text("{}")
    (data / "asr_config.json").write_text('{"base_url": "x"}')
    _env(monkeypatch, str(data))

    storage.ensure_and_migrate()

    assert (data / "db" / "notes.db").read_text() == "DBDATA"
    assert (data / "config" / "lm_config.json").read_text() == "{}"
    assert (data / "config" / "asr_config.json").read_text() == '{"base_url": "x"}'
    # Legacy copies are moved, not left behind.
    assert not (data / "notes.db").exists()
    assert not (data / "lm_config.json").exists()
    # Uploads directory is created.
    assert (data / "uploads").is_dir()


def test_creates_dirs_on_fresh_volume(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _env(monkeypatch, str(data))

    storage.ensure_and_migrate()

    assert (data / "db").is_dir()
    assert (data / "config").is_dir()
    assert (data / "uploads").is_dir()
    assert not (data / "db" / "notes.db").exists()  # nothing to migrate


def test_does_not_clobber_existing_new_layout(monkeypatch, tmp_path):
    data = tmp_path / "data"
    (data / "db").mkdir(parents=True)
    (data / "db" / "notes.db").write_text("CURRENT")
    (data / "notes.db").write_text("STALE")  # a stray legacy file
    _env(monkeypatch, str(data))

    storage.ensure_and_migrate()

    # Existing new-layout DB is preserved; the stale legacy file is left alone.
    assert (data / "db" / "notes.db").read_text() == "CURRENT"
    assert (data / "notes.db").read_text() == "STALE"


def test_is_idempotent(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "notes.db").write_text("DBDATA")
    _env(monkeypatch, str(data))

    storage.ensure_and_migrate()
    storage.ensure_and_migrate()  # second run must not raise or lose data

    assert (data / "db" / "notes.db").read_text() == "DBDATA"
