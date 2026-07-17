from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from notes_agent_v2.evaluation.contracts import ImportedMeeting, ImportedReference, Utterance
from notes_agent_v2.evaluation.fetch import FetchError, extract_archive, fetch_archive
from notes_agent_v2.evaluation.importers import ImportError, import_meetingbank, import_qmsum
from notes_agent_v2.evaluation.labels import LabelProvenance, ReferenceGold


def provenance() -> LabelProvenance:
    return LabelProvenance(
        source_type="qmsum",
        release="1.0",
        license="research",
        upstream_digest="a" * 64,
    )


def test_contracts_are_frozen_and_digest_is_stable() -> None:
    meeting = ImportedMeeting(
        meeting_id="m1",
        source_type="qmsum",
        utterances=(Utterance(id="u1", speaker="A", text="Ship Friday."),),
        provenance=provenance(),
    )
    assert meeting.canonical_digest == ImportedMeeting.model_validate(meeting.model_dump()).canonical_digest
    with pytest.raises(Exception):
        meeting.meeting_id = "changed"  # type: ignore[misc]


def test_reference_gold_requires_upstream_provenance_and_resolved_evidence() -> None:
    with pytest.raises(ValueError, match="upstream digest"):
        LabelProvenance(source_type="qmsum", release="1", license="x", upstream_digest="0" * 64)
    with pytest.raises(ValueError, match="evidence"):
        ReferenceGold(
            reference_id="r1",
            meeting_id="m1",
            text="Ship Friday.",
            evidence_ids=("missing",),
            provenance=provenance(),
            available_utterance_ids=("u1",),
        )


def test_qmsum_import_is_exact_and_rejects_bad_spans(tmp_path: Path) -> None:
    source = tmp_path / "q.json"
    source.write_text(json.dumps({
        "meeting_id": "m1",
        "meeting_transcripts": [
            {"speaker": "A", "content": "Ship Friday."},
            {"speaker": "B", "content": "Approved."},
        ],
        "general_query_list": [{"query": "summary", "answer": "Ship Friday.", "relevant_text_span": [[0, 1]]}],
        "specific_query_list": [],
    }))
    meeting, references = import_qmsum(source, provenance())
    assert [u.text for u in meeting.utterances] == ["Ship Friday.", "Approved."]
    assert references[0].text == "Ship Friday."
    assert references[0].evidence_ids == ("m1:u0", "m1:u1")

    payload = json.loads(source.read_text())
    payload["general_query_list"][0]["relevant_text_span"] = [[1, 0]]
    source.write_text(json.dumps(payload))
    with pytest.raises(ImportError, match="range"):
        import_qmsum(source, provenance())


def test_meetingbank_rejects_segment_outside_transcript(tmp_path: Path) -> None:
    source = tmp_path / "m.json"
    source.write_text(json.dumps({"meeting_id": "m", "transcript": "abc", "segments": [{"start": 0, "end": 4}]}))
    with pytest.raises(ImportError, match="bounds"):
        import_meetingbank(source, provenance())


def test_archive_extraction_rejects_traversal_and_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        handle.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(FetchError, match="unsafe"):
        extract_archive(archive, tmp_path / "out", hashlib.sha256(archive.read_bytes()).hexdigest())
    assert not (tmp_path / "out").exists()


def test_fetch_is_disabled_without_explicit_network_permission(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="disabled"):
        fetch_archive("https://example.test/archive.tar", tmp_path / "archive.tar", allow_network=False)
