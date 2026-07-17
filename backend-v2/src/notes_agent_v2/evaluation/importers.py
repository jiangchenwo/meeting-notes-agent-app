from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import ImportedMeeting, ImportedReference, Utterance
from .labels import LabelProvenance


class ImportError(ValueError):
    pass


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ImportError("benchmark input is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ImportError("benchmark input must be an object")
    return value


def import_qmsum(path: Path, provenance: LabelProvenance) -> tuple[ImportedMeeting, tuple[ImportedReference, ...]]:
    payload = _load_object(path)
    meeting_id = str(payload.get("meeting_id") or path.stem)
    turns = payload.get("meeting_transcripts")
    if not isinstance(turns, list) or not turns:
        raise ImportError("QMSum requires non-empty turns")
    utterances: list[Utterance] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict) or not str(turn.get("content", "")).strip():
            raise ImportError("QMSum turn content is empty")
        utterances.append(Utterance(
            id=f"{meeting_id}:u{index}",
            speaker=str(turn.get("speaker") or "unknown"),
            text=str(turn["content"]).strip(),
        ))
    references: list[ImportedReference] = []
    for task_type, key in (("general", "general_query_list"), ("query_focused", "specific_query_list")):
        rows = payload.get(key, [])
        if not isinstance(rows, list):
            raise ImportError("QMSum query lists must be arrays")
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not str(row.get("answer", "")).strip():
                raise ImportError("QMSum answer is empty")
            evidence: list[str] = []
            spans = row.get("relevant_text_span", [])
            if not isinstance(spans, list):
                raise ImportError("QMSum evidence spans must be an array")
            for span in spans:
                if not isinstance(span, list) or len(span) != 2:
                    raise ImportError("QMSum evidence span is malformed")
                start, end = span
                if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
                    raise ImportError("QMSum evidence range is reversed or invalid")
                if end >= len(utterances):
                    raise ImportError("QMSum evidence range is outside transcript")
                evidence.extend(item.id for item in utterances[start : end + 1])
            references.append(ImportedReference(
                reference_id=f"{meeting_id}:{task_type}:{index}", meeting_id=meeting_id,
                task_type=task_type, text=str(row["answer"]).strip(),
                evidence_ids=tuple(dict.fromkeys(evidence)), query=str(row.get("query") or "") or None,
            ))
    return ImportedMeeting(meeting_id=meeting_id, source_type="qmsum", utterances=tuple(utterances), provenance=provenance), tuple(references)


def import_meetingbank(path: Path, provenance: LabelProvenance) -> tuple[ImportedMeeting, tuple[ImportedReference, ...]]:
    payload = _load_object(path)
    meeting_id = str(payload.get("meeting_id") or path.stem)
    transcript = payload.get("transcript")
    segments = payload.get("segments")
    if not isinstance(transcript, str) or not transcript or not isinstance(segments, list) or not segments:
        raise ImportError("MeetingBank requires transcript and segments")
    utterances: list[Utterance] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ImportError("MeetingBank segment is malformed")
        start, end = segment.get("start"), segment.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(transcript):
            raise ImportError("MeetingBank segment bounds are invalid")
        text = transcript[start:end].strip()
        if not text:
            raise ImportError("MeetingBank segment is empty")
        utterances.append(Utterance(id=f"{meeting_id}:u{index}", speaker=str(segment.get("speaker") or "unknown"), text=text))
    return ImportedMeeting(meeting_id=meeting_id, source_type="meetingbank", utterances=tuple(utterances), provenance=provenance), ()


def import_ami(path: Path, provenance: LabelProvenance) -> tuple[ImportedMeeting, tuple[ImportedReference, ...]]:
    payload = _load_object(path)
    meeting_id = str(payload.get("meeting_id") or path.stem)
    acts = payload.get("dialogue_acts")
    if not isinstance(acts, list) or not acts:
        raise ImportError("AMI requires dialogue acts")
    utterances = []
    for index, act in enumerate(acts):
        if not isinstance(act, dict):
            raise ImportError("AMI dialogue act is malformed")
        text = str(act.get("text") or "").strip()
        if not text or bool(act.get("nonverbal")):
            continue
        pointer = act.get("pointer")
        if pointer is None:
            raise ImportError("AMI NITE pointer is unresolved")
        utterances.append(Utterance(id=f"{meeting_id}:{pointer}", speaker=str(act.get("participant") or "unknown"), text=text))
    if not utterances:
        raise ImportError("AMI contains only nonverbal acts")
    return ImportedMeeting(meeting_id=meeting_id, source_type="ami", utterances=tuple(utterances), provenance=provenance), ()
