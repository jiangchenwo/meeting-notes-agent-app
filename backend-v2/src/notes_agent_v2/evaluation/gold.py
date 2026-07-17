from __future__ import annotations

from collections import Counter, defaultdict
from enum import StrEnum
import hashlib
import json
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field


class GoldSelectionError(RuntimeError):
    pass


class GoldCohort(StrEnum):
    qmsum_general = "qmsum_general"
    qmsum_query = "qmsum_query"
    long_context = "long_context"
    meetingbank_asr = "meetingbank_asr"
    ami_audience = "ami_audience"
    ami_structured = "ami_structured"


REQUIRED_COUNTS = {
    GoldCohort.qmsum_general: 4,
    GoldCohort.qmsum_query: 4,
    GoldCohort.long_context: 4,
    GoldCohort.meetingbank_asr: 4,
    GoldCohort.ami_audience: 8,
    GoldCohort.ami_structured: 8,
}


class GoldCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    case_id: str = Field(min_length=1)
    meeting_id: str = Field(min_length=1)
    source_type: Literal["qmsum", "ami", "meetingbank"]
    cohort: GoldCohort
    upstream_rank: int = Field(ge=0)
    case_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_complete: bool
    synthetic: bool
    authored: bool
    audience: str | None = None


class GoldManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["development-gold-v1"] = "development-gold-v1"
    status: Literal["provisional"] = "provisional"
    cases: tuple[GoldCandidate, ...]
    cohort_counts: dict[str, int]
    selection_digest: str


def build_gold(candidates: Iterable[GoldCandidate], output: Path, *, reserved_case_ids: set[str] | None = None) -> GoldManifest:
    reserved = reserved_case_ids or set()
    eligible = [item for item in candidates if item.provenance_complete and not item.synthetic and not item.authored and item.case_id not in reserved]
    selected: list[GoldCandidate] = []
    for cohort, count in REQUIRED_COUNTS.items():
        cohort_items = sorted((item for item in eligible if item.cohort is cohort), key=lambda item: (item.upstream_rank, item.meeting_id, item.case_id))
        if cohort is GoldCohort.ami_audience:
            grouped: dict[str, list[GoldCandidate]] = defaultdict(list)
            for item in cohort_items:
                grouped[item.meeting_id].append(item)
            complete_groups = []
            for meeting_id, items in grouped.items():
                by_audience = {item.audience: item for item in items if item.audience}
                if len(by_audience) >= 4:
                    complete_groups.append((min(item.upstream_rank for item in items), meeting_id, tuple(sorted(by_audience.values(), key=lambda item: (item.audience or "", item.case_id))[:4])))
            complete_groups.sort(key=lambda item: (item[0], item[1]))
            if len(complete_groups) < 2:
                raise GoldSelectionError(f"cohort={cohort.value} requires two four-audience meeting groups; eligible={len(complete_groups)}")
            selected.extend([item for group in complete_groups[:2] for item in group[2]])
            continue
        if len(cohort_items) < count:
            raise GoldSelectionError(f"cohort={cohort.value} required={count} eligible={len(cohort_items)}")
        selected.extend(cohort_items[:count])
    ids = [item.case_id for item in selected]
    if len(ids) != len(set(ids)):
        raise GoldSelectionError("selected case IDs must be unique")
    meetings: dict[str, list[GoldCandidate]] = defaultdict(list)
    for item in selected:
        meetings[item.meeting_id].append(item)
    if any(len(items) > 1 and any(item.cohort is not GoldCohort.ami_audience for item in items) for items in meetings.values()):
        raise GoldSelectionError("duplicate meetings are allowed only for audience-target groups")
    canonical_cases = sorted((item.model_dump(mode="json") for item in selected), key=lambda item: item["case_id"])
    digest = hashlib.sha256(json.dumps(canonical_cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    counts = Counter(item.cohort.value for item in selected)
    manifest = GoldManifest(cases=tuple(sorted(selected, key=lambda item: item.case_id)), cohort_counts=dict(sorted(counts.items())), selection_digest=digest)
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("cases", "labels", "audit/packets"):
        (output / directory).mkdir(parents=True)
    for item in manifest.cases:
        _write_json(output / "cases" / f"{item.case_id}.json", {"case_id": item.case_id, "digest": item.case_digest})
        _write_json(output / "labels" / f"{item.case_id}.json", {"case_id": item.case_id, "digest": item.label_digest, "status": "provisional"})
        _write_json(output / "audit/packets" / f"{item.case_id}.json", {"case_id": item.case_id, "case_digest": item.case_digest, "label_digest": item.label_digest})
    _write_json(output / "manifest.json", manifest.model_dump(mode="json"))
    return manifest


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
