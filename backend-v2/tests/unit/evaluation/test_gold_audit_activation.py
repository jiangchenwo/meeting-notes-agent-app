from __future__ import annotations

from pathlib import Path

import pytest

from notes_agent_v2.evaluation.activation import ActivationError, activate_gold
from notes_agent_v2.evaluation.audit import AuditRecord, AuditStatus, qualify_auditor
from notes_agent_v2.evaluation.gold import GoldCandidate, GoldCohort, GoldSelectionError, build_gold


def candidates() -> list[GoldCandidate]:
    result = []
    cohorts = {
        GoldCohort.qmsum_general: 4,
        GoldCohort.qmsum_query: 4,
        GoldCohort.long_context: 4,
        GoldCohort.meetingbank_asr: 4,
        GoldCohort.ami_audience: 8,
        GoldCohort.ami_structured: 8,
    }
    for cohort, count in cohorts.items():
        for index in range(count):
            result.append(GoldCandidate(
                case_id=f"{cohort.value}-{index}", meeting_id=f"m-{cohort.value}-{index // 4 if cohort is GoldCohort.ami_audience else index}",
                source_type="meetingbank" if cohort is GoldCohort.meetingbank_asr else "ami" if cohort.value.startswith("ami") else "qmsum",
                cohort=cohort, upstream_rank=index, case_digest=f"{index + 1:064x}", label_digest=f"{index + 101:064x}",
                provenance_complete=True, synthetic=False, authored=False,
                audience=f"audience-{index % 4}" if cohort is GoldCohort.ami_audience else None,
            ))
    return result


def test_gold_builder_selects_exact_32_deterministically(tmp_path: Path) -> None:
    one = build_gold(candidates(), tmp_path / "one")
    two = build_gold(reversed(candidates()), tmp_path / "two")
    assert len(one.cases) == 32
    assert one.selection_digest == two.selection_digest
    assert all(not item.synthetic and not item.authored for item in one.cases)
    one_files = {str(path.relative_to(tmp_path / "one")): path.read_bytes() for path in (tmp_path / "one").rglob("*") if path.is_file()}
    two_files = {str(path.relative_to(tmp_path / "two")): path.read_bytes() for path in (tmp_path / "two").rglob("*") if path.is_file()}
    assert one_files == two_files


def test_gold_builder_reports_typed_shortage(tmp_path: Path) -> None:
    with pytest.raises(GoldSelectionError, match="ami_structured.*eligible=7"):
        build_gold(candidates()[:-1], tmp_path / "out")


def test_gold_builder_requires_two_four_audience_meeting_groups(tmp_path: Path) -> None:
    values = candidates()
    changed = [item.model_copy(update={"meeting_id": f"unique-{index}"}) if item.cohort is GoldCohort.ami_audience else item for index, item in enumerate(values)]
    with pytest.raises(GoldSelectionError, match="audience"):
        build_gold(changed, tmp_path / "out")


def test_auditor_qualification_requires_clean_and_tamper_accuracy() -> None:
    records = [AuditRecord(packet_id=f"clean-{i}", expected_valid=True, observed_valid=True) for i in range(3)]
    records += [AuditRecord(packet_id=f"bad-{i}", expected_valid=False, observed_valid=False) for i in range(3)]
    assert qualify_auditor(records).status is AuditStatus.qualified


def test_activation_requires_two_unanimous_audits_and_rolls_back(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "manifest.json").write_text("manifest")
    audits = {item.case_id: (True, True) for item in candidates()}
    active = tmp_path / "active"
    activate_gold(staging, active, expected_case_ids=set(audits), audit_decisions=audits)
    assert (active / "manifest.json").read_text() == "manifest"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "manifest.json").write_text("new")
    broken = dict(audits)
    broken[next(iter(broken))] = (True, False)
    with pytest.raises(ActivationError):
        activate_gold(replacement, active, expected_case_ids=set(audits), audit_decisions=broken)
    assert (active / "manifest.json").read_text() == "manifest"

    activate_gold(replacement, active, expected_case_ids=set(audits), audit_decisions=audits)
    assert (active / "manifest.json").read_text() == "new"
    archived = tuple((tmp_path / "archive").glob("active-*"))
    assert len(archived) == 1
    assert (archived[0] / "manifest.json").read_text() == "manifest"
