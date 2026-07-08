"""Eval metric helpers — pure Python, no LLM."""
import json

from eval.metrics import action_recall, coverage_score, hallucination_check, summary_alignment


def test_coverage_counts_paraphrased_facts():
    output = (
        "## Summary\nThe Q3 actuals came in at $840,000 against a $900,000 budget. "
        "The observability contract expires on October 15th."
    )
    facts = [
        "Q3 budget was 900,000",
        "Q3 actuals were 840,000",
        "observability contract expires October 15th",
        "two open headcount slots",  # not mentioned
    ]
    assert coverage_score(output, facts) == 0.75


def test_coverage_empty_facts_scores_one():
    assert coverage_score("anything", []) == 1.0


def test_coverage_zero_when_nothing_matches():
    assert coverage_score("totally unrelated text", ["quarterly revenue grew 40 percent"]) == 0.0


def test_action_recall_matches_owners_and_tasks():
    items = [{"task": "Update the headcount tracker", "owner": "Lisa"}]
    assert action_recall(items, ["Lisa", "Raj"], ["update headcount tracker"]) == round(2 / 3, 3)


def test_action_recall_unscored_without_expectations():
    assert action_recall([{"task": "x"}], [], []) is None


def test_hallucination_flags_unknown_proper_nouns():
    transcript = "Alice said the budget is fine."
    output = "Alice and Bertrand discussed the Budget."
    flags = hallucination_check(output, transcript)
    assert "Bertrand" in flags
    assert "Alice" not in flags


def test_summary_alignment_perfect_overlap():
    text = "council adopted resolution 31669 regarding tenant screening"
    assert summary_alignment(text, text) == {
        "reference_recall": 1.0,
        "reference_precision": 1.0,
    }


def test_summary_alignment_partial_recall():
    reference = "council adopted resolution 31669 regarding tenant screening"
    generated = "The council adopted resolution 31669. It also discussed parks funding."
    scores = summary_alignment(generated, reference)
    assert scores is not None
    assert 0 < scores["reference_recall"] < 1
    assert 0 < scores["reference_precision"] < 1


def test_summary_alignment_unscored_when_empty():
    assert summary_alignment("", "some reference") is None
    assert summary_alignment("some output", "") is None


def test_public_cases_loader(tmp_path, monkeypatch):
    import eval.public_cases as pc

    sample = {
        "dataset": "qmsum",
        "domain": "Project",
        "title": "remote control design",
        "transcript": "A: hello\nB: let's design the remote",
        "reference_summary": "The team discussed the remote control design.",
    }
    (tmp_path / "qmsum").mkdir()
    (tmp_path / "qmsum" / "000.json").write_text(json.dumps(sample))
    (tmp_path / "meetingbank").mkdir()
    (tmp_path / "meetingbank" / "000.json").write_text(
        json.dumps({**sample, "dataset": "meetingbank", "domain": "General"})
    )
    monkeypatch.setattr(pc, "DATA_DIR", str(tmp_path))

    cases = pc.get_public_cases()
    assert {c.id for c in cases} == {"meetingbank-000", "qmsum-000"}

    only_qmsum = pc.get_public_cases("qmsum", limit=1)
    assert len(only_qmsum) == 1
    case = only_qmsum[0]
    assert case.domain == "Project"
    assert case.ground_truth == {"reference_summary": sample["reference_summary"]}
    assert case.transcript == sample["transcript"]
