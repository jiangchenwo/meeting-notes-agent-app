"""Loads downloaded public-dataset samples from eval/data/ as EvalCase objects.

Ground truth for these cases is a human reference summary
(``ground_truth["reference_summary"]``), scored by the ReferenceAlignment
evaluator; the hand-authored fact/action metrics stay unscored for them.

Case IDs use the format "<dataset>-<NNN>" (e.g. "qmsum-000").
Run ``uv run python -m eval.download_datasets`` first to fetch the samples.
"""
import json
import os

from .cases import EvalCase

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

PUBLIC_DATASETS = ("meetingbank", "qmsum")

_DATASET_LABEL = {
    "meetingbank": "MeetingBank",
    "qmsum": "QMSum",
}


def get_public_cases(dataset: str | None = None, limit: int | None = None) -> list[EvalCase]:
    """Return downloaded public cases, optionally one dataset, at most `limit` each."""
    names = PUBLIC_DATASETS if dataset in (None, "all") else (dataset,)
    cases: list[EvalCase] = []
    for name in names:
        if name not in PUBLIC_DATASETS:
            raise SystemExit(f"unknown public dataset {name!r} (choose from {', '.join(PUBLIC_DATASETS)})")
        dataset_dir = os.path.join(DATA_DIR, name)
        if not os.path.isdir(dataset_dir):
            raise SystemExit(
                f"no downloaded data for {name!r} — run `uv run python -m eval.download_datasets` first"
            )
        fnames = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".json"))
        for fname in fnames[:limit]:
            with open(os.path.join(dataset_dir, fname)) as f:
                item = json.load(f)
            idx = int(fname.split(".")[0])
            cases.append(EvalCase(
                id=f"{name}-{idx:03d}",
                domain=item["domain"],
                title=f"{_DATASET_LABEL[name]}: {item.get('title', '')[:60]}",
                source=name,
                transcript=item["transcript"],
                ground_truth={"reference_summary": item.get("reference_summary", "")},
            ))
    return cases
