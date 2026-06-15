"""
Loads the downloaded public dataset samples from eval/data/ as EvalCase objects.

Case IDs use the format "<dataset>-<NNN>" (e.g. "qmsum-000") to avoid
URL path-separator issues in the REST endpoints.
"""
import json
import os

from .cases import EvalCase

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

_DATASET_DOMAIN = {
    "qmsum":             "Project",
    "aci-bench":         "Healthcare",
    "mit-ocw":           "Education",
    "coding-interviews": "Interview",
}

_DATASET_LABEL = {
    "qmsum":             "QMSum Meeting",
    "aci-bench":         "ACI-Bench Clinical",
    "mit-ocw":           "MIT OCW Lecture",
    "coding-interviews": "Coding Interview",
}


def get_public_cases(domain: str | None = None) -> list[EvalCase]:
    """Return all downloaded public-dataset cases, optionally filtered by domain."""
    cases: list[EvalCase] = []
    for dataset_name, domain_name in _DATASET_DOMAIN.items():
        if domain and domain.lower() != domain_name.lower():
            continue
        dataset_dir = os.path.join(_DATA_DIR, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        for fname in sorted(os.listdir(dataset_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(dataset_dir, fname)
            with open(fpath) as f:
                item = json.load(f)
            idx = int(fname.split(".")[0])
            case_id = f"{dataset_name}-{idx:03d}"
            # Build minimal ground truth from any available gold label
            gold = item.get("summary") or item.get("note") or ""
            ground_truth: dict = {"gold_label": gold} if gold else {}
            cases.append(EvalCase(
                id=case_id,
                domain=domain_name,
                title=f"{_DATASET_LABEL[dataset_name]} #{idx + 1}",
                source=dataset_name,
                transcript=item.get("transcript", ""),
                ground_truth=ground_truth,
            ))
    return cases


def get_public_case(case_id: str) -> EvalCase | None:
    """Look up a single public-dataset case by its ID."""
    # Extract dataset name from "<dataset>-<NNN>"
    parts = case_id.rsplit("-", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    dataset_name, idx_str = parts
    idx = int(idx_str)
    dataset_dir = os.path.join(_DATA_DIR, dataset_name)
    fpath = os.path.join(dataset_dir, f"{idx:03d}.json")
    if not os.path.isfile(fpath):
        return None
    with open(fpath) as f:
        item = json.load(f)
    domain_name = _DATASET_DOMAIN.get(dataset_name)
    if not domain_name:
        return None
    gold = item.get("summary") or item.get("note") or ""
    ground_truth: dict = {"gold_label": gold} if gold else {}
    return EvalCase(
        id=case_id,
        domain=domain_name,
        title=f"{_DATASET_LABEL[dataset_name]} #{idx + 1}",
        source=dataset_name,
        transcript=item.get("transcript", ""),
        ground_truth=ground_truth,
    )
