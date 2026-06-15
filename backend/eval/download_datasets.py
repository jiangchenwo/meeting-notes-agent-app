"""
Download public text-only eval samples from four datasets:

  Dataset                   Domain       Source
  -------                   ------       ------
  QMSum                     Project      pszemraj/qmsum-cleaned (HF)
  ACI-Bench                 Healthcare   ClinicianFOCUS/ACI-Bench-Refined (HF)
  MIT-OCW                   Education    jablonkagroup/mit-ocw-lecture-transcripts (HF)
  coding-interview-transcripts  Interview    iamanisin/coding_interview_transcripts (HF)

Saves N_SAMPLES items per dataset under eval/data/<dataset>/NNN.json.
Skips a dataset if its output folder already has files.

Run from the backend directory:
    uv run python eval/download_datasets.py
"""

import json
import os

import requests

N_SAMPLES = 10

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")

# Load HF_TOKEN from backend/.env if not already in environment
def _load_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        env_path = os.path.join(os.path.dirname(_HERE), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        os.environ["HF_TOKEN"] = token
                        break
    return token


def _save(dataset_name: str, items: list[dict]) -> None:
    out_dir = os.path.join(DATA_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    for i, item in enumerate(items):
        with open(os.path.join(out_dir, f"{i:03d}.json"), "w") as f:
            json.dump(item, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {dataset_name}: {len(items)} samples → eval/data/{dataset_name}/")


def _already_downloaded(dataset_name: str) -> bool:
    out_dir = os.path.join(DATA_DIR, dataset_name)
    if os.path.isdir(out_dir):
        count = sum(1 for f in os.listdir(out_dir) if f.endswith(".json"))
        if count > 0:
            print(f"  ↩ {dataset_name}: {count} samples already present, skipping")
            return True
    return False


# ---------------------------------------------------------------------------
# QMSum — committee / product / academic meeting transcripts
# HF id: pszemraj/qmsum-cleaned  splits: train/validation/test
# fields: meeting_transcripts (list[{speaker,content}]), input (query), output (answer)
# ---------------------------------------------------------------------------

def download_qmsum() -> None:
    if _already_downloaded("qmsum"):
        return
    print("Downloading QMSum …")
    from datasets import load_dataset  # type: ignore

    # Use train split — test split has empty output (no gold summaries).
    # Each row's "input" field is "query\ntranscript"; split on first newline.
    ds = load_dataset("pszemraj/qmsum-cleaned", split="train")
    items = []
    for row in ds.select(range(min(N_SAMPLES, len(ds)))):
        inp = row.get("input") or ""
        nl = inp.find("\n")
        query = inp[:nl].strip() if nl != -1 else ""
        transcript = inp[nl + 1:].strip() if nl != -1 else inp
        items.append({
            "dataset": "qmsum",
            "domain": "Project",
            "query": query,
            "transcript": transcript,
            "summary": row.get("output") or "",
        })
    _save("qmsum", items)


# ---------------------------------------------------------------------------
# ACI-Bench — ambulatory clinical encounter note generation
# HF id: ClinicianFOCUS/ACI-Bench-Refined  splits: train/validation/test
# fields: dialogue (doctor-patient conversation), note (SOAP-style clinical note)
# ---------------------------------------------------------------------------

def download_aci_bench() -> None:
    if _already_downloaded("aci-bench"):
        return
    print("Downloading ACI-Bench …")
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("ClinicianFOCUS/ACI-Bench-Refined", split="test")
    items = []
    for row in ds.select(range(min(N_SAMPLES, len(ds)))):
        items.append({
            "dataset": "aci-bench",
            "domain": "Healthcare",
            "transcript": row.get("dialogue") or "",
            "note": row.get("note") or "",
        })
    _save("aci-bench", items)


# ---------------------------------------------------------------------------
# MIT OCW Lecture Transcripts — university STEM lectures (Education proxy)
# HF id: jablonkagroup/mit-ocw-lecture-transcripts  splits: train/test/valid
# fields: text (ASR transcript), course, topic
# Note: no pre-made summary — the transcript is the raw material for summarization.
# ---------------------------------------------------------------------------

def download_mit_ocw() -> None:
    if _already_downloaded("mit-ocw"):
        return
    print("Downloading MIT OCW Lecture Transcripts …")
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("jablonkagroup/mit-ocw-lecture-transcripts", split="test")
    items = []
    for row in ds.select(range(min(N_SAMPLES, len(ds)))):
        items.append({
            "dataset": "mit-ocw",
            "domain": "Education",
            "course": row.get("course") or "",
            "topic": row.get("topic") or "",
            "transcript": row.get("text") or "",
            "summary": "",  # no gold summary in this dataset
        })
    _save("mit-ocw", items)


# ---------------------------------------------------------------------------
# Coding Interview Transcripts — simulated tech / SWE job interviews
# HF id: iamanisin/coding_interview_transcripts  split: train (41 samples)
# fields: conversations (list[{role, content}]) — system + interviewer/candidate turns
# We skip the system prompt and the initial trigger, then reconstruct the
# dialogue as "Interviewer: ..." / "Candidate: ..." alternating speaker turns.
# ---------------------------------------------------------------------------

def download_coding_interviews() -> None:
    if _already_downloaded("coding-interviews"):
        return
    print("Downloading Coding Interview Transcripts …")
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("iamanisin/coding_interview_transcripts", split="train")

    # Skip rows with ≤5 turns (just system + trigger — no real content)
    substantive = [r for r in ds if len(r["conversations"]) > 5]

    items = []
    for row in substantive[:N_SAMPLES]:
        convs = row["conversations"]
        # Drop system message and pure trigger lines
        dialogue = [
            m for m in convs
            if m.get("role") != "system"
            and not str(m.get("content", "")).strip().lower().startswith("start the interview")
        ]
        # Both interviewer and candidate appear as "assistant" turns; alternate labels.
        interviewer_turn = True
        lines = []
        for m in dialogue:
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            if m.get("role") == "user":
                label = "Interviewer"
            else:
                label = "Interviewer" if interviewer_turn else "Candidate"
                interviewer_turn = not interviewer_turn
            lines.append(f"{label}: {content}")
        transcript = "\n".join(lines)
        items.append({
            "dataset": "coding-interviews",
            "domain": "Interview",
            "transcript": transcript,
            "summary": "",
        })
    _save("coding-interviews", items)


# ---------------------------------------------------------------------------
# Manifest — index of all downloaded samples
# ---------------------------------------------------------------------------

def write_manifest() -> None:
    manifest = []
    for dataset_name in sorted(os.listdir(DATA_DIR)):
        dataset_dir = os.path.join(DATA_DIR, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        for fname in sorted(os.listdir(dataset_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(dataset_dir, fname)) as f:
                item = json.load(f)
            manifest.append({
                "path": f"{dataset_name}/{fname}",
                "dataset": item.get("dataset"),
                "domain": item.get("domain"),
                "transcript_chars": len(item.get("transcript", "")),
            })
    with open(os.path.join(DATA_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    total = len(manifest)
    by_domain = {}
    for m in manifest:
        by_domain.setdefault(m["domain"], 0)
        by_domain[m["domain"]] += 1
    print(f"\n  ✓ manifest: {total} total samples")
    for domain, count in sorted(by_domain.items()):
        print(f"    {domain}: {count}")


if __name__ == "__main__":
    _load_token()
    os.makedirs(DATA_DIR, exist_ok=True)
    errors = []
    for fn in [download_qmsum, download_aci_bench, download_mit_ocw, download_coding_interviews]:
        try:
            fn()
        except Exception as exc:
            name = fn.__name__.replace("download_", "")
            print(f"  ✗ {name}: {exc}")
            errors.append((name, str(exc)))
    write_manifest()
    if errors:
        print("\nFailed:")
        for name, err in errors:
            print(f"  {name}: {err[:120]}")
