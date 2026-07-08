"""
Download public eval samples for the pipeline eval harness:

  Dataset      Domain   Source                                        Gold label
  -------      ------   ------                                        ----------
  MeetingBank  General  huuuyeah/meetingbank (HF, plain JSON files)   human summary
  QMSum        Project  Yale-LILY/QMSum (GitHub raw jsonl, val set)   "Summarize the whole meeting" answer

Saves N_SAMPLES items per dataset under eval/data/<dataset>/NNN.json and skips
a dataset whose folder already has files. Plain httpx — no `datasets` library.

Samples are picked evenly across the split from transcripts within
[MIN_CHARS, MAX_CHARS]; the cap keeps a serial laptop eval run tractable
(very long meetings exercise the chunked map-reduce path many times over).

Run from the backend directory:
    uv run python -m eval.download_datasets
"""
import json
import os

import httpx

N_SAMPLES = 5
MIN_CHARS = 3_000
MAX_CHARS = 50_000

MEETINGBANK_URL = "https://huggingface.co/datasets/huuuyeah/meetingbank/resolve/main/test.json"
QMSUM_URL = "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl/val.jsonl"

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")


def _already_downloaded(dataset_name: str) -> bool:
    out_dir = os.path.join(DATA_DIR, dataset_name)
    if os.path.isdir(out_dir):
        count = sum(1 for f in os.listdir(out_dir) if f.endswith(".json"))
        if count > 0:
            print(f"  ↩ {dataset_name}: {count} samples already present, skipping")
            return True
    return False


def _save(dataset_name: str, items: list[dict]) -> None:
    out_dir = os.path.join(DATA_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    for i, item in enumerate(items):
        with open(os.path.join(out_dir, f"{i:03d}.json"), "w") as f:
            json.dump(item, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {dataset_name}: {len(items)} samples → eval/data/{dataset_name}/")
    for i, item in enumerate(items):
        print(f"      {i:03d}  {len(item['transcript']):>6} chars  {item['title'][:60]}")


def _pick_spread(items: list[dict], n: int) -> list[dict]:
    """Evenly spaced picks across the (order-preserved) pool for variety."""
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def download_meetingbank() -> None:
    if _already_downloaded("meetingbank"):
        return
    print("Downloading MeetingBank (test split, ~13 MB) …")
    text = httpx.get(MEETINGBANK_URL, timeout=120, follow_redirects=True).text
    # The file is one JSON object per line.
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    pool = [
        {
            "dataset": "meetingbank",
            "domain": "General",
            "title": row.get("uid") or f"meetingbank-{row.get('id')}",
            "transcript": row["transcript"],
            "reference_summary": row["summary"],
        }
        for row in rows
        if MIN_CHARS <= len(row.get("transcript") or "") <= MAX_CHARS
        and (row.get("summary") or "").strip()
    ]
    _save("meetingbank", _pick_spread(pool, N_SAMPLES))


def download_qmsum() -> None:
    if _already_downloaded("qmsum"):
        return
    print("Downloading QMSum (val split) …")
    text = httpx.get(QMSUM_URL, timeout=120, follow_redirects=True).text
    pool = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        transcript = "\n".join(
            f"{t['speaker']}: {t['content']}" for t in row.get("meeting_transcripts", [])
        )
        # Gold label: the answer to the whole-meeting summary query.
        reference = ""
        for q in row.get("general_query_list", []):
            if "summarize the whole meeting" in q.get("query", "").lower():
                reference = q.get("answer", "")
                break
        if not reference or not (MIN_CHARS <= len(transcript) <= MAX_CHARS):
            continue
        topics = ", ".join(t.get("topic", "") for t in row.get("topic_list", [])[:2])
        pool.append({
            "dataset": "qmsum",
            "domain": "Project",
            "title": topics or "QMSum meeting",
            "transcript": transcript,
            "reference_summary": reference,
        })
    _save("qmsum", _pick_spread(pool, N_SAMPLES))


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    for fn in [download_meetingbank, download_qmsum]:
        try:
            fn()
        except Exception as exc:
            print(f"  ✗ {fn.__name__.replace('download_', '')}: {type(exc).__name__}: {exc}")
