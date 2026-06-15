import json
import os
import tempfile
import uuid
from datetime import datetime, timezone

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

import lm_config
import whisper_config as wcfg
from agents.lab_runner import (
    action_recall,
    compute_bertscore,
    compute_rouge,
    coverage_score,
    hallucination_check,
    run_naive,
    run_pipeline,
    schema_check,
)
from eval.cases import get_case, get_cases
from eval.public_cases import get_public_case, get_public_cases
from whisper_utils import _find_binary, _find_model, run_whisper_file

router = APIRouter(prefix="/api/lab", tags=["lab"])

_FIXTURE_MAP: dict = {}  # kept for compatibility — built-in fixtures moved to eval/cases.py

DOMAINS = ["General", "Education", "Healthcare", "Interview", "Project"]
ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".webm"}

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_LIBRARY_DIR = os.path.join(_BACKEND_DIR, "eval", "audio")
HISTORY_DIR = os.path.join(_BACKEND_DIR, "eval", "history")
os.makedirs(HISTORY_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: build eval dict for a completed run
# ---------------------------------------------------------------------------

def _compute_eval(result: dict, naive_summary: str, transcript: str, ground_truth: dict) -> dict:
    all_facts: list[str] = [
        f
        for v in ground_truth.values()
        if isinstance(v, list) and all(isinstance(x, str) for x in v)
        for f in v
    ]
    combined = " ".join([
        result["summary_text"],
        result["suggestions_text"],
        json.dumps(result.get("action_items", [])),
    ])
    agent_cov = coverage_score(combined, all_facts)
    naive_cov = coverage_score(naive_summary, all_facts)
    ar = action_recall(
        result["action_items"],
        ground_truth.get("action_owners", []),
        ground_truth.get("action_tasks", []),
    )
    hallucs = hallucination_check(
        result["summary_text"] + " " + result["suggestions_text"],
        transcript,
    )
    gold_label: str = ground_truth.get("gold_label", "")
    # Metric computations use the full untruncated transcript — no [:N] slicing
    metrics: dict = {
        "agent_coverage": agent_cov,
        "naive_coverage": naive_cov,
        "coverage_delta": round(agent_cov - naive_cov, 3),
        "action_recall": ar,
        "hallucinations": hallucs,
        "hallucination_count": len(hallucs),
        "ground_truth_facts": len(all_facts),
        "schema_check": schema_check(result["results"]),
        "rouge_vs_transcript": compute_rouge(result["summary_text"], transcript),
        "rouge_naive_vs_transcript": compute_rouge(naive_summary, transcript),
        "bertscore_vs_transcript": compute_bertscore(result["summary_text"], transcript),
        "bertscore_naive_vs_transcript": compute_bertscore(naive_summary, transcript),
    }
    if gold_label:
        metrics["rouge_vs_gold"] = compute_rouge(result["summary_text"], gold_label)
        metrics["rouge_naive_vs_gold"] = compute_rouge(naive_summary, gold_label)
        metrics["bertscore_vs_gold"] = compute_bertscore(result["summary_text"], gold_label)
        metrics["bertscore_naive_vs_gold"] = compute_bertscore(naive_summary, gold_label)
    return metrics


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------

def _save_history(result: dict) -> str:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    payload = {"id": run_id, "saved_at": datetime.now(timezone.utc).isoformat(), **result}
    with open(os.path.join(HISTORY_DIR, f"{run_id}.json"), "w") as f:
        json.dump(payload, f)
    return run_id


@router.get("/history")
def list_history():
    entries = []
    for fn in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(HISTORY_DIR, fn)) as f:
                data = json.load(f)
            entries.append({
                "id": data.get("id", fn[:-5]),
                "saved_at": data.get("saved_at"),
                "domain": data.get("domain"),
                "run_label": data.get("run_label"),
                "case_id": data.get("case_id"),
                "confidence_score": data.get("confidence_score"),
                "total_ms": data.get("total_ms"),
                "step_count": len(data.get("steps", [])),
                "chunked": data.get("chunked", False),
                "eval_coverage": data.get("eval", {}).get("agent_coverage"),
                "eval_coverage_delta": data.get("eval", {}).get("coverage_delta"),
            })
        except Exception:
            continue
    return entries


@router.get("/history/{run_id}")
def get_history_run(run_id: str):
    path = os.path.join(HISTORY_DIR, f"{run_id}.json")
    if not os.path.isfile(path):
        raise HTTPException(404, f"History run '{run_id}' not found.")
    with open(path) as f:
        return json.load(f)


@router.delete("/history/{run_id}", status_code=204)
def delete_history_run(run_id: str):
    path = os.path.join(HISTORY_DIR, f"{run_id}.json")
    if not os.path.isfile(path):
        raise HTTPException(404, f"History run '{run_id}' not found.")
    os.remove(path)


# ---------------------------------------------------------------------------
# Domain / fixture endpoints
# ---------------------------------------------------------------------------

@router.get("/domains")
def list_domains():
    return [{"domain": d} for d in DOMAINS]


@router.get("/fixture/{domain}")
def get_fixture(domain: str):
    raise HTTPException(
        404,
        "Built-in fixtures removed. Use GET /api/lab/datasets or supply a transcript directly.",
    )


# ---------------------------------------------------------------------------
# Dataset endpoints (text-only eval cases)
# ---------------------------------------------------------------------------

@router.get("/datasets")
def list_datasets(domain: str | None = None):
    """List built-in synthetic cases and downloaded public-dataset samples."""
    cases = get_cases(domain) + get_public_cases(domain)
    return [
        {
            "id": c.id,
            "domain": c.domain,
            "title": c.title,
            "source": c.source,
            "transcript_length": len(c.transcript),
            "fact_count": len(c.ground_truth.get("facts", [])),
        }
        for c in cases
    ]


@router.get("/datasets/{case_id}")
def get_dataset(case_id: str):
    """Return a single eval case including transcript and ground truth."""
    case = get_case(case_id) or get_public_case(case_id)
    if not case:
        raise HTTPException(404, f"Case '{case_id}' not found.")
    return {
        "id": case.id,
        "domain": case.domain,
        "title": case.title,
        "source": case.source,
        "transcript": case.transcript,
        "ground_truth": case.ground_truth,
    }


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

class WorkflowOverride(BaseModel):
    steps: list[str]
    critique_steps: list[str] = []
    critique_threshold: float = 7.0
    max_retries: int = 1


class LabRunRequest(BaseModel):
    domain: str
    transcript: str | None = None
    case_id: str | None = None
    knowledge_base: str = ""
    system_prompt: str = ""
    template_prompt: str = ""
    workflow_override: WorkflowOverride | None = None
    run_label: str | None = None


@router.post("/run")
def run_lab(body: LabRunRequest):
    ground_truth: dict = {}

    # Resolve transcript: case_id takes precedence, then inline transcript
    if body.case_id:
        case = get_case(body.case_id) or get_public_case(body.case_id)
        if not case:
            raise HTTPException(404, f"Case '{body.case_id}' not found.")
        transcript = case.transcript
        ground_truth = case.ground_truth
    elif body.transcript:
        transcript = body.transcript
    else:
        raise HTTPException(400, "Provide transcript or case_id.")

    cfg = lm_config.load()

    try:
        result = run_pipeline(
            transcript=transcript,
            domain_name=body.domain,
            cfg=cfg,
            knowledge_base=body.knowledge_base,
            system_prompt=body.system_prompt,
            template_prompt=body.template_prompt,
            workflow_override=body.workflow_override.model_dump() if body.workflow_override else None,
        )
    except Exception as exc:
        raise HTTPException(503, f"Pipeline failed: {exc}")

    try:
        naive_summary = run_naive(transcript, body.domain, cfg)
    except Exception:
        naive_summary = ""

    result["naive_summary"] = naive_summary
    result["eval"] = _compute_eval(result, naive_summary, transcript, ground_truth)
    result["run_label"] = body.run_label
    if body.case_id:
        result["case_id"] = body.case_id
        result["ground_truth"] = ground_truth
    run_id = _save_history(result)
    result["history_id"] = run_id
    return result


# ---------------------------------------------------------------------------
# Audio transcription (lab-only, no DB write)
# ---------------------------------------------------------------------------

@router.post("/transcribe")
async def lab_transcribe(file: UploadFile = File(...)):
    """
    Transcribe an audio file for the lab. Synchronous — may take up to 10 minutes
    for long recordings. No DB writes, temp file cleaned up after response.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO:
        raise HTTPException(400, f"Unsupported type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO))}")

    cfg = wcfg.load()
    try:
        _find_binary(cfg)
        _find_model(cfg)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Whisper not configured: {exc}")

    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        async with aiofiles.open(tmp_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                await out.write(chunk)

        transcription = run_whisper_file(tmp_path, cfg)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Transcription failed: {exc}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "transcript": transcription["full_text"],
        "segments": transcription["segments"],
        "model_used": transcription["model_used"],
    }


# ---------------------------------------------------------------------------
# Audio library (domain subfolders under tests/audio/)
# ---------------------------------------------------------------------------

@router.get("/audio-library")
def audio_library():
    """List audio files organised by domain subfolder."""
    result: dict[str, list[dict]] = {}
    for domain in DOMAINS:
        domain_dir = os.path.join(AUDIO_LIBRARY_DIR, domain)
        if not os.path.isdir(domain_dir):
            continue
        files = []
        for fn in sorted(os.listdir(domain_dir)):
            if os.path.splitext(fn)[1].lower() not in ALLOWED_AUDIO:
                continue
            fp = os.path.join(domain_dir, fn)
            files.append({
                "filename": fn,
                "domain": domain,
                "size_bytes": os.path.getsize(fp),
            })
        if files:
            result[domain] = files
    return result


# ---------------------------------------------------------------------------
# Batch item: transcribe + pipeline for one audio file from the library
# ---------------------------------------------------------------------------

class BatchItemRequest(BaseModel):
    domain: str
    filename: str
    workflow_override: WorkflowOverride | None = None


@router.post("/batch-item")
def run_batch_item(body: BatchItemRequest):
    """
    Transcribe a single audio file from the library, run the full pipeline,
    and return metrics. Called once per file by the frontend batch loop.
    """
    audio_path = os.path.join(AUDIO_LIBRARY_DIR, body.domain, body.filename)
    if not os.path.isfile(audio_path):
        raise HTTPException(404, f"Audio file not found: {body.domain}/{body.filename}")

    # Transcribe
    whisper_cfg = wcfg.load()
    try:
        transcription = run_whisper_file(audio_path, whisper_cfg)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Whisper not configured: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Transcription failed: {exc}")

    transcript = transcription["full_text"]
    lm_cfg = lm_config.load()

    try:
        result = run_pipeline(
            transcript=transcript,
            domain_name=body.domain,
            cfg=lm_cfg,
            workflow_override=body.workflow_override.model_dump() if body.workflow_override else None,
        )
    except Exception as exc:
        raise HTTPException(503, f"Pipeline failed: {exc}")

    try:
        naive_summary = run_naive(transcript, body.domain, lm_cfg)
    except Exception:
        naive_summary = ""

    result["naive_summary"] = naive_summary
    result["eval"] = _compute_eval(result, naive_summary, transcript, {})
    result["filename"] = body.filename
    result["transcript"] = transcript
    result["transcript_chars"] = len(transcript)
    result["whisper_model"] = transcription["model_used"]
    result["segments"] = transcription["segments"]
    return result
