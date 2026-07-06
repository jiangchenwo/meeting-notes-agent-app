import os
import tempfile

from fastapi import Depends, FastAPI, Form, UploadFile

from asr_service.config import load_settings
from asr_service.engine import Engine, MacMetalEngine
from asr_service.merge import assign_speakers

app = FastAPI(title="asr-service")

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = MacMetalEngine(load_settings())
    return _engine


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _engine is not None}


def _duration_ms(segments: list[dict]) -> int | None:
    return int(segments[-1]["end"] * 1000) if segments else None


@app.post("/transcribe")
async def transcribe(
    audio_file: UploadFile,
    diarize: str = Form("false"),
    language: str | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
    engine: Engine = Depends(get_engine),
):
    do_diarize = diarize.lower() == "true"
    suffix = os.path.splitext(audio_file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await audio_file.read())
        tmp.flush()
        result = engine.transcribe(tmp.name, language)
        segments = result.segments
        diarized = False
        if do_diarize:
            try:
                turns = engine.diarize(tmp.name, min_speakers, max_speakers)
                segments = assign_speakers(segments, turns)
                diarized = True
            except Exception:
                diarized = False  # graceful degradation

    return {
        "full_text": result.text,
        "language": result.language,
        "model_used": result.model_used,
        "duration_ms": _duration_ms(segments),
        "diarized": diarized,
        "segments": segments,
    }
