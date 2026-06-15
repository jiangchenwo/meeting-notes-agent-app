"""
Standalone whisper-cli runner for the Agent Testing Lab.

Accepts a cfg dict (shape of whisper_config.load()) so it can be called
without touching the production transcribe.py router.
"""
import json
import os
import subprocess
import tempfile


def _find_binary(cfg: dict) -> str:
    binary_path = cfg.get("binary_path", "")
    if binary_path:
        candidate = os.path.join(binary_path, "whisper-cli")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        f"whisper-cli not found in binary_path={binary_path!r}. "
        "Configure it in Settings → Whisper."
    )


def _find_model(cfg: dict) -> str:
    model_path = cfg.get("model_path", "")
    if model_path and os.path.isfile(model_path):
        return model_path
    binary_path = cfg.get("binary_path", "")
    model = cfg.get("model", "base")
    if binary_path:
        # whisper.cpp layout: build/bin/ → ../../models/
        whisper_root = os.path.abspath(os.path.join(binary_path, "..", ".."))
        models_dir = os.path.join(whisper_root, "models")
        for name in [f"ggml-{model}.bin", f"ggml-{model}.en.bin"]:
            path = os.path.join(models_dir, name)
            if os.path.isfile(path):
                return path
    raise FileNotFoundError(
        f"Model '{model}' not found. Set model_path in Settings → Whisper."
    )


def run_whisper_file(audio_path: str, cfg: dict) -> dict:
    """
    Run whisper-cli synchronously on a local audio file.

    Args:
        audio_path: Absolute path to the audio file.
        cfg: whisper_config.load() dict — keys: binary_path, model, model_path.

    Returns:
        {"full_text": str, "segments": [{"start": float, "end": float, "text": str}],
         "model_used": str}

    Raises:
        FileNotFoundError: if binary or model cannot be located.
        RuntimeError: if whisper-cli exits non-zero or produces no output file.
        subprocess.TimeoutExpired: if transcription takes > 10 minutes.
    """
    whisper_bin = _find_binary(cfg)
    model_path = _find_model(cfg)
    model_name = cfg.get("model", "base")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = os.path.join(tmpdir, "out")
        cmd = [
            whisper_bin,
            "-m", model_path,
            "-f", audio_path,
            "--output-json",
            "-of", out_prefix,
            "--no-prints",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        out_json = out_prefix + ".json"
        if proc.returncode != 0 or not os.path.isfile(out_json):
            raise RuntimeError(
                f"whisper-cli exited {proc.returncode}: {proc.stderr[:400]}"
            )
        with open(out_json) as f:
            data = json.load(f)

    raw_segs = data.get("transcription", [])
    full_text = " ".join(s.get("text", "").strip() for s in raw_segs).strip()
    segments = [
        {
            "start": s.get("offsets", {}).get("from", 0) / 1000.0,
            "end":   s.get("offsets", {}).get("to",   0) / 1000.0,
            "text":  s.get("text", "").strip(),
        }
        for s in raw_segs
        if s.get("text", "").strip()
    ]
    return {"full_text": full_text, "segments": segments, "model_used": model_name}
