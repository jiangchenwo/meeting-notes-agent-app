"""Lightweight audio metadata helpers."""
import shutil
import subprocess
from typing import Optional


def probe_duration_ms(audio_path: str) -> Optional[int]:
    """Return the audio duration in milliseconds via ffprobe, or None if unavailable.

    ffprobe reads only container metadata, so this is fast regardless of file size.
    Returns None when ffprobe is not installed or the file has no readable duration.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=30,
        )
        return int(float(proc.stdout.strip()) * 1000)
    except (ValueError, subprocess.SubprocessError):
        return None
