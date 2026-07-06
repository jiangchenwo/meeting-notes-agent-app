import httpx


def transcribe_via_asr(
    audio_bytes: bytes,
    filename: str,
    *,
    diarize: bool,
    language: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    base_url: str,
    timeout: float = 1800.0,
    client: httpx.Client | None = None,
) -> dict:
    files = {"audio_file": (filename, audio_bytes)}
    data: dict[str, str] = {"diarize": "true" if diarize else "false"}
    if language:
        data["language"] = language
    if min_speakers is not None:
        data["min_speakers"] = str(min_speakers)
    if max_speakers is not None:
        data["max_speakers"] = str(max_speakers)

    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        resp = client.post(f"{base_url}/transcribe", files=files, data=data)
        resp.raise_for_status()
        return resp.json()
    finally:
        if owns_client:
            client.close()
