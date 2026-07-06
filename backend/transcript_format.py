import json


def build_speaker_transcript(full_text: str | None, segments_json: str | None) -> str:
    """Speaker-labeled transcript from segments; falls back to full_text when no speakers.

    Consecutive segments from the same speaker are grouped into a single
    ``Speaker: ...`` line. When no segment carries a speaker label (e.g. a
    non-diarized transcript) the original ``full_text`` is returned unchanged.
    """
    try:
        segments = json.loads(segments_json or "[]")
    except Exception:
        segments = []

    if not any(s.get("speaker") for s in segments):
        return full_text or ""

    lines: list[str] = []
    prev: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if buf:
            body = " ".join(buf)
            lines.append(f"{prev}: {body}" if prev else body)

    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        spk = s.get("speaker")
        if spk == prev:
            buf.append(text)
        else:
            flush()
            prev, buf = spk, [text]
    flush()

    return "\n".join(lines).strip() or (full_text or "")
