from asr_service.types import SpeakerTurn


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _label_map(turns: list[SpeakerTurn]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for turn in sorted(turns, key=lambda t: t.start):
        if turn.speaker not in mapping:
            mapping[turn.speaker] = f"Speaker {len(mapping) + 1}"
    return mapping


def assign_speakers(segments: list[dict], turns: list[SpeakerTurn]) -> list[dict]:
    mapping = _label_map(turns)
    out: list[dict] = []
    for seg in segments:
        best_turn = None
        best_overlap = 0.0
        for turn in turns:
            ov = _overlap(seg["start"], seg["end"], turn.start, turn.end)
            if ov > best_overlap:
                best_overlap = ov
                best_turn = turn
        speaker = mapping[best_turn.speaker] if best_turn is not None else None
        out.append({**seg, "speaker": speaker})
    return out
