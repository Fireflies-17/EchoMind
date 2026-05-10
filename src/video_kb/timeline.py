from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import load_json, save_json


def overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _segments_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("segments", [])
    return []


def _guess_speaker(
    segment: dict[str, Any],
    speaker_segments: list[dict[str, Any]],
) -> tuple[str, float]:
    best_speaker = "UNKNOWN"
    best_overlap = 0
    start = int(segment["start_ms"])
    end = int(segment["end_ms"])
    duration = max(1, end - start)
    for speaker_segment in speaker_segments:
        current = overlap_ms(
            start,
            end,
            int(speaker_segment["start_ms"]),
            int(speaker_segment["end_ms"]),
        )
        if current > best_overlap:
            best_overlap = current
            best_speaker = str(speaker_segment.get("speaker", "UNKNOWN"))
    return best_speaker, round(best_overlap / duration, 4)


def merge_adjacent_segments(
    timeline: list[dict[str, Any]],
    max_gap_ms: int = 1200,
    max_chars: int = 500,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for segment in timeline:
        if not merged:
            merged.append(dict(segment))
            continue
        previous = merged[-1]
        same_speaker = previous.get("speaker") == segment.get("speaker")
        gap = int(segment["start_ms"]) - int(previous["end_ms"])
        combined_len = len(previous.get("text", "")) + len(segment.get("text", ""))
        if same_speaker and 0 <= gap <= max_gap_ms and combined_len <= max_chars:
            previous["end_ms"] = segment["end_ms"]
            previous["text"] = f"{previous.get('text', '').rstrip()} {segment.get('text', '').lstrip()}".strip()
            previous["source_asr_ids"] = previous.get("source_asr_ids", []) + segment.get("source_asr_ids", [])
            previous["speaker_overlap_ratio"] = min(
                float(previous.get("speaker_overlap_ratio", 0)),
                float(segment.get("speaker_overlap_ratio", 0)),
            )
        else:
            merged.append(dict(segment))
    return merged


def merge_timeline(
    asr_path: str | Path,
    speakers_path: str | Path,
    output_path: str | Path,
    merge_adjacent: bool = True,
) -> list[dict[str, Any]]:
    asr_segments = _segments_from_payload(load_json(asr_path))
    speaker_payload = load_json(speakers_path) if Path(speakers_path).exists() else {"segments": []}
    speaker_segments = _segments_from_payload(speaker_payload)

    timeline: list[dict[str, Any]] = []
    for segment in asr_segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        speaker, ratio = _guess_speaker(segment, speaker_segments)
        timeline.append(
            {
                "start_ms": int(segment["start_ms"]),
                "end_ms": int(segment["end_ms"]),
                "speaker": speaker,
                "speaker_overlap_ratio": ratio,
                "text": text,
                "source_asr_ids": [segment.get("id", segment.get("source_segment_id"))],
            }
        )
    timeline.sort(key=lambda item: (item["start_ms"], item["end_ms"]))
    if merge_adjacent:
        timeline = merge_adjacent_segments(timeline)
    save_json(timeline, output_path)
    return timeline

