from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .audio import audio_duration_ms
from .utils import DependencyError, load_dotenv_if_available, load_json, resolve_device, save_json, to_jsonable


def _iter_segments(output: Any):
    annotation = getattr(output, "speaker_diarization", output)
    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            yield turn, speaker
        return
    for item in annotation:
        if isinstance(item, tuple) and len(item) == 2:
            yield item[0], item[1]
        elif isinstance(item, tuple) and len(item) >= 3:
            yield item[0], item[-1]


def write_skipped_diarization(
    output_path: str | Path,
    reason: str,
    audio_path: str | Path | None = None,
) -> dict[str, Any]:
    segments = []
    if audio_path is not None:
        try:
            duration = audio_duration_ms(audio_path)
            if duration > 0:
                segments.append({"start_ms": 0, "end_ms": duration, "speaker": "SPEAKER_00"})
        except Exception:
            segments = []
    payload = {
        "skipped": True,
        "reason": reason,
        "segments": segments,
    }
    save_json(payload, output_path)
    return payload


def _normalize_segments(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        source = payload
    elif isinstance(payload, dict):
        source = payload.get("segments", [])
    else:
        source = []

    normalized: list[dict[str, Any]] = []
    for item in source:
        try:
            start = int(item["start_ms"])
            end = int(item["end_ms"])
        except Exception:
            continue
        speaker = str(item.get("speaker", "")).strip()
        if not speaker or end <= start:
            continue
        segment = dict(item)
        segment["start_ms"] = start
        segment["end_ms"] = end
        segment["speaker"] = speaker
        normalized.append(segment)
    return sorted(normalized, key=lambda segment: (segment["start_ms"], segment["end_ms"], segment["speaker"]))


def _duration(segment: dict[str, Any]) -> int:
    return max(0, int(segment["end_ms"]) - int(segment["start_ms"]))


def _speaker_totals(segments: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for segment in segments:
        speaker = str(segment["speaker"])
        totals[speaker] = totals.get(speaker, 0) + _duration(segment)
    return totals


def _merge_same_speaker_segments(
    segments: list[dict[str, Any]],
    max_gap_ms: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda item: (item["start_ms"], item["end_ms"], item["speaker"])):
        if not merged:
            merged.append(dict(segment))
            continue
        previous = merged[-1]
        same_speaker = previous["speaker"] == segment["speaker"]
        gap = int(segment["start_ms"]) - int(previous["end_ms"])
        if same_speaker and gap <= max_gap_ms:
            previous["end_ms"] = max(int(previous["end_ms"]), int(segment["end_ms"]))
            if "absorbed_speakers" in segment:
                previous.setdefault("absorbed_speakers", []).extend(segment["absorbed_speakers"])
            if "original_speaker" in segment:
                previous.setdefault("absorbed_speakers", []).append(segment["original_speaker"])
        else:
            merged.append(dict(segment))
    return merged


def _time_gap_ms(a: dict[str, Any], b: dict[str, Any]) -> int:
    if int(a["end_ms"]) < int(b["start_ms"]):
        return int(b["start_ms"]) - int(a["end_ms"])
    if int(b["end_ms"]) < int(a["start_ms"]):
        return int(a["start_ms"]) - int(b["end_ms"])
    return 0


def _nearest_major_speaker(
    segment: dict[str, Any],
    candidates: list[dict[str, Any]],
    reassign_gap_ms: int,
) -> str | None:
    best: tuple[int, int, str] | None = None
    midpoint = (int(segment["start_ms"]) + int(segment["end_ms"])) // 2
    for candidate in candidates:
        if candidate["speaker"] == segment["speaker"]:
            continue
        gap = _time_gap_ms(segment, candidate)
        if gap > reassign_gap_ms:
            continue
        candidate_midpoint = (int(candidate["start_ms"]) + int(candidate["end_ms"])) // 2
        distance = abs(midpoint - candidate_midpoint)
        current = (gap, distance, str(candidate["speaker"]))
        if best is None or current < best:
            best = current
    return best[2] if best else None


def clean_diarization_payload(
    payload: dict[str, Any] | list[dict[str, Any]],
    min_segment_ms: int = 800,
    merge_gap_ms: int = 500,
    min_total_ms: int = 0,
    max_speakers: int | None = None,
    reassign_gap_ms: int = 3000,
) -> dict[str, Any]:
    """Remove diarization fragments that usually create unstable speaker labels."""
    original_segments = _normalize_segments(payload)
    if isinstance(payload, dict) and payload.get("skipped"):
        cleaned_payload = dict(payload)
        cleaned_payload["segments"] = original_segments
        cleaned_payload["cleaning"] = {
            "skipped": True,
            "reason": "Input diarization was skipped.",
        }
        return cleaned_payload

    kept: list[dict[str, Any]] = []
    dropped_short = 0
    for segment in original_segments:
        if min_segment_ms > 0 and _duration(segment) < min_segment_ms:
            dropped_short += 1
            continue
        kept.append(dict(segment))

    if not kept and original_segments:
        kept = [dict(segment) for segment in original_segments]
        dropped_short = 0

    kept = _merge_same_speaker_segments(kept, max(0, merge_gap_ms))
    totals = _speaker_totals(kept)
    ranked_speakers = sorted(totals, key=lambda speaker: (-totals[speaker], speaker))

    major_speakers = set(ranked_speakers)
    if min_total_ms > 0:
        major_speakers = {speaker for speaker in major_speakers if totals[speaker] >= min_total_ms}
    if max_speakers is not None and max_speakers > 0:
        allowed_by_count = set(ranked_speakers[:max_speakers])
        major_speakers = major_speakers & allowed_by_count if major_speakers else allowed_by_count
    if not major_speakers and ranked_speakers:
        major_speakers = {ranked_speakers[0]}

    major_candidates = [segment for segment in kept if segment["speaker"] in major_speakers]
    cleaned: list[dict[str, Any]] = []
    reassigned = 0
    dropped_minor = 0
    for segment in kept:
        if segment["speaker"] in major_speakers:
            cleaned.append(dict(segment))
            continue
        new_speaker = _nearest_major_speaker(segment, major_candidates, max(0, reassign_gap_ms))
        if new_speaker:
            reassigned_segment = dict(segment)
            reassigned_segment["original_speaker"] = reassigned_segment["speaker"]
            reassigned_segment["speaker"] = new_speaker
            cleaned.append(reassigned_segment)
            reassigned += 1
        else:
            dropped_minor += 1

    cleaned = _merge_same_speaker_segments(cleaned, max(0, merge_gap_ms))
    if not cleaned and kept:
        cleaned = kept
        dropped_minor = 0

    cleaned_totals = _speaker_totals(cleaned)
    cleaned_payload = dict(payload) if isinstance(payload, dict) else {}
    cleaned_payload["segments"] = cleaned
    cleaned_payload["cleaned"] = True
    cleaned_payload["cleaning"] = {
        "min_segment_ms": min_segment_ms,
        "merge_gap_ms": merge_gap_ms,
        "min_total_ms": min_total_ms,
        "max_speakers": max_speakers,
        "reassign_gap_ms": reassign_gap_ms,
        "original_segments": len(original_segments),
        "segments_after_cleaning": len(cleaned),
        "speakers_before": sorted(_speaker_totals(original_segments)),
        "speakers_after": sorted(cleaned_totals),
        "dropped_short_segments": dropped_short,
        "reassigned_segments": reassigned,
        "dropped_minor_segments": dropped_minor,
        "speaker_durations_ms": cleaned_totals,
    }
    return cleaned_payload


def clean_diarization(
    input_path: str | Path,
    output_path: str | Path,
    min_segment_ms: int = 800,
    merge_gap_ms: int = 500,
    min_total_ms: int = 0,
    max_speakers: int | None = None,
    reassign_gap_ms: int = 3000,
) -> dict[str, Any]:
    payload = load_json(input_path)
    cleaned = clean_diarization_payload(
        payload,
        min_segment_ms=min_segment_ms,
        merge_gap_ms=merge_gap_ms,
        min_total_ms=min_total_ms,
        max_speakers=max_speakers,
        reassign_gap_ms=reassign_gap_ms,
    )
    cleaned["source_diarization"] = str(input_path)
    save_json(to_jsonable(cleaned), output_path)
    return cleaned


def diarize(
    audio_path: str | Path,
    output_path: str | Path,
    token: str | None = None,
    model_name: str = "pyannote/speaker-diarization-community-1",
    device: str = "auto",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict[str, Any]:
    load_dotenv_if_available()
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise DependencyError("HF_TOKEN is required for pyannote speaker diarization.")

    try:
        import torch
        from pyannote.audio import Pipeline
    except Exception as exc:
        raise DependencyError("pyannote.audio is required. Install with: pip install -e .[speech]") from exc

    pipeline = Pipeline.from_pretrained(model_name, token=token)
    resolved_device = resolve_device(device)
    if resolved_device != "cpu" and hasattr(pipeline, "to"):
        pipeline.to(torch.device(resolved_device))

    kwargs: dict[str, Any] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    output = pipeline(str(audio_path), **kwargs)

    segments = []
    for turn, speaker in _iter_segments(output):
        segments.append(
            {
                "start_ms": int(float(turn.start) * 1000),
                "end_ms": int(float(turn.end) * 1000),
                "speaker": str(speaker),
            }
        )
    payload = {
        "skipped": False,
        "source_audio": str(audio_path),
        "model": model_name,
        "segments": segments,
        "raw_type": type(output).__name__,
    }
    save_json(to_jsonable(payload), output_path)
    return payload
