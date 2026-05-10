from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .audio import audio_duration_ms, slice_wav
from .utils import DependencyError, resolve_device, save_json, to_jsonable


MODEL_ALIASES = {
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "iic/SenseVoiceSmall": "iic/SenseVoiceSmall",
}


def _cached_model_dir(model_name: str) -> str:
    model_id = MODEL_ALIASES.get(model_name, model_name)
    if "/" not in model_id or Path(model_id).exists():
        return model_name
    group, name = model_id.split("/", 1)
    roots: list[Path] = []
    env_cache = os.getenv("MODELSCOPE_CACHE")
    if env_cache:
        roots.extend([Path(env_cache) / "models", Path(env_cache)])
    roots.extend(
        [
            Path.home() / ".cache" / "modelscope" / "hub" / "models",
            Path.home() / ".cache" / "modelscope" / "hub",
        ]
    )
    for root in roots:
        candidate = root / group / name
        if (candidate / "config.yaml").exists() and (candidate / "model.pt").exists():
            return str(candidate)
    return model_name


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _collect_intervals(value: Any, out: list[tuple[int, int]]) -> None:
    if isinstance(value, dict):
        for key in ["value", "timestamp", "timestamps", "segments"]:
            if key in value:
                _collect_intervals(value[key], out)
        return
    if not isinstance(value, list):
        return
    if value and all(
        isinstance(item, (list, tuple))
        and len(item) >= 2
        and _is_number(item[0])
        and _is_number(item[1])
        for item in value
    ):
        for item in value:
            start = int(item[0])
            end = int(item[1])
            if end > start:
                out.append((start, end))
        return
    for item in value:
        _collect_intervals(item, out)


def parse_vad_intervals(raw_result: Any) -> list[dict[str, int]]:
    intervals: list[tuple[int, int]] = []
    _collect_intervals(raw_result, intervals)
    deduped = sorted(set(intervals))
    return [{"start_ms": start, "end_ms": end} for start, end in deduped]


def split_long_segments(
    segments: list[dict[str, int]],
    max_segment_ms: int = 30000,
) -> list[dict[str, int]]:
    if max_segment_ms <= 0:
        return segments
    split: list[dict[str, int]] = []
    for segment in segments:
        start = int(segment["start_ms"])
        end = int(segment["end_ms"])
        cursor = start
        while cursor < end:
            piece_end = min(end, cursor + max_segment_ms)
            split.append({"start_ms": cursor, "end_ms": piece_end})
            cursor = piece_end
    return split


def detect_vad(
    audio_path: str | Path,
    device: str = "auto",
    model_name: str = "fsmn-vad",
    max_segment_ms: int = 30000,
) -> tuple[list[dict[str, int]], Any]:
    os.environ.setdefault("MODELSCOPE_HUB_FILE_LOCK", "false")
    try:
        from funasr import AutoModel
    except Exception as exc:
        raise DependencyError("funasr is required for VAD. Install with: pip install -e .[speech]") from exc

    resolved_device = resolve_device(device)
    resolved_model = _cached_model_dir(model_name)
    try:
        model = AutoModel(model=resolved_model, device=resolved_device, disable_update=True)
    except TypeError:
        model = AutoModel(model=resolved_model)

    raw = model.generate(input=str(audio_path))
    segments = parse_vad_intervals(raw)
    if not segments:
        duration = audio_duration_ms(audio_path)
        segments = [{"start_ms": 0, "end_ms": duration}]
    return split_long_segments(segments, max_segment_ms=max_segment_ms), raw


def _progress(items: list[dict[str, int]], label: str):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=label)
    except Exception:
        return items


def _postprocess_text(text: str) -> str:
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        return rich_transcription_postprocess(text)
    except Exception:
        return text.strip()


def _extract_text(raw_result: Any) -> str:
    if isinstance(raw_result, str):
        return raw_result.strip()
    if isinstance(raw_result, dict):
        return str(raw_result.get("text", "")).strip()
    if isinstance(raw_result, list):
        texts: list[str] = []
        for item in raw_result:
            if isinstance(item, dict) and item.get("text"):
                texts.append(str(item["text"]))
            elif isinstance(item, str):
                texts.append(item)
        return " ".join(texts).strip()
    return ""


def _build_sensevoice_model(model_name: str, device: str):
    os.environ.setdefault("MODELSCOPE_HUB_FILE_LOCK", "false")
    try:
        from funasr import AutoModel
    except Exception as exc:
        raise DependencyError("funasr is required for ASR. Install with: pip install -e .[speech]") from exc

    resolved_device = resolve_device(device)
    resolved_model = _cached_model_dir(model_name)
    kwargs = {
        "model": resolved_model,
        "trust_remote_code": False,
        "device": resolved_device,
        "disable_update": True,
    }
    try:
        return AutoModel(**kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("disable_update", None)
        return AutoModel(**kwargs)


def _generate_text(model: Any, chunk_path: Path, language: str, use_itn: bool, batch_size_s: int) -> Any:
    kwargs = {
        "input": str(chunk_path),
        "language": language,
        "use_itn": use_itn,
        "batch_size_s": batch_size_s,
    }
    try:
        return model.generate(**kwargs)
    except TypeError:
        kwargs.pop("batch_size_s", None)
        kwargs["batch_size"] = 1
        return model.generate(**kwargs)


def transcribe_segments(
    audio_path: str | Path,
    segments: list[dict[str, int]],
    chunks_dir: str | Path,
    device: str = "auto",
    model_name: str = "iic/SenseVoiceSmall",
    language: str = "zh",
    use_itn: bool = True,
    batch_size_s: int = 60,
) -> list[dict[str, Any]]:
    chunks = Path(chunks_dir)
    chunks.mkdir(parents=True, exist_ok=True)
    model = _build_sensevoice_model(model_name=model_name, device=device)

    output: list[dict[str, Any]] = []
    for idx, segment in enumerate(_progress(segments, "ASR")):
        start_ms = int(segment["start_ms"])
        end_ms = int(segment["end_ms"])
        chunk_path = chunks / f"{idx:05d}_{start_ms}_{end_ms}.wav"
        slice_wav(audio_path, chunk_path, start_ms, end_ms)
        raw = _generate_text(model, chunk_path, language=language, use_itn=use_itn, batch_size_s=batch_size_s)
        text = _postprocess_text(_extract_text(raw))
        if not text:
            continue
        output.append(
            {
                "id": len(output),
                "source_segment_id": idx,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text,
                "chunk_path": str(chunk_path),
                "raw": to_jsonable(raw),
            }
        )
    return output


def run_asr(
    audio_path: str | Path,
    asr_output_path: str | Path,
    vad_output_path: str | Path | None = None,
    chunks_dir: str | Path | None = None,
    device: str = "auto",
    language: str = "zh",
    vad_model: str = "fsmn-vad",
    asr_model: str = "iic/SenseVoiceSmall",
    max_segment_ms: int = 30000,
    batch_size_s: int = 60,
) -> dict[str, Any]:
    audio = Path(audio_path)
    chunks = Path(chunks_dir) if chunks_dir else Path(asr_output_path).parent / "chunks"
    vad_segments, raw_vad = detect_vad(
        audio,
        device=device,
        model_name=vad_model,
        max_segment_ms=max_segment_ms,
    )
    vad_payload = {
        "source_audio": str(audio),
        "model": vad_model,
        "max_segment_ms": max_segment_ms,
        "segments": vad_segments,
        "raw": to_jsonable(raw_vad),
    }
    if vad_output_path:
        save_json(vad_payload, vad_output_path)

    segments = transcribe_segments(
        audio,
        vad_segments,
        chunks,
        device=device,
        model_name=asr_model,
        language=language,
        batch_size_s=batch_size_s,
    )
    payload = {
        "source_audio": str(audio),
        "asr_model": asr_model,
        "vad_model": vad_model,
        "language": language,
        "segments": segments,
    }
    save_json(payload, asr_output_path)
    return payload
