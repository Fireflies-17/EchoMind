from __future__ import annotations

import subprocess
from pathlib import Path

from .utils import DependencyError, ensure_parent, require_executable


def extract_audio(
    input_path: str | Path,
    output_path: str | Path,
    ffmpeg_bin: str = "ffmpeg",
    sample_rate: int = 16000,
) -> Path:
    source = Path(input_path)
    target = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(f"Input media does not exist: {source}")
    ensure_parent(target)
    ffmpeg = require_executable(ffmpeg_bin)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-3000:]
        raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}:\n{detail}")
    return target


def audio_duration_ms(audio_path: str | Path) -> int:
    try:
        import soundfile as sf
    except Exception as exc:
        raise DependencyError("soundfile is required to inspect WAV duration.") from exc

    info = sf.info(str(audio_path))
    return int(info.frames * 1000 / info.samplerate)


def slice_wav(
    audio_path: str | Path,
    output_path: str | Path,
    start_ms: int,
    end_ms: int,
) -> Path:
    try:
        import soundfile as sf
    except Exception as exc:
        raise DependencyError("soundfile is required to slice WAV files.") from exc

    source = Path(audio_path)
    target = Path(output_path)
    ensure_parent(target)
    if end_ms <= start_ms:
        raise ValueError(f"Invalid audio segment: {start_ms}..{end_ms}")

    with sf.SoundFile(str(source)) as wav:
        start_frame = max(0, int(start_ms * wav.samplerate / 1000))
        end_frame = max(start_frame + 1, int(end_ms * wav.samplerate / 1000))
        frames = min(end_frame - start_frame, len(wav) - start_frame)
        wav.seek(start_frame)
        data = wav.read(frames, dtype="float32", always_2d=False)
        sf.write(str(target), data, wav.samplerate)

    return target

