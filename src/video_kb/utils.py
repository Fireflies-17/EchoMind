from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


class DependencyError(RuntimeError):
    pass


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def require_executable(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise DependencyError(f"Cannot find executable '{name}' in PATH.")
    return resolved


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_dotenv_if_available() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv

        explicit_env = os.getenv("VIDEO_KB_ENV_FILE")
        if explicit_env:
            load_dotenv(explicit_env)
            return

        found = find_dotenv(filename=".env", usecwd=True)
        if found:
            load_dotenv(found)
            return

        load_dotenv()
    except Exception:
        return


def slugify(value: str, default: str = "run") -> str:
    value = value.strip()
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value, flags=re.UNICODE)
    value = value.strip("._-")
    return value or default


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def compact_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def format_ms(ms: int | float | None) -> str:
    if ms is None:
        return "00:00.000"
    total_ms = max(0, int(ms))
    seconds, milli = divmod(total_ms, 1000)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minute:02d}:{sec:02d}.{milli:03d}"
    return f"{minute:02d}:{sec:02d}.{milli:03d}"
