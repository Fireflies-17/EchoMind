from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .summarize import call_openai_compatible
from .utils import compact_text, ensure_parent, format_ms, load_json, save_json


TOPIC_BREAK_RE = re.compile(
    r"(接下来|下一个|另一个|另外|第二个|第三个|关于.+?项目|关于.+?方案|"
    r"我这次要提出来|我想的是|我的方案|这个方案|这个项目|大家发言|总结)"
)


def _clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _title_from_text(text: str, fallback: str) -> str:
    clean = re.sub(r"\s+", "", text)
    clean = re.sub(r"[，。！？；：,.!?;:]+", " ", clean).strip()
    return compact_text(clean or fallback, 32)


def _load_timeline(timeline_path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(timeline_path)
    if not isinstance(payload, list):
        raise ValueError("timeline must be a JSON list.")

    timeline = []
    for idx, item in enumerate(payload):
        text = _clean_text(item.get("text", ""))
        if not text:
            continue
        start = _safe_int(item.get("start_ms"))
        end = _safe_int(item.get("end_ms"))
        if end <= start:
            continue
        timeline.append(
            {
                "source_index": idx,
                "start_ms": start,
                "end_ms": end,
                "speaker": str(item.get("speaker") or "UNKNOWN"),
                "text": text,
            }
        )
    return sorted(timeline, key=lambda item: (item["start_ms"], item["end_ms"]))


def _public_segment(segment: dict[str, Any], index: int | None = None) -> dict[str, Any]:
    public = {
        "speaker": segment["speaker"],
        "start_ms": int(segment["start_ms"]),
        "end_ms": int(segment["end_ms"]),
        "start_time": format_ms(segment["start_ms"]),
        "end_time": format_ms(segment["end_ms"]),
        "text": segment["text"],
        "summary": segment.get("summary") or compact_text(segment["text"], 160),
    }
    if index is not None:
        public = {"index": index, **public}
    if "source_indices" in segment:
        public["source_indices"] = segment["source_indices"]
    elif "source_index" in segment:
        public["source_indices"] = [segment["source_index"]]
    return public


def group_by_speaker_turn(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for segment in timeline:
        if current is not None and current["speaker"] == segment["speaker"]:
            current["end_ms"] = max(int(current["end_ms"]), int(segment["end_ms"]))
            current["text"] = f"{current['text'].rstrip()} {segment['text'].lstrip()}".strip()
            current["source_indices"].append(segment["source_index"])
            continue
        if current is not None:
            turns.append(current)
        current = {
            "start_ms": int(segment["start_ms"]),
            "end_ms": int(segment["end_ms"]),
            "speaker": segment["speaker"],
            "text": segment["text"],
            "source_indices": [segment["source_index"]],
        }
    if current is not None:
        turns.append(current)
    for idx, turn in enumerate(turns):
        turn["index"] = idx
    return turns


def _llm_json(prompt: str, engine: str) -> dict[str, Any] | None:
    if engine == "heuristic":
        return None
    try:
        result = call_openai_compatible(prompt)
        if result is None and engine == "llm":
            raise RuntimeError("LLM endpoint is not configured.")
        return result
    except Exception:
        if engine == "llm":
            raise
        return None


def _apply_turn_summaries(turns: list[dict[str, Any]], engine: str) -> None:
    for turn in turns:
        turn["summary"] = compact_text(turn["text"], 160)

    if not turns or engine == "heuristic":
        return

    lines = "\n".join(
        f'{turn["index"]}. [{format_ms(turn["start_ms"])}-{format_ms(turn["end_ms"])}] '
        f'{turn["speaker"]}: {compact_text(turn["text"], 600)}'
        for turn in turns
    )
    prompt = f"""你是中文视频转写摘要助手。请为每个发言片段生成一句短摘要。

输出严格 JSON，不要输出 Markdown。格式：
{{
  "segments": [
    {{"index": 0, "summary": "..."}}
  ]
}}

要求：
- index 必须使用输入中的编号。
- summary 不要超过 60 个中文字符。
- 不要编造输入中没有的信息。

发言片段：
{lines}
"""
    data = _llm_json(prompt, engine)
    if not data:
        return
    by_index = {int(turn["index"]): turn for turn in turns}
    for item in data.get("segments", []):
        idx = _safe_int(item.get("index"), -1)
        if idx in by_index:
            summary = _clean_text(item.get("summary"))
            if summary:
                by_index[idx]["summary"] = summary


def build_full_report(
    timeline_path: str | Path,
    engine: str = "auto",
) -> dict[str, Any]:
    timeline = _load_timeline(timeline_path)
    turns = group_by_speaker_turn(timeline)
    _apply_turn_summaries(turns, engine)
    return {
        "mode": "full",
        "source_timeline": str(timeline_path),
        "segments": [_public_segment(turn, index=idx) for idx, turn in enumerate(turns)],
        "meta": {
            "summary_engine": engine,
            "segment_count": len(turns),
            "base_timeline_segments": len(timeline),
        },
    }


def _normalize_speaker_query(speaker: str) -> set[str]:
    raw = speaker.strip()
    values = {raw, raw.upper()}
    if raw.isdigit():
        values.add(f"SPEAKER_{int(raw):02d}")
        values.add(f"SPEAKER_{int(raw)}")
    if raw.upper().startswith("SPEAKER_"):
        suffix = raw.split("_", 1)[-1]
        if suffix.isdigit():
            values.add(f"SPEAKER_{int(suffix):02d}")
            values.add(f"SPEAKER_{int(suffix)}")
    return values


def build_speaker_report(
    timeline_path: str | Path,
    speaker: str,
    engine: str = "auto",
) -> dict[str, Any]:
    timeline = _load_timeline(timeline_path)
    turns = group_by_speaker_turn(timeline)
    allowed = _normalize_speaker_query(speaker)
    speaker_turns = [turn for turn in turns if turn["speaker"] in allowed or turn["speaker"].upper() in allowed]
    _apply_turn_summaries(speaker_turns, engine)
    total_ms = sum(max(0, int(turn["end_ms"]) - int(turn["start_ms"])) for turn in speaker_turns)
    speaker_label = speaker_turns[0]["speaker"] if speaker_turns else speaker
    return {
        "mode": "speaker",
        "source_timeline": str(timeline_path),
        "speaker": speaker_label,
        "total_ms": total_ms,
        "total_time": format_ms(total_ms),
        "segments": [_public_segment(turn, index=idx) for idx, turn in enumerate(speaker_turns)],
        "meta": {
            "summary_engine": engine,
            "segment_count": len(speaker_turns),
            "base_timeline_segments": len(timeline),
        },
    }


def _topic_text(topic_segments: list[dict[str, Any]]) -> str:
    return " ".join(segment["text"] for segment in topic_segments)


def _speaker_blocks_for_topic(
    topic_segments: list[dict[str, Any]],
    speaker_summaries: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for segment in topic_segments:
        by_speaker.setdefault(segment["speaker"], []).append(segment)

    blocks = []
    for speaker, segments in sorted(by_speaker.items()):
        text = _topic_text(segments)
        blocks.append(
            {
                "speaker": speaker,
                "summary": (speaker_summaries or {}).get(speaker) or compact_text(text, 180),
                "segments": [_public_segment(segment, index=segment["index"]) for segment in segments],
            }
        )
    return blocks


def _topic_from_segments(
    topic_id: int,
    segments: list[dict[str, Any]],
    title: str | None = None,
    summary: str | None = None,
    speaker_summaries: dict[str, str] | None = None,
) -> dict[str, Any]:
    text = _topic_text(segments)
    return {
        "topic_id": topic_id,
        "title": title or _title_from_text(segments[0]["text"], f"议题 {topic_id}"),
        "summary": summary or compact_text(text, 260),
        "start_ms": int(segments[0]["start_ms"]),
        "end_ms": int(segments[-1]["end_ms"]),
        "start_time": format_ms(segments[0]["start_ms"]),
        "end_time": format_ms(segments[-1]["end_ms"]),
        "speakers": _speaker_blocks_for_topic(segments, speaker_summaries=speaker_summaries),
    }


def _heuristic_topic_groups(
    turns: list[dict[str, Any]],
    topic_gap_ms: int,
    topic_window_ms: int,
    min_topic_ms: int,
) -> list[list[dict[str, Any]]]:
    if not turns:
        return []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for turn in turns:
        if not current:
            current.append(turn)
            continue

        previous = current[-1]
        gap = int(turn["start_ms"]) - int(previous["end_ms"])
        duration = int(previous["end_ms"]) - int(current[0]["start_ms"])
        text = turn["text"]
        starts_new_topic = (
            gap > topic_gap_ms
            or duration > topic_window_ms
            or (duration >= min_topic_ms and len(current) >= 2 and TOPIC_BREAK_RE.search(text))
        )
        if starts_new_topic:
            groups.append(current)
            current = [turn]
        else:
            current.append(turn)
    if current:
        groups.append(current)
    return groups


def _normalize_indices(indices: Any, max_len: int) -> list[int]:
    if not isinstance(indices, list):
        return []
    values = []
    for value in indices:
        idx = _safe_int(value, -1)
        if idx >= 0:
            values.append(idx)
    if values and 0 not in values and max(values) <= max_len:
        maybe_one_based = [idx - 1 for idx in values]
        if all(0 <= idx < max_len for idx in maybe_one_based):
            values = maybe_one_based
    return sorted({idx for idx in values if 0 <= idx < max_len})


def _llm_topics(turns: list[dict[str, Any]], engine: str, max_llm_chars: int) -> list[dict[str, Any]] | None:
    if engine == "heuristic" or not turns:
        return None

    lines = []
    total_chars = 0
    for turn in turns:
        line = (
            f'{turn["index"]}. [{format_ms(turn["start_ms"])}-{format_ms(turn["end_ms"])}] '
            f'{turn["speaker"]}: {compact_text(turn["text"], 500)}'
        )
        total_chars += len(line)
        if total_chars > max_llm_chars:
            break
        lines.append(line)

    prompt = f"""你是中文会议和视频内容分析助手。请根据语义把发言片段切分成不同议题。

输出严格 JSON，不要输出 Markdown。格式：
{{
  "topics": [
    {{
      "title": "...",
      "summary": "...",
      "segment_indices": [0, 1],
      "speaker_summaries": [
        {{"speaker": "SPEAKER_00", "summary": "...", "segment_indices": [0]}}
      ]
    }}
  ]
}}

要求：
- segment_indices 必须使用输入中的编号。
- 每个议题应尽量连续，除非语义上确实需要合并。
- 每个 speaker_summaries 只总结该说话人在该议题下的发言。
- 不要编造输入中没有的信息。

发言片段：
{chr(10).join(lines)}
"""
    data = _llm_json(prompt, engine)
    if not data:
        return None

    topics = []
    used_indices: set[int] = set()
    for item in data.get("topics", []):
        indices = _normalize_indices(item.get("segment_indices"), len(turns))
        if not indices:
            continue
        topic_segments = [turns[idx] for idx in indices]
        speaker_summaries = {}
        for speaker_item in item.get("speaker_summaries", []):
            speaker = _clean_text(speaker_item.get("speaker"))
            summary = _clean_text(speaker_item.get("summary"))
            if speaker and summary:
                speaker_summaries[speaker] = summary
        topics.append(
            _topic_from_segments(
                len(topics) + 1,
                topic_segments,
                title=_clean_text(item.get("title")) or None,
                summary=_clean_text(item.get("summary")) or None,
                speaker_summaries=speaker_summaries,
            )
        )
        used_indices.update(indices)

    missing = [turn for turn in turns if turn["index"] not in used_indices]
    if missing:
        topics.append(_topic_from_segments(len(topics) + 1, missing, title="未归类片段"))
    return topics or None


def build_topics_report(
    timeline_path: str | Path,
    engine: str = "auto",
    topic_gap_ms: int = 45_000,
    topic_window_ms: int = 5 * 60 * 1000,
    min_topic_ms: int = 45_000,
    max_llm_chars: int = 24_000,
) -> dict[str, Any]:
    timeline = _load_timeline(timeline_path)
    turns = group_by_speaker_turn(timeline)
    _apply_turn_summaries(turns, "heuristic")

    topics = _llm_topics(turns, engine=engine, max_llm_chars=max_llm_chars)
    topic_engine = "llm" if topics else "heuristic"
    if topics is None:
        groups = _heuristic_topic_groups(
            turns,
            topic_gap_ms=topic_gap_ms,
            topic_window_ms=topic_window_ms,
            min_topic_ms=min_topic_ms,
        )
        topics = [_topic_from_segments(idx, group) for idx, group in enumerate(groups, start=1)]

    return {
        "mode": "topics",
        "source_timeline": str(timeline_path),
        "topics": topics,
        "meta": {
            "summary_engine": topic_engine,
            "topic_count": len(topics),
            "base_timeline_segments": len(timeline),
            "speaker_turn_count": len(turns),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    mode = report.get("mode")
    lines: list[str] = []
    if mode == "full":
        lines.append("# 整体转译报告")
        for segment in report.get("segments", []):
            lines.append("")
            lines.append(f"## {segment['speaker']} {segment['start_time']} - {segment['end_time']}")
            lines.append("")
            lines.append(f"摘要：{segment['summary']}")
            lines.append("")
            lines.append(segment["text"])
    elif mode == "speaker":
        lines.append(f"# 说话人报告：{report.get('speaker')}")
        lines.append("")
        lines.append(f"累计发言时长：{report.get('total_time')}")
        for segment in report.get("segments", []):
            lines.append("")
            lines.append(f"## {segment['start_time']} - {segment['end_time']}")
            lines.append("")
            lines.append(f"摘要：{segment['summary']}")
            lines.append("")
            lines.append(segment["text"])
    elif mode == "topics":
        lines.append("# 议题报告")
        for topic in report.get("topics", []):
            lines.append("")
            lines.append(f"## 议题 {topic['topic_id']}：{topic['title']}")
            lines.append("")
            lines.append(f"时间：{topic['start_time']} - {topic['end_time']}")
            lines.append("")
            lines.append(f"摘要：{topic['summary']}")
            for speaker in topic.get("speakers", []):
                lines.append("")
                lines.append(f"### {speaker['speaker']}")
                lines.append("")
                lines.append(f"发言摘要：{speaker['summary']}")
                for segment in speaker.get("segments", []):
                    lines.append("")
                    lines.append(f"- {segment['start_time']} - {segment['end_time']}：{segment['text']}")
    else:
        raise ValueError(f"Unsupported report mode: {mode}")
    return "\n".join(lines).rstrip() + "\n"


def write_report(
    timeline_path: str | Path,
    output_path: str | Path,
    mode: str,
    speaker: str | None = None,
    engine: str = "auto",
    output_format: str = "json",
    topic_gap_ms: int = 45_000,
    topic_window_ms: int = 5 * 60 * 1000,
    min_topic_ms: int = 45_000,
    max_llm_chars: int = 24_000,
) -> dict[str, Any]:
    if mode == "full":
        report = build_full_report(timeline_path, engine=engine)
    elif mode == "speaker":
        if not speaker:
            raise ValueError("--speaker is required when --mode speaker.")
        report = build_speaker_report(timeline_path, speaker=speaker, engine=engine)
    elif mode == "topics":
        report = build_topics_report(
            timeline_path,
            engine=engine,
            topic_gap_ms=topic_gap_ms,
            topic_window_ms=topic_window_ms,
            min_topic_ms=min_topic_ms,
            max_llm_chars=max_llm_chars,
        )
    else:
        raise ValueError("mode must be one of: full, speaker, topics")

    if output_format == "json":
        save_json(report, output_path)
    elif output_format == "md":
        ensure_parent(output_path)
        Path(output_path).write_text(render_markdown(report), encoding="utf-8")
    else:
        raise ValueError("format must be one of: json, md")
    return report
