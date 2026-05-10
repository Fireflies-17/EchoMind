from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests

from .utils import compact_text, load_dotenv_if_available, load_json, save_json


ACTION_RE = re.compile(r"(需要|待办|负责|跟进|确认|完成|计划|下周|明天|今天|截止|deadline|todo)", re.I)
TAG_CANDIDATES = [
    "预算",
    "风险",
    "方案",
    "时间",
    "进度",
    "问题",
    "结论",
    "数据",
    "模型",
    "部署",
    "测试",
    "成本",
]


def build_prompt(timeline: list[dict[str, Any]]) -> str:
    lines = []
    for segment in timeline:
        start = int(segment["start_ms"])
        end = int(segment["end_ms"])
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "")
        lines.append(f"[{start}-{end}] {speaker}: {text}")
    content = "\n".join(lines)
    return f"""你是会议纪要与视频知识点分析助手。

下面是带毫秒时间戳和说话人的转写文本。请输出严格 JSON，不要输出 Markdown。

输出结构：
{{
  "overall_summary": "...",
  "chapters": [
    {{"title": "...", "summary": "...", "start_ms": 0, "end_ms": 0}}
  ],
  "knowledge_points": [
    {{
      "title": "...",
      "summary": "...",
      "evidence_text": "...",
      "start_ms": 0,
      "end_ms": 0,
      "speaker": "...",
      "tags": ["..."]
    }}
  ],
  "action_items": [
    {{
      "task": "...",
      "owner": null,
      "deadline": null,
      "evidence_text": "...",
      "start_ms": 0,
      "end_ms": 0
    }}
  ]
}}

要求：
- 不要编造转写文本中没有的信息。
- 时间点必须来自原文片段。
- 知识点要短、准、可检索。

转写文本：
{content}
"""


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def call_openai_compatible(prompt: str) -> dict[str, Any] | None:
    load_dotenv_if_available()
    base_url = (
        os.getenv("LLM_CHAT_COMPLETIONS_URL")
        or os.getenv("LLM_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
    )
    model = os.getenv("LLM_MODEL") or os.getenv("DASHSCOPE_MODEL")
    if not base_url or not model:
        return None
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你只输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    url = _chat_completions_url(base_url)
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400 and "response_format" in payload:
        payload.pop("response_format", None)
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    return _extract_json(content)


def _title_from_text(text: str, fallback: str) -> str:
    clean = re.sub(r"\s+", "", text)
    clean = re.sub(r"[，。！？；：,.!?;:]+", " ", clean).strip()
    return compact_text(clean or fallback, 28)


def _tags_for_text(text: str) -> list[str]:
    tags = [tag for tag in TAG_CANDIDATES if tag in text]
    ascii_terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
    for term in ascii_terms[:3]:
        if term not in tags:
            tags.append(term)
    return tags[:5] or ["视频"]


def _chapter_groups(timeline: list[dict[str, Any]], target_ms: int = 5 * 60 * 1000):
    if not timeline:
        return []
    groups = []
    current = []
    group_start = int(timeline[0]["start_ms"])
    for segment in timeline:
        if current and int(segment["end_ms"]) - group_start > target_ms:
            groups.append(current)
            current = []
            group_start = int(segment["start_ms"])
        current.append(segment)
    if current:
        groups.append(current)
    return groups


def heuristic_summary(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    if not timeline:
        return {
            "overall_summary": "未识别到可用转写内容。",
            "chapters": [],
            "knowledge_points": [],
            "action_items": [],
            "meta": {"summary_engine": "heuristic"},
        }

    full_text = " ".join(str(item.get("text", "")) for item in timeline)
    first_points = "；".join(compact_text(item.get("text", ""), 80) for item in timeline[:4])
    overall = f"本视频共转写 {len(timeline)} 个时间段，主要内容包括：{compact_text(first_points or full_text, 360)}"

    chapters = []
    for idx, group in enumerate(_chapter_groups(timeline), start=1):
        group_text = " ".join(item.get("text", "") for item in group)
        chapters.append(
            {
                "title": _title_from_text(group[0].get("text", ""), f"章节 {idx}"),
                "summary": compact_text(group_text, 260),
                "start_ms": int(group[0]["start_ms"]),
                "end_ms": int(group[-1]["end_ms"]),
            }
        )

    knowledge_points = []
    for segment in timeline:
        text = str(segment.get("text", "")).strip()
        if len(text) < 10:
            continue
        knowledge_points.append(
            {
                "title": _title_from_text(text, "知识点"),
                "summary": compact_text(text, 180),
                "evidence_text": text,
                "start_ms": int(segment["start_ms"]),
                "end_ms": int(segment["end_ms"]),
                "speaker": segment.get("speaker", "UNKNOWN"),
                "tags": _tags_for_text(text),
            }
        )
        if len(knowledge_points) >= 80:
            break

    action_items = []
    for segment in timeline:
        text = str(segment.get("text", "")).strip()
        if not ACTION_RE.search(text):
            continue
        action_items.append(
            {
                "task": compact_text(text, 120),
                "owner": None,
                "deadline": None,
                "evidence_text": text,
                "start_ms": int(segment["start_ms"]),
                "end_ms": int(segment["end_ms"]),
            }
        )

    return {
        "overall_summary": overall,
        "chapters": chapters,
        "knowledge_points": knowledge_points,
        "action_items": action_items,
        "meta": {"summary_engine": "heuristic"},
    }


def normalize_summary(data: dict[str, Any], engine: str) -> dict[str, Any]:
    normalized = {
        "overall_summary": data.get("overall_summary", ""),
        "chapters": data.get("chapters", []),
        "knowledge_points": data.get("knowledge_points", []),
        "action_items": data.get("action_items", []),
        "meta": data.get("meta", {}),
    }
    normalized["meta"]["summary_engine"] = engine
    return normalized


def summarize_timeline(
    timeline_path: str | Path,
    output_path: str | Path,
    engine: str = "auto",
    prompt_output_path: str | Path | None = None,
) -> dict[str, Any]:
    timeline = load_json(timeline_path)
    prompt = build_prompt(timeline)
    if prompt_output_path:
        Path(prompt_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(prompt_output_path).write_text(prompt, encoding="utf-8")

    if engine not in {"auto", "llm", "heuristic"}:
        raise ValueError("engine must be one of: auto, llm, heuristic")

    if engine in {"auto", "llm"}:
        try:
            llm_result = call_openai_compatible(prompt)
            if llm_result is not None:
                summary = normalize_summary(llm_result, engine="llm")
                save_json(summary, output_path)
                return summary
            if engine == "llm":
                raise RuntimeError("LLM_BASE_URL and LLM_MODEL are required when engine=llm.")
        except Exception:
            if engine == "llm":
                raise

    summary = heuristic_summary(timeline)
    save_json(summary, output_path)
    return summary
