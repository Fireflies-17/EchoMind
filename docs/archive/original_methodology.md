> 归档说明：这是项目早期的 POC 方案草稿，仅作为设计来源保留。当前可运行实现、命令和目录结构以根目录 `README.md`、`docs/current_capabilities.md` 和 `src/video_kb/` 为准。

下面给你一套**中文优先、便捷性和准确率兼顾的本地 POC 部署方案**。目标是先跑通：

**视频/录音 → 中文转写 → 时间段标注 → 说话人分离 → 摘要/知识点/待办 → 可检索知识库**

---

# 1. 推荐总体架构

```text
输入视频/录音
  ↓
FFmpeg / yt-dlp
  ↓
16kHz 单声道 WAV
  ↓
FunASR + SenseVoiceSmall + fsmn-vad
  ↓
中文转写 + VAD 时间段
  ↓
pyannote.audio
  ↓
说话人时间段
  ↓
时间轴合并
  ↓
Qwen3 / 其他中文大模型
  ↓
摘要、知识点、待办、结论
  ↓
Milvus Lite
  ↓
知识库检索 / 问答 / 点击时间点回看
```

这条路径的核心选择是：**转写用 FunASR + SenseVoiceSmall，时间段先用 VAD 片段做稳定锚点，后续再逐步升级到更细的字/词级时间戳**。FunASR 官方覆盖 ASR、VAD、标点、说话人相关能力；SenseVoice 是面向语音理解的多语言模型；pyannote.audio 是开源说话人分离工具；Milvus Lite 适合本地快速搭知识库 POC。([GitHub][1])

---

# 2. 环境准备

建议先用 **Python 3.10 或 3.11**。GPU 不是必须，但如果你处理 30 分钟以上会议，强烈建议用 NVIDIA GPU。

```bash
conda create -n cn-video-summary python=3.10 -y
conda activate cn-video-summary
```

安装基础依赖：

```bash
pip install -U funasr modelscope soundfile librosa pydub tqdm pandas
pip install -U torch torchaudio
pip install -U pyannote.audio
pip install -U pymilvus[milvus-lite]
pip install -U sentence-transformers transformers accelerate
pip install -U yt-dlp ffmpeg-python python-dotenv
```

Milvus 官方推荐本地快速启动时使用 `pymilvus[milvus-lite]`，它会自动安装 Milvus Lite；Milvus Lite 适合 notebook、笔记本电脑、边缘设备等轻量场景。([Milvus][2])

如果后面要做画面 OCR，再装 PaddleOCR。PaddleOCR 3.x 依赖 PaddlePaddle 3.0 或以上版本，安装时要按你的 CUDA/CPU 环境选对应命令。([PaddlePaddle][3])

---

# 3. 推荐目录结构

```text
cn_video_summary/
  data/
    input/              # 原始视频或音频
    audio/              # 转换后的 wav
    transcript/         # ASR 结果
    diarization/        # 说话人分离结果
    summary/            # 摘要和知识点
    vector_db/          # Milvus Lite 本地库
  scripts/
    00_extract_audio.sh
    01_asr_funasr.py
    02_diarize_pyannote.py
    03_merge_timeline.py
    04_summarize.py
    05_build_kb.py
  .env
  requirements.txt
```

---

# 4. 第一步：视频/录音转成标准 WAV

## 工具

**FFmpeg**

## 作用

把各种格式的视频、录音统一转成：

```text
16kHz
单声道
wav
pcm_s16le
```

FFmpeg 官方文档说明它是通用媒体转换工具，可以读取多种输入并转码到多种输出格式。([FFmpeg][4])

## 脚本：`scripts/00_extract_audio.sh`

```bash
#!/usr/bin/env bash

INPUT=$1
OUTPUT=$2

ffmpeg -y \
  -i "$INPUT" \
  -vn \
  -ac 1 \
  -ar 16000 \
  -c:a pcm_s16le \
  "$OUTPUT"
```

使用：

```bash
bash scripts/00_extract_audio.sh data/input/demo.mp4 data/audio/demo.wav
```

如果是公开视频，先用 yt-dlp 下载。yt-dlp 是命令行音视频下载工具，适合抓取公开视频样本做测试。([GitHub][5])

```bash
yt-dlp -f bestaudio -x --audio-format wav -o "data/input/%(title)s.%(ext)s" "视频URL"
```

---

# 5. 第二步：中文转写 + 基础时间段

## 工具

**FunASR + SenseVoiceSmall + fsmn-vad**

## 能实现什么

这一层负责：

```text
中文语音识别
长音频自动切分
基础时间段锚定
标点/文本规范化
```

FunASR 的 VAD 示例说明，`fsmn-vad` 输出格式是 `[[beg1, end1], ...]`，其中开始和结束时间单位是毫秒；这正好可以作为后续知识点回跳的基础时间锚。([GitHub][6])

## 脚本：`scripts/01_asr_funasr.py`

```python
import json
import sys
from pathlib import Path

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def main(audio_path: str, output_path: str):
    device = "cuda:0"  # 没有 GPU 可改成 "cpu"

    model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
    )

    result = model.generate(
        input=audio_path,
        language="zh",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )

    items = []
    for idx, r in enumerate(result):
        text = rich_transcription_postprocess(r.get("text", ""))
        item = {
            "id": idx,
            "text": text,
            "raw": r,
        }

        # 不同 FunASR / SenseVoice 版本返回字段可能不同，先完整保留 raw。
        if "timestamp" in r:
            item["timestamp"] = r["timestamp"]

        items.append(item)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"ASR saved to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

运行：

```bash
python scripts/01_asr_funasr.py \
  data/audio/demo.wav \
  data/transcript/demo_asr.json
```

## 这里的关键建议

POC 阶段不要过度追求“每个字 100% 对齐”。先实现：

```text
每 10～30 秒一段
每段有 start_ms / end_ms
每段有文本
每段后续可绑定 speaker
```

原因是 FunASR 的时间戳能力虽然可用，但社区 issue 中确实有人反馈过 `sentence_timestamp` 或字符时间戳边界不稳定的问题；所以工程上更稳的是：**先以 VAD 片段作为时间锚，再做句子聚合**。([GitHub][7])

---

# 6. 第三步：说话人分离

## 工具

**pyannote.audio**

## 能实现什么

它会输出类似：

```text
00:00:03.2 - 00:00:08.7 SPEAKER_00
00:00:08.9 - 00:00:15.1 SPEAKER_01
```

pyannote.audio 官方说明它是基于 PyTorch 的开源 speaker diarization 工具箱；pyannote 的 `speaker-diarization-community-1` pipeline 输入 16kHz 单声道音频并输出说话人分离结果。([GitHub][8])

## 准备 Hugging Face Token

pyannote 的部分模型需要 Hugging Face token。`.env`：

```text
HF_TOKEN=你的_huggingface_token
```

## 脚本：`scripts/02_diarize_pyannote.py`

```python
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pyannote.audio import Pipeline


def main(audio_path: str, output_path: str):
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("请在 .env 中设置 HF_TOKEN")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )

    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start_ms": int(turn.start * 1000),
            "end_ms": int(turn.end * 1000),
            "speaker": speaker,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"Diarization saved to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

运行：

```bash
python scripts/02_diarize_pyannote.py \
  data/audio/demo.wav \
  data/diarization/demo_speakers.json
```

---

# 7. 第四步：合并 ASR 与说话人时间轴

## 目标

把 ASR 段落和 speaker 段落合并成：

```json
{
  "start_ms": 12300,
  "end_ms": 26800,
  "speaker": "SPEAKER_00",
  "text": "我们今天主要讨论这个项目的落地方案。"
}
```

## 脚本：`scripts/03_merge_timeline.py`

这里先给一个简单版本：按时间重叠最多的 speaker 归属。

```python
import json
import sys
from pathlib import Path


def overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def guess_speaker(seg, speaker_segments):
    best_speaker = "UNKNOWN"
    best_overlap = 0

    for spk in speaker_segments:
        ov = overlap(
            seg["start_ms"],
            seg["end_ms"],
            spk["start_ms"],
            spk["end_ms"],
        )
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = spk["speaker"]

    return best_speaker


def normalize_asr(asr_items):
    """
    POC 简化版：
    如果 ASR 结果没有可靠 timestamp，就先把整段文本放在 0 开始。
    正式版建议从 FunASR raw 里解析 VAD/timestamp 字段。
    """
    normalized = []

    for i, item in enumerate(asr_items):
        text = item.get("text", "").strip()
        if not text:
            continue

        # 这里先用占位时间，正式项目中应替换为 VAD/timestamp 解析结果。
        start_ms = item.get("start_ms", i * 15000)
        end_ms = item.get("end_ms", (i + 1) * 15000)

        normalized.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
        })

    return normalized


def main(asr_path: str, speaker_path: str, output_path: str):
    asr_items = json.load(open(asr_path, "r", encoding="utf-8"))
    speaker_segments = json.load(open(speaker_path, "r", encoding="utf-8"))

    asr_segments = normalize_asr(asr_items)

    timeline = []
    for seg in asr_segments:
        speaker = guess_speaker(seg, speaker_segments)
        timeline.append({
            "start_ms": seg["start_ms"],
            "end_ms": seg["end_ms"],
            "speaker": speaker,
            "text": seg["text"],
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)

    print(f"Merged timeline saved to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
```

运行：

```bash
python scripts/03_merge_timeline.py \
  data/transcript/demo_asr.json \
  data/diarization/demo_speakers.json \
  data/transcript/demo_timeline.json
```

正式版里，这一步要把 FunASR 的真实 VAD/timestamp 结果解析进来；上面代码是为了让 POC 目录先能跑通。

---

# 8. 第五步：摘要、知识点、待办抽取

## 工具

**Qwen3 或你已有的大模型接口**

## 能实现什么

把时间轴文本转换成结构化结果：

```text
全文摘要
章节摘要
知识点
决策
待办
风险
关键引用
对应时间点
```

Qwen3 是 Qwen 系列公开权重模型，官方仓库称其为 Qwen 家族新一代大语言模型；如果后续做向量检索，还可以用 Qwen3-Embedding 系列做中文/多语言 embedding 和 reranking。([GitHub][9])

## 推荐 Prompt 模板

```text
你是会议纪要与视频知识点分析助手。

下面是带时间戳和说话人的转写文本。
请输出 JSON，不要输出 Markdown。

要求：
1. 给出 overall_summary。
2. 按主题生成 chapters，每个 chapter 包含 title、summary、start_ms、end_ms。
3. 抽取 knowledge_points，每个知识点包含：
   - title
   - summary
   - evidence_text
   - start_ms
   - end_ms
   - speaker
   - tags
4. 抽取 action_items，每个待办包含：
   - task
   - owner，如果无法判断则为 null
   - deadline，如果无法判断则为 null
   - evidence_text
   - start_ms
   - end_ms
5. 不要编造文本中没有的信息。
```

## 脚本：`scripts/04_summarize.py`

这里先写成接口适配形式，你可以接 Qwen、本地 vLLM、OpenAI 兼容 API 或公司内部模型。

```python
import json
import sys
from pathlib import Path


def build_prompt(timeline):
    lines = []
    for seg in timeline:
        start = seg["start_ms"] / 1000
        end = seg["end_ms"] / 1000
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg["text"]
        lines.append(f"[{start:.1f}-{end:.1f}] {speaker}: {text}")

    content = "\n".join(lines)

    return f"""
你是会议纪要与视频知识点分析助手。

下面是带时间戳和说话人的转写文本。
请输出严格 JSON，不要输出 Markdown。

输出结构：
{{
  "overall_summary": "...",
  "chapters": [
    {{
      "title": "...",
      "summary": "...",
      "start_ms": 0,
      "end_ms": 0
    }}
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
- 不要编造。
- 时间点必须来自原文片段。
- 知识点要短、准、可检索。

转写文本：
{content}
"""


def call_llm(prompt: str):
    """
    这里留作适配层：
    1. 可以接本地 Qwen3 + vLLM
    2. 可以接 OpenAI-compatible API
    3. 可以接公司内部模型服务

    POC 时你也可以先 print(prompt)，手动丢给模型测试。
    """
    raise NotImplementedError("请在这里接入你的 LLM API")


def main(timeline_path: str, output_path: str):
    timeline = json.load(open(timeline_path, "r", encoding="utf-8"))
    prompt = build_prompt(timeline)

    # POC 阶段先保存 prompt，方便手动调试
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path.replace(".json", "_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"Prompt saved to {output_path.replace('.json', '_prompt.txt')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

运行：

```bash
python scripts/04_summarize.py \
  data/transcript/demo_timeline.json \
  data/summary/demo_summary.json
```

---

# 9. 第六步：构建知识库

## 工具

**Milvus Lite + Qwen3-Embedding 或 sentence-transformers**

## 能实现什么

把每个知识点存成：

```text
向量
标题
摘要
证据原文
start_ms
end_ms
speaker
source_file
```

以后可以支持：

```text
问：这段会议里关于预算的结论是什么？
答：返回总结 + 原文证据 + 时间点
```

Milvus 官方说明它是面向 AI 应用的开源向量数据库，Milvus Lite 则适合本地快速原型和轻量部署。([Milvus][10])

## 脚本：`scripts/05_build_kb.py`

```python
import json
import sys
from pathlib import Path

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer


def main(summary_path: str, db_path: str):
    data = json.load(open(summary_path, "r", encoding="utf-8"))
    points = data.get("knowledge_points", [])

    embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    client = MilvusClient(db_path)

    collection_name = "video_knowledge"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=512,
    )

    rows = []
    for idx, p in enumerate(points):
        text = f"{p.get('title', '')}\n{p.get('summary', '')}\n{p.get('evidence_text', '')}"
        vector = embedder.encode(text).tolist()

        rows.append({
            "id": idx,
            "vector": vector,
            "title": p.get("title"),
            "summary": p.get("summary"),
            "evidence_text": p.get("evidence_text"),
            "start_ms": p.get("start_ms"),
            "end_ms": p.get("end_ms"),
            "speaker": p.get("speaker"),
            "tags": ",".join(p.get("tags", [])),
        })

    if rows:
        client.insert(collection_name=collection_name, data=rows)

    print(f"Inserted {len(rows)} knowledge points into {db_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

运行：

```bash
python scripts/05_build_kb.py \
  data/summary/demo_summary.json \
  data/vector_db/demo.db
```

---

# 10. 一键运行顺序

```bash
# 1. 转音频
bash scripts/00_extract_audio.sh \
  data/input/demo.mp4 \
  data/audio/demo.wav

# 2. 中文 ASR
python scripts/01_asr_funasr.py \
  data/audio/demo.wav \
  data/transcript/demo_asr.json

# 3. 说话人分离
python scripts/02_diarize_pyannote.py \
  data/audio/demo.wav \
  data/diarization/demo_speakers.json

# 4. 合并时间轴
python scripts/03_merge_timeline.py \
  data/transcript/demo_asr.json \
  data/diarization/demo_speakers.json \
  data/transcript/demo_timeline.json

# 5. 生成摘要 Prompt / 接入 LLM
python scripts/04_summarize.py \
  data/transcript/demo_timeline.json \
  data/summary/demo_summary.json

# 6. 构建知识库
python scripts/05_build_kb.py \
  data/summary/demo_summary.json \
  data/vector_db/demo.db
```

---

# 11. 准确率优先时的升级路线

## POC 版

```text
SenseVoiceSmall + VAD 时间段
```

适合先跑通。

## 准生产版

```text
SenseVoiceSmall 转写
+ fsmn-vad 切段
+ 自己做句子聚合
+ pyannote 说话人分离
```

适合会议、访谈、课程视频。

## 更高时间戳精度版

```text
SenseVoiceSmall / Paraformer 负责转写
+ FunASR fa-zh 做强制对齐
+ 人工抽样校准偏移
```

FunASR 的 `fa-zh` 是中文时间戳预测模型，官方示例中通过音频和文本一起输入来生成时间戳；但社区里也有关于时间戳偏移和边界问题的反馈，所以建议把它作为“精细对齐增强层”，不要一开始就把全部系统稳定性压在它上面。([Hugging Face][11])

---

# 12. 最终建议

你现在最应该先做的是这套：

```text
FFmpeg
+ FunASR / SenseVoiceSmall
+ fsmn-vad
+ pyannote.audio
+ Qwen3 或现有中文大模型
+ Milvus Lite
```

先不要急着做很复杂的 UI。第一阶段只要交付这四个结果，就已经能验证项目可行性：

```text
1. demo_timeline.json：带时间段、说话人的全文转写
2. demo_summary.json：全文摘要、章节摘要、知识点、待办
3. demo.db：可检索知识库
4. 点击知识点能回到视频 start_ms
```

这就是中文场景下**最容易落地、准确率也相对稳**的一条路径。

[1]: https://github.com/modelscope/FunASR?utm_source=chatgpt.com "modelscope/FunASR: A Fundamental End-to- ..."
[2]: https://milvus.io/docs/milvus_lite.md?utm_source=chatgpt.com "Run Milvus Lite Locally"
[3]: https://paddlepaddle.github.io/PaddleOCR/main/en/quick_start.html?utm_source=chatgpt.com "Quick Start - PaddleOCR Documentation"
[4]: https://ffmpeg.org/ffmpeg.html?utm_source=chatgpt.com "ffmpeg Documentation"
[5]: https://github.com/pyannote?utm_source=chatgpt.com "pyannote"
[6]: https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README.md?utm_source=chatgpt.com "FunASR/docs/tutorial/README.md at main"
[7]: https://github.com/modelscope/FunASR/issues/2110?utm_source=chatgpt.com "sentence_timestamp error!!! #2110 - modelscope/FunASR"
[8]: https://github.com/pyannote/pyannote-audio?utm_source=chatgpt.com "pyannote/pyannote-audio: Neural building blocks for ..."
[9]: https://github.com/QwenLM/qwen3?utm_source=chatgpt.com "Qwen3 is the large language model series developed by ..."
[10]: https://milvus.io/docs/quickstart.md?utm_source=chatgpt.com "Quickstart | Milvus Documentation"
[11]: https://huggingface.co/funasr/fa-zh?utm_source=chatgpt.com "funasr/fa-zh"
