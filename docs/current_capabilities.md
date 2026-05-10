# 当前能力与待补全功能

本文档记录项目当前能做什么、关键命令放在哪里，以及还需要补齐的能力。快速启动命令见根目录 `README.md`。

## 已验证环境

```text
conda 环境：cn-video-kb
路径：D:\Programs\anaconda3\envs\cn-video-kb
Python：3.10.20
PyTorch：2.11.0+cu130
CUDA：可用
FFmpeg：8.1.1
知识库后端：Qdrant Local Mode
```

PowerShell 中如果 `ffmpeg` 不在 PATH，可在运行时显式传入：

```powershell
--ffmpeg D:\Programs\anaconda3\envs\cn-video-kb\Library\bin\ffmpeg.exe
```

完整 `run` 默认使用输入文件名作为 `run_id`。例如 `data/input/demo.mp4` 会输出到 `data/runs/demo/`；只有需要自定义输出目录时才传 `--run-id`。

## 当前能力

| 能力 | 实现文件 | 输出 |
| --- | --- | --- |
| 本地视频/音频输入 | `src/video_kb/cli.py`、`src/video_kb/paths.py` | `data/runs/<run_id>/` |
| 音频提取 | `src/video_kb/audio.py` | 16kHz 单声道 WAV |
| VAD 与中文 ASR | `src/video_kb/asr.py` | `*_vad.json`、`*_asr.json` |
| 说话人分离 | `src/video_kb/diarization.py` | pyannote 或 3D-Speaker 输出的 `*_speakers_raw.json` |
| 说话人清洗 | `src/video_kb/diarization.py` | cleaned speaker segments |
| ASR 与 speaker 合并 | `src/video_kb/timeline.py` | `*_timeline.json` |
| 摘要、章节、知识点、待办 | `src/video_kb/summarize.py` | `*_summary.json` |
| 本地知识库构建 | `src/video_kb/kb.py` | Qdrant local DB + meta JSON |
| 知识库检索 | `src/video_kb/kb.py` | evidence、speaker、start_ms、end_ms |
| 报告模式 | `src/video_kb/reports.py` | full / speaker / topics JSON 或 Markdown |

## 分步命令

```powershell
python -m video_kb.cli extract-audio .\data\input\demo.mp4 .\data\runs\demo\audio\demo.wav

python -m video_kb.cli asr `
  .\data\runs\demo\audio\demo.wav `
  .\data\runs\demo\transcript\demo_asr.json `
  --vad-output .\data\runs\demo\transcript\demo_vad.json `
  --device cuda:0

python -m video_kb.cli diarize `
  .\data\runs\demo\audio\demo.wav `
  .\data\runs\demo\diarization\demo_speakers_raw.json `
  --backend pyannote

python -m video_kb.cli clean-diarization `
  .\data\runs\demo\diarization\demo_speakers_raw.json `
  .\data\runs\demo\diarization\demo_speakers.json

python -m video_kb.cli merge `
  .\data\runs\demo\transcript\demo_asr.json `
  .\data\runs\demo\diarization\demo_speakers.json `
  .\data\runs\demo\transcript\demo_timeline.json

python -m video_kb.cli summarize `
  .\data\runs\demo\transcript\demo_timeline.json `
  .\data\runs\demo\summary\demo_summary.json `
  --engine llm

python -m video_kb.cli build-kb `
  .\data\runs\demo\summary\demo_summary.json `
  .\data\runs\demo\vector_db\qdrant `
  --backend qdrant `
  --embedding-provider hash `
  --meta-output .\data\runs\demo\vector_db\demo_meta.json

python -m video_kb.cli query `
  --db .\data\runs\demo\vector_db\qdrant `
  --meta .\data\runs\demo\vector_db\demo_meta.json `
  --query "这段视频主要讲了什么？"

python -m video_kb.cli report `
  .\data\runs\demo\transcript\demo_timeline.json `
  .\data\runs\demo\reports\demo_topics.md `
  --mode topics `
  --format md
```

## 环境变量

- `HF_TOKEN`：pyannote 说话人分离需要。没有 token 时，完整 `run` 会写入 `skipped=true` 的 `SPEAKER_00` 兜底结果，除非使用 `--require-diarization`。
- `DASHSCOPE_API_KEY`、`DASHSCOPE_MODEL`、`DASHSCOPE_BASE_URL`：Qwen / DashScope OpenAI-compatible 摘要接口。
- `LLM_API_KEY`、`LLM_MODEL`、`LLM_BASE_URL`：其他 OpenAI-compatible 摘要接口。
- `EMBEDDING_MODEL`：sentence-transformers embedding 模型名，默认尝试 `BAAI/bge-small-zh-v1.5`。

## 已验证链路

已验证：

```text
音频提取
FunASR VAD
SenseVoiceSmall 中文转写
pyannote 说话人分离
说话人清洗
timeline 合并
heuristic / LLM 摘要入口
Qdrant Local Mode 建库
query 检索
full / speaker / topics 报告生成
无 HF_TOKEN 时的 SPEAKER_00 兜底
```

在 `data/runs/demo/` 上验证过不限制人数的 speaker 清洗：原始 diarization 片段被清理为更少的有效片段，同时保留 `SPEAKER_00` 到 `SPEAKER_07` 的标签。

## 待补全功能

- 摘要质量：增加 LLM JSON 输出校验、失败重试和自动修复。
- 检索质量：接入更高质量中文 embedding，例如 BGE 或 Qwen embedding，并增加 rerank。
- 时间戳精度：接入 FunASR fa-zh 或 forced alignment 做字/词级时间戳。
- 长视频稳定性：增加分段处理、断点续跑、模型加载复用和中间结果缓存。
- 可视化界面：展示 timeline、summary、knowledge points，并支持点击时间戳回放。
- 配置与日志：增加 `config.yaml`、统一日志文件、记录模型版本和每步耗时。
- 测试覆盖：补 CLI smoke test、ASR normalize fixture、Qdrant backend 测试。
