# 原始方案审查记录

`docs/archive/original_methodology.md` 的技术路线是合理的：FFmpeg 负责统一音频格式，FunASR/SenseVoice 负责中文转写，pyannote.audio 负责说话人分离，LLM 负责摘要抽取，Milvus Lite 负责本地知识库。

这次落地时修正了以下工程问题：

1. 原文 `03_merge_timeline.py` 在 ASR 缺少 timestamp 时使用 `i * 15000` 作为占位时间。项目实现改为先单独运行 `fsmn-vad`，再按 VAD 毫秒时间段切片转写，避免知识点回跳时间不可信。
2. 原文摘要脚本只保存 prompt，没有生成 `summary.json`。项目实现支持 OpenAI-compatible LLM，未配置 LLM 时自动生成可用的抽取式摘要。
3. 原文只构建 Milvus，没有查询入口。项目实现新增 `query` 命令，检索结果带原文证据与 `start_ms` / `end_ms`。
4. 原文以 Bash 脚本为主。项目实现改为跨平台 Python CLI，并提供 PowerShell 脚本入口。
5. 原文没有处理 pyannote token 缺失。项目实现默认可跳过并继续跑完整链路，也可用 `--require-diarization` 强制要求真实说话人分离。
6. Milvus Lite 本地 `.db` 引擎不支持当前 Windows 环境。项目实现改为默认使用 Qdrant Local Mode，保留 JSON fallback。

截图中的实现路径均保留在项目中：

- FFmpeg：`video_kb.audio.extract_audio`
- FunASR / SenseVoice：`video_kb.asr.run_asr`
- pyannote.audio：`video_kb.diarization.diarize`
- Qdrant Local Mode：`video_kb.kb.build_kb` 与 `video_kb.kb.query_kb`
