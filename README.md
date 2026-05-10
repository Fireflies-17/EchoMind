# 中文视频知识库流水线

本项目提供一个可在 Windows 本地运行的 CLI 流水线，用于处理中文视频或音频：

```text
本地视频/音频
  -> FFmpeg 提取音频
  -> FunASR VAD + SenseVoiceSmall 转写
  -> pyannote.audio 说话人分离
  -> 时间轴合并与说话人清洗
  -> Qwen / OpenAI-compatible LLM 摘要
  -> Qdrant 本地知识库
  -> 检索并返回证据文本、speaker、时间戳
```

## 快速开始

```powershell
conda activate cn-video-kb
pip install .
Copy-Item .env.example .env
```

将本地视频放到：

```text
data/input/demo.mp4
```

默认情况下，`run_id` 会自动使用输入文件名去掉扩展名后的部分。比如 `demo.mp4` 会输出到 `data/runs/demo/`，所以日常只需要修改 `--input`。

运行完整链路：

```powershell
python -m video_kb.cli run `
  --input .\data\input\demo.mp4 `
  --device cuda:0 `
  --summary-engine llm `
  --embedding-provider hash `
  --kb-backend qdrant
```

检索知识库：

```powershell
python -m video_kb.cli query `
  --db .\data\runs\demo\vector_db\qdrant `
  --meta .\data\runs\demo\vector_db\demo_meta.json `
  --query "这段视频主要讲了什么？"
```

生成报告：

```powershell
python -m video_kb.cli report `
  .\data\runs\demo\transcript\demo_timeline.json `
  .\data\runs\demo\reports\demo_topics.md `
  --mode topics `
  --format md
```

## 环境变量

复制 `.env.example` 后按需配置：

```text
HF_TOKEN=...
DASHSCOPE_API_KEY=...
DASHSCOPE_MODEL=qwen-plus
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

`HF_TOKEN` 用于 pyannote 说话人分离；Qwen / DashScope 配置用于生成更高质量的结构化摘要。未配置 LLM 时，项目会退回本地启发式摘要。

## 文档

- [当前能力与使用说明](docs/current_capabilities.md)
- [报告模式](docs/report_modes.md)
- [说话人分离后端](docs/diarization_backends.md)
- [说话人清洗说明](docs/diarization_cleaning.md)
- [方法论审查记录](docs/methodology_review.md)
- [项目结构与提交规则](docs/project_structure.md)
- [文档索引](docs/README.md)

## 测试

```powershell
python -B -m unittest discover -s tests
```

## GitHub 提交注意

`.env`、本地视频、音频、运行结果和向量库已在 `.gitignore` 中排除。提交前确认：

```powershell
git status --short --ignored
```
