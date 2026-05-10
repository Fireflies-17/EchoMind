# 项目结构

本项目按“源码、文档、脚本、测试、本地数据”分层。上传 GitHub 时，应提交源码和说明文件，不提交本地视频、模型缓存、运行结果和向量库。

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── .gitattributes
├── data/
│   ├── README.md
│   └── input/
│       └── .gitkeep
├── docs/
│   ├── README.md
│   ├── archive/
│   │   └── original_methodology.md
│   ├── current_capabilities.md
│   ├── diarization_backends.md
│   ├── diarization_cleaning.md
│   ├── methodology_review.md
│   ├── project_structure.md
│   └── report_modes.md
├── scripts/
│   ├── run_pipeline.ps1
│   └── run_pipeline.sh
├── src/
│   └── video_kb/
│       ├── __init__.py
│       ├── audio.py
│       ├── asr.py
│       ├── cli.py
│       ├── diarization.py
│       ├── kb.py
│       ├── paths.py
│       ├── reports.py
│       ├── summarize.py
│       ├── timeline.py
│       └── utils.py
└── tests/
    ├── fixtures/
    │   └── sample_timeline.json
    ├── test_diarization.py
    ├── test_kb.py
    ├── test_paths.py
    ├── test_reports.py
    ├── test_summarize.py
    └── test_timeline.py
```

## 应提交的内容

- `src/video_kb/`：核心 Python 包。
- `tests/`：轻量单元测试和 fixture。
- `docs/`：文档索引、当前能力、报告模式、说话人分离后端、说话人清洗、方法论评审、项目结构说明。
- `scripts/`：PowerShell 和 Bash 运行脚本。
- `pyproject.toml`、`requirements.txt`：安装和依赖配置。
- `.env.example`：环境变量模板，不包含真实 token。

## 不应提交的内容

- `.env`：包含 `HF_TOKEN`、`DASHSCOPE_API_KEY` 等私密配置。
- `data/input/*.mp4`、`data/input/*.wav`：本地视频和音频输入。
- `data/runs/`：ASR、diarization、summary、Qdrant 等运行结果。
- `build/`、`dist/`、`*.egg-info/`：打包产物。
- `models/`、`.cache/`：模型缓存。

## GitHub 上传前检查

```powershell
git status --short
git check-ignore -v .env data/input/demo.mp4 data/runs/demo
```

如果还没有初始化 Git 仓库：

```powershell
git init
git add .
git status --short
```

确认 `data/input/demo.mp4`、`.env`、`data/runs/` 没有进入待提交列表后，再提交：

```powershell
git commit -m "Initial video knowledge base pipeline"
```
