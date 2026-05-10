# 报告模式

报告模式基于已经生成的 `timeline.json` 工作，不会重新跑 FFmpeg、ASR 或说话人分离。

## 1. 整个视频转译

按说话人切换合并连续发言，输出每段的说话人、时间范围、原文和摘要。

```powershell
python -m video_kb.cli report `
  .\data\runs\demo\transcript\demo_timeline.json `
  .\data\runs\demo\reports\demo_full.md `
  --mode full `
  --format md `
  --engine auto
```

JSON 输出：

```powershell
python -m video_kb.cli report `
  .\data\runs\demo\transcript\demo_timeline.json `
  .\data\runs\demo\reports\demo_full.json `
  --mode full
```

## 2. 筛选说话人

只提取某个 speaker 的每一段发言文本和摘要。`--speaker` 支持 `SPEAKER_01` 或数字简写 `1`。

```powershell
python -m video_kb.cli report `
  .\data\runs\demo\transcript\demo_timeline.json `
  .\data\runs\demo\reports\speaker_01.md `
  --mode speaker `
  --speaker SPEAKER_01 `
  --format md `
  --engine auto
```

## 3. 筛选议题

优先使用已配置的 Qwen / OpenAI-compatible LLM 智能识别议题；如果没有配置 LLM，则使用本地规则按时间间隔、话题提示词和窗口长度兜底切分。

```powershell
python -m video_kb.cli report `
  .\data\runs\demo\transcript\demo_timeline.json `
  .\data\runs\demo\reports\demo_topics.md `
  --mode topics `
  --format md `
  --engine auto
```

议题模式输出结构：

```text
议题
  摘要
  时间范围
  SPEAKER_00
    发言摘要
    该议题下的每段发言
  SPEAKER_01
    发言摘要
    该议题下的每段发言
```

## 引擎选择

- `--engine heuristic`：只用本地规则，速度快，不需要 API。
- `--engine auto`：有 LLM 配置时使用 LLM，否则自动回退到本地规则。
- `--engine llm`：强制使用 LLM；如果没有配置接口会报错。

## 输出格式

- `--format json`：适合后续程序读取。
- `--format md`：适合直接阅读和提交报告。
