# 说话人清洗说明

pyannote 输出的 `SPEAKER_00`、`SPEAKER_01` 等只是声音聚类标签，不等同于真实人物编号。多人讨论、插话、重叠说话、噪声、音量变化都会让同一个人被拆成多个 speaker。

本项目现在增加了说话人清洗步骤：

```powershell
python -m video_kb.cli clean-diarization `
  .\data\runs\demo\diarization\demo_speakers.json `
  .\data\runs\demo\diarization\demo_speakers_clean.json `
  --min-segment-ms 800 `
  --merge-gap-ms 500 `
  --reassign-gap-ms 3000
```

参数含义：

- `--min-segment-ms`：丢弃过短的 speaker 碎片，默认 `800`。
- `--merge-gap-ms`：合并相邻且间隔很短的同 speaker 片段，默认 `500`。
- `--max-speakers`：可选。只有明确知道真实人数或想限制聚类数量时才设置；人数不确定时不要设置。
- `--reassign-gap-ms`：被压缩掉的小 speaker 片段，如果靠近主 speaker，就重分配给最近的主 speaker。

清洗后重新合并文本：

```powershell
python -m video_kb.cli merge `
  .\data\runs\demo\transcript\demo_asr.json `
  .\data\runs\demo\diarization\demo_speakers_clean.json `
  .\data\runs\demo\transcript\demo_timeline_clean.json
```

之后再基于 `demo_timeline_clean.json` 做摘要、建库和检索。

完整 `run` 命令现在默认会保留 raw diarization，并把 cleaned diarization 用于后续 timeline。输出路径包括：

```text
data/runs/<run_id>/diarization/<run_id>_speakers_raw.json
data/runs/<run_id>/diarization/<run_id>_speakers.json
```

人数不确定或人数较多时，完整运行不要传 `--min-speakers`、`--max-speakers` 或 `--clean-max-speakers`：

```powershell
python -m video_kb.cli run `
  --input .\data\input\demo.mp4 `
  --device cuda:0 `
  --summary-engine llm `
  --embedding-provider hash `
  --kb-backend qdrant
```

如果想完全保留 pyannote 原始结果，可以加：

```powershell
--no-clean-diarization
```
