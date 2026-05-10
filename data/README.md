# data 目录

这个目录用于本地输入和流水线输出，不建议把真实视频、音频、转写结果或向量库提交到 GitHub。

推荐结构：

```text
data/
  input/              # 本地视频或音频输入，仅保留 .gitkeep
  runs/<run_id>/      # 每次完整处理的输出，已被 .gitignore 忽略
```

示例输入可以放在：

```text
data/input/demo.mp4
```

运行结果会生成到：

```text
data/runs/<run_id>/
```
