# 说话人分离后端

项目现在支持两个说话人分离后端：

```text
pyannote    默认后端，安装和调用路径已经验证
3dspeaker   ModelScope 3D-Speaker / SpeakerLab，可用于对比中文会议场景效果
```

## pyannote

默认用法：

```powershell
python -m video_kb.cli run `
  --input .\data\input\1.MP4 `
  --diarization-backend pyannote
```

已知真实人数时：

```powershell
python -m video_kb.cli run `
  --input .\data\input\1.MP4 `
  --diarization-backend pyannote `
  --min-speakers 3 `
  --max-speakers 3 `
  --clean-max-speakers 3
```

## 3D-Speaker

3D-Speaker 不是本项目的直接依赖。它的 `requirements.txt` 会安装较旧的 `numpy` 和 `scikit-learn`，会破坏 `cn-video-kb` 里的 pyannote 依赖。

推荐做法：单独创建一个 3D-Speaker conda 环境，然后让主项目通过子进程调用它。

```powershell
git clone https://github.com/modelscope/3D-Speaker external\3D-Speaker
conda create -n 3dspeaker python=3.8 -y
conda activate 3dspeaker
cd external\3D-Speaker
pip install -r requirements.txt
cd ..\..
conda activate cn-video-kb
```

配置仓库路径和 3D-Speaker 环境的 Python：

```powershell
$env:THREED_SPEAKER_REPO = ".\external\3D-Speaker"
$env:THREED_SPEAKER_PYTHON = "D:\Programs\anaconda3\envs\3dspeaker\python.exe"
```

完整链路：

```powershell
python -m video_kb.cli run `
  --input .\data\input\1.MP4 `
  --diarization-backend 3dspeaker `
  --threed-speaker-repo .\external\3D-Speaker `
  --threed-speaker-python D:\Programs\anaconda3\envs\3dspeaker\python.exe `
  --speaker-num 3 `
  --clean-max-speakers 3
```

只重跑说话人分离：

```powershell
python -m video_kb.cli diarize `
  .\data\runs\1\audio\1.wav `
  .\data\runs\1\diarization\1_speakers_raw_3dspeaker.json `
  --backend 3dspeaker `
  --threed-speaker-repo .\external\3D-Speaker `
  --threed-speaker-python D:\Programs\anaconda3\envs\3dspeaker\python.exe `
  --speaker-num 3
```

然后清洗和合并：

```powershell
python -m video_kb.cli clean-diarization `
  .\data\runs\1\diarization\1_speakers_raw_3dspeaker.json `
  .\data\runs\1\diarization\1_speakers_3dspeaker.json `
  --max-speakers 3

python -m video_kb.cli merge `
  .\data\runs\1\transcript\1_asr.json `
  .\data\runs\1\diarization\1_speakers_3dspeaker.json `
  .\data\runs\1\transcript\1_timeline_3dspeaker.json
```

## 参数建议

- 确定人数：`--speaker-num N`，同时加 `--clean-max-speakers N`。
- 不确定人数：不要传 `--speaker-num`，只用默认清洗。
- 有重叠说话：可尝试 `--include-overlap`，但它需要 `HF_TOKEN`，因为 3D-Speaker 的 overlap detection 会用到 pyannote segmentation。

## 注意

3D-Speaker 第一次运行会下载 ModelScope 模型，耗时取决于网络。主项目会读取 3D-Speaker 生成的 JSON，并转换成和 pyannote 一样的格式：

```json
{
  "start_ms": 100,
  "end_ms": 1200,
  "speaker": "SPEAKER_00"
}
```

如果已经不小心在 `cn-video-kb` 中安装了 3D-Speaker requirements，需要把主环境依赖修回来：

```powershell
python -m pip install "numpy>=2.2.2,<3" "scikit-learn>=1.6.1"
python -m pip check
```
