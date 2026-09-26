# Video → AI 资料包

将本地视频转换为可交给 ChatGPT、Codex、Claude 等 Agent 阅读的证据资料包。集成 faster-whisper、场景变化与最长间隔视觉证据、PaddleOCR、时间轴融合、OpenAI-compatible 分块总结、配置感知断点续跑及 ChatGPT 精简上传包。

## 给 Windows 新手：解压即用版

从项目维护者取得 `Video2AI-Windows-CPU.zip`，完整解压后打开 `Video2AI/先看我.txt`，再双击 `Video2AI.exe`。选择一个短 MP4 视频，保留默认设置，点击“开始生成”。首次运行需联网下载语音识别模型；之后可复用缓存。完成后可点“查看网页报告”，或者在“打开上传包目录”中找到可交给 ChatGPT 的 `upload_bundle.zip`。

此版默认使用 CPU，不要求 NVIDIA 显卡、Conda、Python 或单独安装 FFmpeg。视频留在本机处理；首次下载语音模型仍需网络。入门版暂不包含 OCR 和 AI 自动总结，这两项在界面中不可选；需要时请按下文安装完整版。便携版未经代码签名，只运行来自可信来源的副本。

维护者在 Windows 的 `video2ai` Conda 环境中执行 `python -m pip install -r requirements-build.txt`，然后运行 `python scripts/build_portable.py`，可重新制作 ZIP；不要把 `dist/` 提交到 Git。

## 重要：只使用独立 Conda 环境

本项目固定使用环境名 `video2ai`。不要在 `base`、系统 Python 或其他项目环境中安装依赖。

### 1. 创建环境（推荐）

打开 **Anaconda Prompt**，进入本项目：

```bat
cd /d "D:\aerospaceboy\video转换"
conda env create -f environment.yml
conda activate video2ai
python scripts\check_env.py
```

如果 `video2ai` 已存在，不要删除它。检查其用途后，可在确认需要同步本项目依赖时执行：

```bat
conda env update -n video2ai -f environment.yml --prune
conda activate video2ai
python scripts\check_env.py
```

`environment.yml` 会把 Python 3.11、FFmpeg、faster-whisper 所需的 CUDA 12.8/cuBLAS/cuDNN 运行库和 Python 包都装进 `video2ai`。这些依赖不需要写入 Windows 全局 PATH；没有 NVIDIA GPU 时程序仍会回退 CPU。

### 2. PaddleOCR

新建环境时 `environment.yml` 会安装已验证的 PaddleOCR 3.x。如果你的环境是在 OCR 加入项目前创建的，可在原有 `video2ai` 环境中补装：

```bat
conda activate video2ai
python -c "import sys,os; print(sys.executable); print(os.environ.get('CONDA_DEFAULT_ENV'))"
python -m pip install -r requirements-ocr-paddle.txt
```

上面的检查必须显示 `...\envs\video2ai\python.exe` 和 `video2ai`。如果不是，请停止，不要安装。

即使 OCR 初始化失败，ASR、关键帧、时间轴和资料包导出仍可继续。OCR 模型默认缓存在当前 `video2ai` 环境的 `paddlex_cache/` 中，不写入 base 环境。

## 快速运行

### Windows 桌面 App

双击项目中的 `start_video2ai.bat`，即可打开桌面界面。选择视频后点击“开始生成”；界面会显示运行日志和场景扫描进度。完成后可打开输出目录或 `chatgpt/upload_bundle.zip` 所在目录。“停止”会终止当前任务；再次开始会利用已有阶段缓存继续。

处理完成后点击“查看网页报告”，可在默认浏览器离线阅读章节、字幕、OCR 和关键画面，并搜索文字。也可以直接打开输出目录中的 `report.html`；页面只引用同目录的 `frames/` 图片，不需要网络服务。

桌面界面调用同一个 CLI，默认参数来自 `video2ai.yaml`。支持切换语音模型、计算设备、OCR 和 AI 总结；其他参数仍可在 YAML 中配置。AI 总结需要事先配置可访问的 OpenAI-compatible 服务。

也可以在 Anaconda Prompt 中启动：

```bat
conda activate video2ai
cd /d "D:\aerospaceboy\video转换"
python video2ai_app.py
```

### 命令行

```bat
conda activate video2ai
python video2ai.py "D:\Videos\test.mp4"
```

项目根目录的 `video2ai.yaml` 会被自动读取。本机配置已设为 CUDA、`large-v3`、PaddleOCR 和常用 FPGA 热词，因此日常使用只需要上面这一条命令。也可显式指定其他配置：

```bat
python video2ai.py "D:\Videos\test.mp4" --config "D:\配置\my-video2ai.yaml"
```

命令行参数优先于 YAML；例如配置关闭 OCR 时，追加 `--ocr` 可临时开启。

默认输出为 `D:\Videos\test_ai\`。指定输出目录：

```bat
python video2ai.py "D:\视频\测试视频.mp4" --output "D:\资料包\测试"
```

纯 CPU、关闭 OCR 和 LLM：

```bat
python video2ai.py "D:\Videos\test.mp4" --device cpu --no-ocr --no-llm
```

PowerShell 多行命令：

```powershell
python video2ai.py "D:\Videos\test.mp4" `
  --llm `
  --base-url "http://127.0.0.1:11434/v1" `
  --model "qwen2.5:7b"
```

CMD 多行命令：

```bat
python video2ai.py "D:\Videos\test.mp4" ^
  --llm ^
  --base-url "http://127.0.0.1:11434/v1" ^
  --model "qwen2.5:7b"
```

## LLM 配置

LLM 不是基础资料包的必需项。当前本机没有检测到 Ollama，因此 `video2ai.yaml` 默认关闭 LLM。安装并启动任意 OpenAI-compatible 服务后，可在 YAML 中设置 `llm.enabled: true`，或使用 `--llm --base-url --model` 临时开启。密钥建议放在项目根目录的 `.env`：

```text
OPENAI_API_KEY=
VIDEO2AI_BASE_URL=http://127.0.0.1:11434/v1
VIDEO2AI_MODEL=qwen2.5:7b
```

复制 `.env.example` 后填写即可；`.env` 已被 Git 忽略。API Key 不会写入 metadata 或输出资料包。只设置模型而不设置 base URL 时，默认使用 `https://api.openai.com/v1`。

长视频默认按 2 分钟时间轴分块。每个分块及摘要保存在 `raw/chunks/`，然后进行全局汇总。分段摘要缓存同时绑定内容、提示词版本、服务地址和模型，切换模型或证据变化时不会误用旧摘要。每次 LLM 调用都带有“仅根据字幕/OCR/时间戳/画面元数据、禁止脑补”的约束。

## 常用选项

```text
--language zh                 指定 ASR/OCR 语言
--whisper-model small         tiny/base/small/medium/large-v3
--hotwords "FPGA,Vivado"      提示专业术语，提高专有名词识别率
--initial-prompt-file FILE    UTF-8 术语/上下文提示文件
--beam-size 5                 Whisper 解码 beam size
--batch-size 8                启用批量 GPU ASR；0 表示标准高质量模式
--device auto|cuda|cpu        默认自动检测并回退 CPU
--max-frame-gap 30            无场景变化时的最大截图间隔
--scene-sample-fps 2          场景检测采样率；0 表示逐帧扫描
--timeline-chunk-seconds 120  基础证据时间段长度
--config FILE                读取 YAML 默认参数；默认自动读取 ./video2ai.yaml
--version                    显示 video2ai 版本
--ocr / --no-ocr             临时开启/关闭 OCR，覆盖 YAML
--llm / --no-llm             临时开启/关闭 LLM，覆盖 YAML
--overwrite                   强制重新生成各阶段结果
--force-asr                   仅强制 ASR
--force-frames                仅强制关键帧
--force-ocr                   仅强制 OCR
--force-timeline              仅强制时间轴
--force-summary               仅强制总结
```

重复运行同一视频会复用缓存。缓存会记录源视频和各阶段参数指纹；模型、语言、关键帧间隔、OCR 或时间轴配置改变时，只自动重建受影响阶段。不会创建 `_ai_1`、`_ai_2`，也不会复制原视频。

强制重算上游阶段时，依赖它的时间轴、总结和上传包会同步更新。新关键帧先使用独立文件名生成，确认资料包完成后才清理旧帧；中途停止时旧画面证据仍可用于恢复。OCR 初始化或 AI 服务请求失败后，下次运行会自动重试。上传压缩包或联系表缺失时，也会重新生成。

场景变化检测默认采样 2 FPS，但固定的 `--max-frame-gap` 视觉证据点不会减少。在 53,098 帧、约 29 分钟的测试视频上，场景扫描由约 6 分钟缩短到 48.33 秒。需要逐帧检测极短闪屏时，可使用 `--scene-sample-fps 0`；这会明显增加处理时间。

## 输出结构

```text
test_ai/
├── README.md                 资料包入口（不是本文件）
├── AI_HANDOFF.md             给下一个 Agent 的接手提示
├── summary.md
├── report.html               离线可搜索的网页报告
├── transcript.md
├── timeline.md
├── timeline.json
├── ocr.md
├── metadata.json
├── frames.zip                全部唯一关键帧（兼容旧版）
├── chatgpt/
│   ├── CHATGPT_HANDOFF.md
│   ├── VIDEO_EVIDENCE.md
│   ├── manifest.json
│   ├── upload_bundle.zip     推荐直接上传给 ChatGPT
│   └── contact_sheet_*.jpg
├── frames/
└── raw/
    ├── state.json
    ├── transcript.json
    ├── scenes.json
    ├── ocr.json
    └── chunks/
```

OCR 原文不会被“修正”。规则分类会把疑似代码、命令和报错分别展示；技术事实可通过时间戳和 `frames/` 回查。

`chatgpt/` 是推荐上传目录：优先直接上传 `upload_bundle.zip`；如果目标 AI 不读取 ZIP，则上传两个 Markdown 文件以及所有联系表图片。相同画面只保存一次，但每个最长间隔仍会保留可追溯的视觉时间点。`frames.zip` 与资料包指纹同步重建，不会残留旧关键帧。

## CUDA

`--device auto` 会检查 PyTorch（如果环境中已有）和 faster-whisper 使用的 CTranslate2。发现可用 CUDA 时选择 `cuda/float16`，否则使用 `cpu/int8`。本项目不安装或覆盖 PyTorch；CTranslate2 需要的 CUDA 12.8、cuBLAS 和 cuDNN 运行库由 Conda 隔离安装在 `video2ai` 环境中。

如果指定 `--device cuda` 但运行时不可用，程序会记录警告并回退 CPU。

## 测试

必须先确认解释器：

```bat
conda activate video2ai
python -c "import sys,os; print(sys.executable); print(os.environ.get('CONDA_DEFAULT_ENV'))"
python -m pytest -q
```

不要在 base 或系统 Python 下运行安装与项目测试。

## 故障排除

- `FFmpeg not found`：激活环境后执行 `conda install -c conda-forge ffmpeg`，不要优先改全局 PATH。
- OCR 初始化失败：程序会继续生成无 OCR 资料包。可先用 `--no-ocr`，再检查 PaddlePaddle wheel 是否支持当前 Windows/Python。
- LLM 请求失败：程序会保留字幕、OCR、关键帧和时间轴，并生成基础 `summary.md`。
- 视频无音轨：`metadata.json` 会记录 `has_audio: false`，画面/OCR/时间轴照常执行。
- 中文路径：程序使用 `pathlib` 和 subprocess 参数列表，不使用 `shell=True`。

## 第一版边界

已实现 ASR、关键帧、保守去重、可选 OCR、代码/命令/错误线索分类、结构化时间轴、分块 LLM 总结、断点续跑与 Agent 接手文件。当前没有实现 WhisperX、说话人分离、VLM 直接看图、向量检索、Web UI 或批处理；这些均可沿现有后端接口后续扩展。
