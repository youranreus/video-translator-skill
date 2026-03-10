---
name: video-subtitles
description: 从音视频或 SRT 生成双语字幕。默认日语转录并翻译为中文，输出源语言与目标语言两份 SRT，支持软字幕封装与硬字幕烧录，并支持自定义 OpenAI 端点/模型/API Key（CLI + env）。
---

# Video Subtitles

用于生成电影风格字幕，保持核心流程：输入 -> 源字幕 -> 翻译 -> 输出。

## Features

- **输入类型**: 支持音视频与 `.srt` 文件
- **默认语言**: 源语言 `ja`，目标语言 `zh`
- **双文件输出**: 同时输出源语言 SRT 与目标语言 SRT
- **翻译配置**: 支持自定义 OpenAI 端点、模型、API Key
- **提示词增强**: 支持自定义翻译附加提示词（领域词汇/术语风格）
- **兜底清洗**: 自动过滤模型可能返回的 `<think>...</think>` 等思考过程内容
- **视频输出**: 可封装软字幕（`--embed`）或烧录硬字幕（`--burn`）
- **字幕分段**: 电影风格自然断句（单行约 42 字符，总时长 1-7 秒）

## Quick Start

```bash
# 音视频输入，默认输出两份 SRT（ja + zh）
./scripts/generate_srt.py video.mp4

# 直接输入 SRT，跳过转录，输出两份 SRT
./scripts/generate_srt.py source.srt

# 烧录硬字幕到视频（使用目标语言字幕）
./scripts/generate_srt.py video.mp4 --burn

# 自定义 OpenAI 模型
./scripts/generate_srt.py video.mp4 --openai-model gpt-4o-mini

# 增加领域提示词（示例：机械操作说明）
./scripts/generate_srt.py video.mp4 \
  --translator-prompt "这是工业设备操作培训视频，请优先使用机加工与设备控制领域常用术语。"

# 自定义 OpenAI 兼容端点
./scripts/generate_srt.py video.mp4 \
  --openai-base-url https://your.endpoint/v1 \
  --openai-api-key sk-xxx
```

## Env 配置

可通过 `.env` 或系统环境变量配置：

```bash
SOURCE_LANG=ja
TARGET_LANG=zh
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-xxx
TRANSLATOR_PROMPT=这是工业设备操作培训视频，请优先使用机加工与设备控制领域常用术语。
```

优先级：`CLI 参数 > env > 默认值`

## 关键参数

| 参数 | 说明 |
|------|------|
| `--source-lang` | 源语言代码（默认 `ja`） |
| `--target-lang` | 目标语言代码（默认 `zh`） |
| `--openai-base-url` | OpenAI 兼容端点 |
| `--openai-model` | OpenAI 模型名 |
| `--openai-api-key` | OpenAI API Key |
| `--translator-prompt` | 翻译附加提示词 |
| `--output-dir` | 输出目录 |
| `--embed` | 封装软字幕到视频 |
| `--burn` | 烧录硬字幕到视频 |
| `--accurate` | 转录使用更准确但更慢的模型 |

## Requirements

- **uv**: Python 包管理器（自动安装依赖）
- **ffmpeg-full**: 烧录字幕建议安装（`brew install ffmpeg-full`）
- **Whisper 模型**: 首次运行会自动下载

## Output

- 源字幕：`<输入名>.<source_lang>.srt`
- 目标字幕：`<输入名>.<target_lang>.srt`
- 若使用 `--embed` 或 `--burn`：额外输出 `<输入名>_subtitled.mp4`
