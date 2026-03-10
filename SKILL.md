---
name: video-subtitles
description: Generate SRT subtitles from video/audio with translation support. Transcribes Hebrew (ivrit.ai), English, and Japanese (whisper), translates subtitles (including Japanese to Simplified Chinese), burns subtitles into video. Use for creating captions, transcripts, or hardcoded subtitles for WhatsApp/social media.
---

# Video Subtitles

Generate movie-style subtitles from video or audio files. Supports transcription, translation, and burning subtitles directly into video.

## Features

- **Hebrew**: ivrit.ai fine-tuned model (best Hebrew transcription)
- **English**: OpenAI Whisper large-v3
- **Auto-detect**: Automatically detects language and selects best model
- **Translation**: Translate to English (`--translate en`) or Simplified Chinese (`--translate zh`)
- **Burn-in**: Hardcode subtitles into video (visible everywhere, including WhatsApp)
- **Movie-style**: Natural subtitle breaks (42 chars/line, 1-7s duration)

## Quick Start

```bash
# Plain transcript
./scripts/generate_srt.py video.mp4

# Generate SRT file
./scripts/generate_srt.py video.mp4 --srt

# Burn subtitles into video (always visible)
./scripts/generate_srt.py video.mp4 --srt --burn

# Translate to English + burn in
./scripts/generate_srt.py video.mp4 --srt --burn --translate en

# Japanese -> Simplified Chinese subtitles
./scripts/generate_srt.py video.mp4 --srt --translate zh --lang ja

# Force language
./scripts/generate_srt.py video.mp4 --lang he    # Hebrew
./scripts/generate_srt.py video.mp4 --lang en    # English
./scripts/generate_srt.py video.mp4 --lang ja    # Japanese
```

## Options

| Flag | Description |
|------|-------------|
| `--srt` | Generate SRT subtitle file |
| `--burn` | Burn subtitles into video (hardcoded, always visible) |
| `--embed` | Embed soft subtitles (toggle in player) |
| `--translate en/zh` | Translate to English or Simplified Chinese |
| `--lang he/en/ja` | Force input language |
| `--translation-model MODEL` | LLM model for `--translate zh` (overrides config file model) |
| `-o FILE` | Custom output path |

## Output

- **Default**: Plain text transcript to stdout
- **With `--srt`**: Creates `video.srt` alongside input
- **With `--burn`**: Creates `video_subtitled.mp4` with hardcoded subs

## Requirements

- **uv**: Python package manager (auto-installs dependencies)
- **ffmpeg-full**: For burning subtitles (`brew install ffmpeg-full`)
- **Models**: ~3GB each, auto-downloaded on first use
- **OpenAI config**: `--translate zh` reads `openai.api_key`, optional `openai.base_url`, and `openai.translation` from `config/openai.json`
- **Env fallback**: `OPENAI_API_KEY` and `OPENAI_BASE_URL` can be used when config file is absent/incomplete

## OpenAI Config File

Create `config/openai.json`:

```json
{
  "openai": {
    "api_key": "sk-...",
    "base_url": "https://api.openai.com/v1",
    "translation": {
      "model": "gpt-4.1-mini",
      "prompt": {
        "role": "You are a bilingual subtitle localization specialist for Japanese media.",
        "expertise": "Use terms common in anime/film workflows and keep names, honorifics, and domain terms consistent.",
        "instructions": "Keep subtitle tone natural in Simplified Chinese and avoid over-literal wording."
      },
      "prompt_append": "Prefer concise subtitle phrasing when a literal translation sounds too long."
    }
  }
}
```

- `api_key` is required for `--translate zh`
- `base_url` is optional and supports custom compatible endpoints
- `openai.translation.model` sets the default zh translation model (can be overridden by `--translation-model`)
- `openai.translation.prompt` supports string or object form (`role` / `expertise` / `instructions`)
- `openai.translation.prompt_append` adds extra prompt text for translator persona/domain guidance

## Subtitle Style

- Font size 12, white text with black outline
- Bottom-aligned, movie-style positioning
- Max 42 chars/line, 2 lines max
- Natural breaks at punctuation and pauses
