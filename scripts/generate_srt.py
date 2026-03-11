#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "faster-whisper>=1.0.0",
#     "openai>=1.30.0",
#     "python-dotenv>=1.0.1",
# ]
# ///
"""
字幕生成与翻译脚本（核心流程保持：输入 -> 源字幕 -> 翻译 -> 输出）。

支持输入:
1. 音视频文件（先转录）
2. SRT 文件（直接作为源字幕）

默认输出:
1. 源语言 SRT（默认 ja）
2. 目标语言 SRT（默认 zh）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SOURCE_LANG = "ja"
DEFAULT_TARGET_LANG = "zh"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

SRT_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})$"
)
FONT_EXTENSIONS = {".ttf", ".otf", ".ttc", ".otc"}


@dataclass
class Subtitle:
    index: int
    start: float
    end: float
    text: str

    def to_srt(self) -> str:
        return (
            f"{self.index}\n"
            f"{format_srt_timestamp(self.start)} --> {format_srt_timestamp(self.end)}\n"
            f"{self.text}"
        )


@dataclass
class RuntimeConfig:
    source_lang: str
    target_lang: str
    openai_base_url: str | None
    openai_model: str
    openai_api_key: str | None
    translator_prompt: str | None
    use_turbo: bool


def log(message: str) -> None:
    print(message, file=sys.stderr)


def quote_ffmpeg_filter_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def escape_ass_style_value(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(",", "\\,")
        .replace("'", "\\'")
    )


def detect_project_fonts() -> tuple[Path | None, str | None]:
    project_root = Path(__file__).resolve().parent.parent
    fonts_dir = project_root / "fonts"
    if not fonts_dir.is_dir():
        return None, None

    font_files = sorted(
        file
        for file in fonts_dir.iterdir()
        if file.is_file() and file.suffix.lower() in FONT_EXTENSIONS
    )
    if not font_files:
        return None, None

    # 使用文件名推断 FontName，配合 fontsdir 提高命中概率。
    font_name = font_files[0].stem.replace("_", " ").strip()
    return fonts_dir, font_name or None


def resolve_ffmpeg_bin() -> str:
    env_bin = clean_optional_text(os.getenv("FFMPEG_BIN"))
    candidates = [
        env_bin,
        "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
        "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
        "ffmpeg",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.sep in candidate:
            if os.path.exists(candidate):
                return candidate
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return "ffmpeg"


def ffmpeg_has_filter(ffmpeg_bin: str, filter_name: str) -> bool:
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
    )
    output = f"{result.stdout}\n{result.stderr}"
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        columns = line.split()
        if len(columns) >= 2 and columns[1] == filter_name:
            return True
    return False


def format_srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_srt_timestamp(value: str) -> float:
    match = SRT_TIMESTAMP_PATTERN.match(value.strip())
    if not match:
        raise ValueError(f"无效的 SRT 时间戳: {value}")
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    millis = int(match.group("ms"))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def sanitize_translation_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return cleaned

    # 清理代码块包裹
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # 清理常见推理标签（重点处理 <think>...</think>）
    cleaned = re.sub(
        r"<\s*(think|analysis|reasoning)\b[^>]*>.*?<\s*/\s*\1\s*>",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"</?\s*(think|analysis|reasoning)\b[^>]*>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 清理常见“思考过程”行
    lines = [line.strip() for line in cleaned.split("\n")]
    meta_line_pattern = re.compile(
        r"^(思考过程|推理过程|分析过程|reasoning|analysis|thoughts?)\s*[:：]",
        re.IGNORECASE,
    )
    kept_lines = [line for line in lines if line and not meta_line_pattern.match(line)]
    cleaned = "\n".join(kept_lines).strip()

    return cleaned


def chunk_text_naturally(text: str, max_chars: int = 42) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    break_points = [
        "。", "，", "？", "！", "；", "：",
        ". ", ", ", "? ", "! ", "; ", ": ",
        " - ", " – ", " — ",
    ]
    lines: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            lines.append(remaining)
            break

        best_break = -1
        for marker in break_points:
            idx = remaining[:max_chars].rfind(marker)
            if idx > best_break:
                best_break = idx + len(marker)

        if best_break <= 0:
            best_break = remaining[:max_chars].rfind(" ")
        if best_break <= 0:
            best_break = max_chars

        lines.append(remaining[:best_break].strip())
        remaining = remaining[best_break:].strip()

    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]
    return lines


def merge_into_subtitles(
    segments: list,
    min_duration: float = 1.0,
    max_duration: float = 7.0,
    max_chars: int = 84,
) -> list[Subtitle]:
    if not segments:
        return []

    subtitles: list[Subtitle] = []
    current_text = ""
    current_start: float | None = None
    current_end: float | None = None

    for seg in segments:
        seg_text = seg.text.strip()
        seg_start = seg.start
        seg_end = seg.end

        if current_start is None:
            current_start = seg_start
            current_end = seg_end
            current_text = seg_text
            continue

        potential_text = f"{current_text} {seg_text}".strip()
        potential_duration = seg_end - current_start
        gap = seg_start - (current_end or seg_start)

        should_merge = (
            (
                potential_duration <= max_duration
                and len(potential_text) <= max_chars
                and ((current_end or current_start) - current_start) < min_duration
            )
            or (
                gap < 0.3
                and potential_duration <= max_duration
                and len(potential_text) <= max_chars
            )
        )

        if should_merge:
            current_text = potential_text
            current_end = seg_end
        else:
            lines = chunk_text_naturally(current_text)
            subtitles.append(
                Subtitle(
                    index=len(subtitles) + 1,
                    start=current_start,
                    end=current_end or current_start,
                    text="\n".join(lines),
                )
            )
            current_start = seg_start
            current_end = seg_end
            current_text = seg_text

    if current_text and current_start is not None:
        lines = chunk_text_naturally(current_text)
        subtitles.append(
            Subtitle(
                index=len(subtitles) + 1,
                start=current_start,
                end=current_end or current_start,
                text="\n".join(lines),
            )
        )

    return subtitles


def normalize_subtitles(subtitles: list[Subtitle]) -> list[Subtitle]:
    return [
        Subtitle(index=i + 1, start=sub.start, end=sub.end, text=sub.text.strip())
        for i, sub in enumerate(subtitles)
    ]


def subtitles_to_srt(subtitles: list[Subtitle]) -> str:
    normalized = normalize_subtitles(subtitles)
    if not normalized:
        return ""
    return "\n\n".join(sub.to_srt() for sub in normalized) + "\n"


def parse_srt_file(file_path: Path) -> list[Subtitle]:
    content = file_path.read_text(encoding="utf-8-sig")
    content = content.replace("\r\n", "\n").strip()
    if not content:
        raise ValueError("SRT 文件为空")

    blocks = re.split(r"\n\s*\n", content)
    subtitles: list[Subtitle] = []

    for block in blocks:
        lines = [line.rstrip() for line in block.split("\n") if line.strip()]
        if len(lines) < 2:
            continue

        ts_idx = 1 if lines[0].strip().isdigit() else 0
        if ts_idx >= len(lines) or "-->" not in lines[ts_idx]:
            continue

        start_str, end_str = [item.strip() for item in lines[ts_idx].split("-->", 1)]
        start = parse_srt_timestamp(start_str)
        end = parse_srt_timestamp(end_str)
        text = "\n".join(lines[ts_idx + 1:]).strip()
        if not text:
            continue

        subtitles.append(
            Subtitle(index=len(subtitles) + 1, start=start, end=end, text=text)
        )

    if not subtitles:
        raise ValueError("SRT 文件中没有可解析的字幕内容")
    return subtitles


def transcribe_media(
    file_path: str, language: str, use_turbo: bool = True
) -> tuple[list[Subtitle], float, str]:
    from faster_whisper import WhisperModel

    model_name = "large-v3-turbo" if use_turbo else "large-v3"
    log(f"加载转录模型: {model_name}")
    model = WhisperModel(model_name, device="auto", compute_type="auto")

    log(f"开始转录媒体文件: {file_path}")
    segments, info = model.transcribe(
        file_path,
        language=language,
        task="transcribe",
        word_timestamps=True,
        vad_filter=True,
    )

    raw_segments = list(segments)
    subtitles = merge_into_subtitles(raw_segments)
    detected_lang = info.language or language
    probability = getattr(info, "language_probability", 0.0)
    log(f"转录完成，识别语言: {detected_lang}，置信度: {probability:.0%}")
    log(f"源字幕条数: {len(subtitles)}")
    duration = subtitles[-1].end if subtitles else 0.0
    return subtitles, duration, detected_lang


def extract_json_array(raw_text: str) -> list[str]:
    if not raw_text:
        raise ValueError("模型未返回内容")

    raw_text = raw_text.strip()
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass

    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("返回结果中未找到 JSON 数组")

    parsed = json.loads(raw_text[start:end + 1])
    if not isinstance(parsed, list):
        raise ValueError("JSON 结构不是数组")
    return [str(item) for item in parsed]


def build_translation_system_prompt(base_prompt: str, custom_prompt: str | None) -> str:
    if not custom_prompt:
        return base_prompt
    return f"{base_prompt}\n\n附加翻译要求：\n{custom_prompt}"


def translate_single_text(
    client,
    model: str,
    source_lang: str,
    target_lang: str,
    text: str,
    custom_prompt: str | None,
) -> str:
    system_prompt = build_translation_system_prompt(
        "你是字幕翻译助手。只返回翻译后的文本，保持原有换行，不要添加解释。",
        custom_prompt=custom_prompt,
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"请将以下字幕从 {source_lang} 翻译为 {target_lang}：\n\n{text}"
                ),
            },
        ],
    )
    if not response.choices or response.choices[0].message.content is None:
        raise RuntimeError("单条翻译失败：模型返回为空")
    return sanitize_translation_text(response.choices[0].message.content)


def translate_text_batch(
    client,
    model: str,
    source_lang: str,
    target_lang: str,
    texts: list[str],
    custom_prompt: str | None,
) -> list[str]:
    system_prompt = build_translation_system_prompt(
        (
            "你是专业字幕翻译助手。你会收到一个字幕文本数组。"
            "请按原顺序逐条翻译，保留每条中的换行。"
            "只返回 JSON 数组，不要输出额外文字。"
        ),
        custom_prompt=custom_prompt,
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"请把下面的字幕从 {source_lang} 翻译为 {target_lang}。\n"
                    f"输入数组：\n{json.dumps(texts, ensure_ascii=False)}"
                ),
            },
        ],
    )
    content = ""
    if response.choices and response.choices[0].message.content is not None:
        content = response.choices[0].message.content

    try:
        translated = extract_json_array(content)
        if len(translated) != len(texts):
            raise ValueError("批量翻译条目数量不匹配")
        sanitized = [sanitize_translation_text(item) for item in translated]
        normalized: list[str] = []
        for original_text, translated_text in zip(texts, sanitized):
            if translated_text:
                normalized.append(translated_text)
                continue
            log("检测到空翻译结果，回退为单条重试。")
            retry_text = translate_single_text(
                client=client,
                model=model,
                source_lang=source_lang,
                target_lang=target_lang,
                text=original_text,
                custom_prompt=custom_prompt,
            )
            normalized.append(retry_text or original_text)
        return normalized
    except Exception:
        log("批量翻译解析失败，改为逐条翻译。")
        return [
            translate_single_text(
                client=client,
                model=model,
                source_lang=source_lang,
                target_lang=target_lang,
                text=text,
                custom_prompt=custom_prompt,
            ) or text
            for text in texts
        ]


def translate_subtitles(
    subtitles: list[Subtitle],
    config: RuntimeConfig,
    batch_size: int = 20,
) -> list[Subtitle]:
    if config.source_lang == config.target_lang:
        log("源语言与目标语言一致，跳过翻译。")
        return normalize_subtitles(subtitles)

    if not config.openai_api_key:
        raise RuntimeError(
            "缺少 OpenAI API Key。请通过 --openai-api-key 或 OPENAI_API_KEY 提供。"
        )

    from openai import OpenAI

    kwargs = {"api_key": config.openai_api_key}
    if config.openai_base_url:
        kwargs["base_url"] = config.openai_base_url
    client = OpenAI(**kwargs)

    translated_subtitles: list[Subtitle] = []
    total = len(subtitles)
    for start_idx in range(0, total, batch_size):
        chunk = subtitles[start_idx:start_idx + batch_size]
        log(
            f"翻译进度: {start_idx + 1}-{start_idx + len(chunk)} / {total}"
        )
        translated_texts = translate_text_batch(
            client=client,
            model=config.openai_model,
            source_lang=config.source_lang,
            target_lang=config.target_lang,
            texts=[item.text for item in chunk],
            custom_prompt=config.translator_prompt,
        )
        for item, translated_text in zip(chunk, translated_texts):
            translated_subtitles.append(
                Subtitle(
                    index=len(translated_subtitles) + 1,
                    start=item.start,
                    end=item.end,
                    text=translated_text,
                )
            )

    return translated_subtitles


def embed_subtitles(
    video_path: str,
    srt_content: str,
    output_path: str,
    subtitle_lang: str,
    burn: bool = False,
) -> None:
    ffmpeg_bin = resolve_ffmpeg_bin()
    log(f"使用 ffmpeg: {ffmpeg_bin}")

    with tempfile.NamedTemporaryFile("w", suffix=".srt", encoding="utf-8", delete=False) as temp_srt:
        temp_srt.write(srt_content)
        srt_path = temp_srt.name

    try:
        if burn:
            log("开始烧录字幕到视频。")
            if not ffmpeg_has_filter(ffmpeg_bin, "subtitles"):
                raise RuntimeError(
                    "当前 ffmpeg 不支持 subtitles 过滤器，无法烧录字幕。\n"
                    f"当前 ffmpeg: {ffmpeg_bin}\n"
                    "请安装 ffmpeg-full（含 libass）后重试，或通过 FFMPEG_BIN "
                    "指定支持 subtitles 的 ffmpeg 可执行文件。"
                )
            filter_options = [f"filename={quote_ffmpeg_filter_value(srt_path)}"]
            fonts_dir, font_name = detect_project_fonts()
            if fonts_dir:
                filter_options.append(
                    f"fontsdir={quote_ffmpeg_filter_value(str(fonts_dir))}"
                )
                log(f"烧录字幕将加载字体目录: {fonts_dir}")
            else:
                log("未检测到项目 fonts 目录中的字体文件，使用系统默认字体。")

            style_parts = []
            if font_name:
                style_parts.append(f"Fontname={escape_ass_style_value(font_name)}")
                log(f"烧录字幕优先使用字体: {font_name}")
            style_parts.extend(
                [
                    "FontSize=12",
                    "PrimaryColour=&Hffffff",
                    "OutlineColour=&H000000",
                    "BorderStyle=1",
                    "Outline=1",
                    "Shadow=0",
                    "MarginV=12",
                    "Alignment=2",
                ]
            )
            filter_options.append(
                f"force_style={quote_ffmpeg_filter_value(','.join(style_parts))}"
            )
            filter_str = f"subtitles={':'.join(filter_options)}"
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                video_path,
                "-vf",
                filter_str,
                "-c:a",
                "copy",
                output_path,
            ]
        else:
            log("开始封装软字幕到视频。")
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                video_path,
                "-i",
                srt_path,
                "-c",
                "copy",
                "-c:s",
                "mov_text",
                "-metadata:s:s:0",
                f"language={subtitle_lang}",
                output_path,
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 执行失败: {result.stderr.strip()}")
        log(f"视频输出完成: {output_path}")
    finally:
        if os.path.exists(srt_path):
            os.unlink(srt_path)


def is_srt_input(file_path: Path) -> bool:
    return file_path.suffix.lower() == ".srt"


def load_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    from dotenv import load_dotenv

    load_dotenv(override=False)
    source_lang = args.source_lang or os.getenv("SOURCE_LANG", DEFAULT_SOURCE_LANG)
    target_lang = args.target_lang or os.getenv("TARGET_LANG", DEFAULT_TARGET_LANG)
    openai_base_url = args.openai_base_url or os.getenv("OPENAI_BASE_URL")
    openai_model = args.openai_model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    translator_prompt = clean_optional_text(
        args.translator_prompt or os.getenv("TRANSLATOR_PROMPT")
    )

    return RuntimeConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        openai_api_key=openai_api_key,
        translator_prompt=translator_prompt,
        use_turbo=not args.accurate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将媒体或 SRT 生成双语字幕（默认 ja -> zh）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s input.mp4
  %(prog)s input.srt
  %(prog)s input.mp4 --burn
  %(prog)s input.mp4 --openai-model gpt-4o-mini
  %(prog)s input.mp4 --openai-base-url https://your.endpoint/v1 --openai-api-key xxx
        """,
    )
    parser.add_argument("file", help="输入文件路径（音视频或 .srt）")
    parser.add_argument(
        "--source-lang",
        help=f"源语言代码（默认 {DEFAULT_SOURCE_LANG}，支持 env: SOURCE_LANG）",
    )
    parser.add_argument(
        "--target-lang",
        help=f"目标语言代码（默认 {DEFAULT_TARGET_LANG}，支持 env: TARGET_LANG）",
    )
    parser.add_argument(
        "--accurate",
        action="store_true",
        help="转录使用 large-v3 模型（更慢但更准确）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="输出目录（默认与输入文件同目录）",
    )
    parser.add_argument(
        "--openai-base-url",
        help="OpenAI 兼容端点（支持 env: OPENAI_BASE_URL）",
    )
    parser.add_argument(
        "--openai-model",
        help=f"OpenAI 模型名（默认 {DEFAULT_OPENAI_MODEL}，支持 env: OPENAI_MODEL）",
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API Key（支持 env: OPENAI_API_KEY）",
    )
    parser.add_argument(
        "--translator-prompt",
        help="翻译附加提示词（支持 env: TRANSLATOR_PROMPT）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="批量翻译的字幕条数（默认 20）",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--embed",
        action="store_true",
        help="封装软字幕到视频（可在播放器开关）",
    )
    mode_group.add_argument(
        "--burn",
        action="store_true",
        help="烧录硬字幕到视频（始终可见）",
    )

    args = parser.parse_args()
    input_path = Path(args.file)

    if not input_path.exists():
        log(f"错误: 文件不存在: {args.file}")
        sys.exit(1)
    if args.batch_size <= 0:
        log("错误: --batch-size 必须大于 0")
        sys.exit(1)

    config = load_runtime_config(args)
    log(
        "运行配置: "
        f"source_lang={config.source_lang}, "
        f"target_lang={config.target_lang}, "
        f"openai_model={config.openai_model}"
    )
    if config.translator_prompt:
        log("已启用自定义翻译附加提示词。")
    if config.openai_base_url:
        log(f"使用自定义 OpenAI 端点: {config.openai_base_url}")

    if is_srt_input(input_path) and (args.embed or args.burn):
        log("错误: 输入为 SRT 文件时，不能使用 --embed 或 --burn")
        sys.exit(1)

    try:
        if is_srt_input(input_path):
            log("检测到 SRT 输入，跳过转录阶段。")
            source_subtitles = parse_srt_file(input_path)
            log(f"读取源字幕完成，条数: {len(source_subtitles)}")
        else:
            log("检测到媒体输入，开始转录。")
            source_subtitles, _, detected_lang = transcribe_media(
                str(input_path),
                language=config.source_lang,
                use_turbo=config.use_turbo,
            )
            log(f"转录阶段完成（识别语言: {detected_lang}）")

        if not source_subtitles:
            raise RuntimeError("没有可用的源字幕数据")

        target_subtitles = translate_subtitles(
            subtitles=source_subtitles,
            config=config,
            batch_size=args.batch_size,
        )
        log(f"翻译阶段完成，目标字幕条数: {len(target_subtitles)}")

        output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = input_path.stem
        source_srt_path = output_dir / f"{base_name}.{config.source_lang}.srt"
        target_srt_path = output_dir / f"{base_name}.{config.target_lang}.srt"

        source_srt_path.write_text(subtitles_to_srt(source_subtitles), encoding="utf-8")
        target_srt_path.write_text(subtitles_to_srt(target_subtitles), encoding="utf-8")

        log(f"已输出源语言字幕: {source_srt_path}")
        log(f"已输出目标语言字幕: {target_srt_path}")

        if args.embed or args.burn:
            output_video = output_dir / f"{base_name}_subtitled.mp4"
            embed_subtitles(
                video_path=str(input_path),
                srt_content=subtitles_to_srt(target_subtitles),
                output_path=str(output_video),
                subtitle_lang=config.target_lang,
                burn=args.burn,
            )
    except Exception as exc:
        log(f"执行失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
