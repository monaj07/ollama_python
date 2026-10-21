"""Reuse Week 09 whisper.cpp transcription for one downloaded Telegram voice note.

Intentionally independent of Pi's agent/model process. Prints only the transcript to
stdout so a Pi input extension can replace the attachment metadata with speech.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

WEEK_11 = Path(__file__).resolve().parent
WEEK_09 = WEEK_11.parent / "week_09"
MAX_VOICE_BYTES = 10 * 1024 * 1024
MAX_DURATION_SECONDS = 60.0


def validated_voice_file(path: Path, allowed_root: Path) -> Path:
    """Allow only a real Telegram voice file inside the active Pi project directory."""
    root = allowed_root.resolve(strict=True)
    if path.is_symlink():
        raise ValueError("Voice files may not be symbolic links")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("Voice file must be inside the Pi working directory")
    if not resolved.name.endswith("-voice.oga"):
        raise ValueError("Not a Telegram voice attachment")
    info = resolved.stat()
    if not resolved.is_file() or not (0 < info.st_size <= MAX_VOICE_BYTES):
        raise ValueError("Voice file is empty or exceeds the size limit")
    return resolved


def check_duration(source: Path) -> None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
        check=True, capture_output=True, text=True, timeout=10,
    )
    duration = float(result.stdout.strip())
    if not (0 < duration <= MAX_DURATION_SECONDS):
        raise ValueError("Telegram voice duration exceeds the permitted range")


def transcribe_voice(source: Path, output_prefix: Path) -> str:
    # Reuse exactly the transcription interface already deployed in Week 09.
    sys.path.insert(0, str(WEEK_09))
    from voice_assistant import DEFAULT_WHISPER_DIR, transcribe

    with tempfile.TemporaryDirectory(prefix="tg-stt-") as tmp:
        wav = Path(tmp) / "voice.wav"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(source), "-t", str(int(MAX_DURATION_SECONDS)),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
            check=True, capture_output=True, text=True, timeout=35,
        )
        text = transcribe(
            wav,
            DEFAULT_WHISPER_DIR / "build/bin/whisper-cli",
            DEFAULT_WHISPER_DIR / "models/ggml-tiny.en.bin",
            output_prefix,
        )
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Whisper transcription of Telegram .oga voice")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        source = validated_voice_file(args.input, args.allowed_root)
        check_duration(source)
        with tempfile.TemporaryDirectory(prefix="tg-whisper-") as tmp:
            text = transcribe_voice(source, Path(tmp) / "voice")
        if not text:
            raise RuntimeError("Whisper produced an empty transcription")
        print(text)
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Voice transcription failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
