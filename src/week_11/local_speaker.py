"""Speak an existing Pi assistant reply over the current PipeWire Bluetooth sink.

This is output-only: no microphone, Ollama, Telegram API, or extra LLM call.
Environment:
  PI_SPEAKER_MODEL: optional voice .onnx override; defaults to Week 09 voice
  PI_SPEAKER_PIPER: optional standalone Piper executable; default: python -m piper
  PI_SPEAKER_PLAYER: playback executable (optional, defaults to pw-play)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def is_bluetooth_default(inspect_output: str) -> bool:
    """Only speak when PipeWire's default audio sink is a Bluetooth device."""
    info = inspect_output.lower()
    return "bluez_output" in info or "device.api = \"bluez5\"" in info or "api.bluez5" in info


def voice_text(text: str) -> str:
    """Strip markup that would sound awkward, without changing the answer's facts."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*#_>|]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1000].strip()


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "week_09" / "voices" / "en_US-lessac-medium.onnx"


def speak(text: str, *, check_sink: bool = True) -> str:
    text = voice_text(text)
    if not text:
        return "skipped: empty assistant response"

    if check_sink:
        if not shutil.which("wpctl"):
            return "skipped: wpctl unavailable; Bluetooth routing cannot be verified"
        check = subprocess.run(
            ["wpctl", "inspect", "@DEFAULT_AUDIO_SINK@"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if check.returncode != 0 or not is_bluetooth_default(check.stdout):
            return "skipped: Bluetooth speaker is not the default audio sink"

    model = Path(os.environ.get("PI_SPEAKER_MODEL") or DEFAULT_MODEL).expanduser()
    if not model.is_file():
        raise RuntimeError(f"Piper voice model missing: {model}. Set PI_SPEAKER_MODEL to override.")

    player = os.environ.get("PI_SPEAKER_PLAYER", "pw-play")
    if not shutil.which(player) and not Path(player).is_file():
        raise RuntimeError("pw-play not found; install PipeWire playback utilities or set PI_SPEAKER_PLAYER")

    with tempfile.TemporaryDirectory(prefix="week11-speaker-") as temp_dir:
        wav = Path(temp_dir) / "reply.wav"
        standalone_piper = os.environ.get("PI_SPEAKER_PIPER")
        if standalone_piper:
            # Optional compatibility with standalone Piper binaries.
            piper_command = [standalone_piper, "--model", str(model)]
        else:
            # Match Week 09: the virtualenv has piper-tts as a Python module,
            # not necessarily a `piper` executable on PATH.
            piper_command = [
                sys.executable, "-m", "piper",
                "--data-dir", str(model.parent), "--model", model.stem,
            ]
        tts = subprocess.run(
            [*piper_command, "--output_file", str(wav)],
            input=text, capture_output=True, text=True, timeout=90, check=False,
        )
        if tts.returncode != 0 or not wav.is_file():
            raise RuntimeError(f"Piper synthesis failed: {tts.stderr[-500:]}")
        playback = subprocess.run([player, str(wav)], capture_output=True, text=True, timeout=120, check=False)
        if playback.returncode != 0:
            raise RuntimeError(f"Bluetooth playback failed: {playback.stderr[-500:]}")

    return "spoken: Bluetooth speaker"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Already-generated assistant reply (or read stdin)")
    parser.add_argument("--no-sink-check", action="store_true", help="For manual output diagnostics only")
    args = parser.parse_args()
    text = args.text if args.text is not None else sys.stdin.read()
    try:
        print(speak(text, check_sink=not args.no_sink_check), flush=True)
        return 0
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        print(f"[local-speaker] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
