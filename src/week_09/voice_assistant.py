"""Week 09: offline speech -> Ollama -> speech, with per-stage timings.

Run on a Raspberry Pi with local whisper.cpp, Piper and Ollama installed.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import wave
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ollama import Client

from agent import VoiceAgent


HERE = Path(__file__).resolve().parent
DEFAULT_WHISPER_DIR = Path.home() / "opt" / "whisper.cpp"


@contextmanager
def timed(name: str, timings: dict[str, float]):
    start = time.perf_counter()
    try:
        yield
    finally:
        seconds = time.perf_counter() - start
        timings[name] = round(seconds, 3)
        print(f"[time] {name}: {seconds:.3f} s", flush=True)


def checked_run(command: list[str], timeout: int) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "No error details").strip()
        raise RuntimeError(f"{command[0]} failed: {details[-1500:]}") from exc


def validate_wav(audio_path: Path) -> None:
    """whisper-cli requires compatible uncompressed WAV input."""
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            good = (
                wav_file.getnchannels() == 1
                and wav_file.getframerate() == 16000
                and wav_file.getsampwidth() == 2
                and wav_file.getcomptype() == "NONE"
                and wav_file.getnframes() > 0
            )
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError(f"Not a readable WAV file: {audio_path}") from exc
    if not good:
        raise ValueError(
            "Audio must be nonempty, mono, 16 kHz, 16-bit PCM WAV. "
            "Convert with: ffmpeg -i input.wav -ar 16000 -ac 1 "
            "-c:a pcm_s16le converted.wav"
        )


def record_audio(audio_path: Path, device: str, seconds: int) -> Path:
    print(f"Recording {seconds} seconds from ALSA device {device!r}. Speak now...", flush=True)
    checked_run(
        [
            "arecord", "-q", "-D", device, "-t", "wav", "-f", "S16_LE",
            "-r", "16000", "-c", "1", "-d", str(seconds), str(audio_path),
        ],
        timeout=seconds + 15,
    )
    validate_wav(audio_path)
    return audio_path


def valid_question(text: str) -> str:
    text = " ".join(text.split()).strip()
    markers = {"[blank_audio]", "[silence]", "(silence)", "[music]", "[noise]", "[applause]"}
    if not text or text.casefold() in markers or not re.search(r"[\w\d]", text, re.UNICODE):
        raise ValueError("No intelligible speech was transcribed; not sending to Ollama.")
    return text


def transcribe(audio_path: Path, whisper_bin: Path, whisper_model: Path, prefix: Path) -> str:
    if not whisper_bin.is_file() or not whisper_model.is_file():
        raise FileNotFoundError(f"Missing whisper-cli ({whisper_bin}) or model ({whisper_model})")
    checked_run(
        [
            str(whisper_bin), "-m", str(whisper_model), "-f", str(audio_path),
            "-t", "4", "-l", "en", "--output-txt", "--output-file",
            str(prefix), "--no-prints",
        ],
        timeout=180,
    )
    transcript_file = Path(f"{prefix}.txt")
    if not transcript_file.is_file():
        raise RuntimeError("Whisper did not create the expected transcript file")
    return valid_question(transcript_file.read_text(encoding="utf-8"))


def warm_up_model(model: str) -> None:
    print(f"[startup] Loading {model} into memory...", flush=True)
    start = time.perf_counter()

    client = Client(
        host="http://127.0.0.1:11434",
        timeout=180,
    )

    client.chat(
        model=model,
        messages=[],
        think=False,
        options={
            "temperature": 0,
            "num_ctx": 2048,
            "num_predict": 100,
        },
        keep_alive="15m",
    )

    print(
        f"[startup] Model ready in "
        f"{time.perf_counter() - start:.3f} s",
        flush=True,
    )


_agent: VoiceAgent | None = None


def ask_agent(question: str, model: str) -> str:
    global _agent

    if _agent is None or _agent.model != model:
        _agent = VoiceAgent(model=model)

    return _agent.ask(question)


def synthesize(text: str, wav_path: Path, voice_dir: Path, voice: str) -> Path:
    if not (voice_dir / f"{voice}.onnx").is_file():
        raise FileNotFoundError(f"Missing Piper voice: {voice_dir / (voice + '.onnx')}")
    checked_run(
        [
            sys.executable, "-m", "piper", "--data-dir", str(voice_dir),
            "-m", voice, "-f", str(wav_path), "--", text,
        ],
        timeout=120,
    )
    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        raise RuntimeError("Piper did not produce a valid WAV output file")
    return wav_path


def play_audio(audio_path: Path) -> None:
    # paplay uses PipeWire/PulseAudio's default sink (our Bluetooth speaker).
    checked_run(["paplay", str(audio_path)], timeout=120)


def process_turn(args):
    # Generate unique files for this turn
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid4().hex[:6]
    )

    audio_file = args.output_dir / f"{run_id}_question.wav"
    output_file = args.output_dir / f"{run_id}_reply.wav"
    transcript_prefix = args.output_dir / f"{run_id}_transcript"

    # Fresh metrics for each turn
    timings = {}
    start = time.perf_counter()

    # Keep your existing input selection logic
    if args.text is not None:
        question = valid_question(args.text)
        print("[input] Text supplied; skipping capture and transcription.")

    else:
        if args.audio is not None:
            audio_file = args.audio.expanduser().resolve()
            validate_wav(audio_file)
            print(f"[input] Pre-recorded audio: {audio_file}")
        else:
            with timed("record_s", timings):
                record_audio(audio_file, args.mic_device, args.duration)

        with timed("stt_s", timings):
            question = transcribe(
                audio_file,
                args.whisper_bin.expanduser(),
                args.whisper_model.expanduser(),
                transcript_prefix,
            )

    print(f"[user] {question}")

    with timed("llm_s", timings):
        answer = ask_agent(question, args.model)

    print(f"[assistant] {answer}")

    with timed("tts_s", timings):
        synthesize(
            answer,
            output_file,
            args.voice_dir.expanduser().resolve(),
            args.voice,
        )

    print(f"[audio] {output_file}")

    if not args.no_play:
        with timed("play_s", timings):
            play_audio(output_file)

    timings["total_s"] = round(time.perf_counter() - start, 3)
    print("[timings] " + json.dumps(timings, indent=2))

    with (args.output_dir / "runs.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({
            "run_id": run_id,
            "model": args.model,
            "timings": timings,
        }) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--audio", type=Path, help="Existing 16 kHz mono PCM WAV file")
    source.add_argument("--text", help="Skip microphone and STT; useful for testing Ollama + Piper")
    parser.add_argument("--mic-device", default="default", help="ALSA microphone device, e.g. plughw:1,0")
    parser.add_argument("--duration", type=int, default=5, help="Recording seconds (default: 5)")
    parser.add_argument("--model", default="gemma4:e2b", help="Locally installed Ollama model")
    parser.add_argument("--whisper-bin", type=Path, default=DEFAULT_WHISPER_DIR / "build/bin/whisper-cli")
    parser.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER_DIR / "models/ggml-tiny.en.bin")
    parser.add_argument("--voice-dir", type=Path, default=HERE / "voices")
    parser.add_argument("--voice", default="en_US-lessac-medium")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--no-play", action="store_true", help="Write synthesized WAV without speaker playback")
    args = parser.parse_args()

    if args.duration < 1:
        parser.error("--duration must be at least 1 second")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load the model once at startup.
    warm_up_model(args.model)

    # Preserve one-shot test modes.
    one_shot = args.text is not None or args.audio is not None

    while True:
        try:
            if not one_shot:
                command = input(
                    "\nPress Enter to speak, or 'q' to quit: "
                ).strip().lower()

                if command in {"q", "quit", "exit"}:
                    break

            process_turn(args)

        except (KeyboardInterrupt, EOFError):
            print("\n[assistant] Goodbye!")
            break

        except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
            print(f"[error] {exc}", file=sys.stderr)

            if one_shot:
                sys.exit(1)

            # Continue accepting requests after a failed turn.
            continue

        if one_shot:
            break


if __name__ == "__main__":
    main()
