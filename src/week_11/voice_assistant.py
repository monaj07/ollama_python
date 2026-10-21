"""Week 11 voice: Week 09 Whisper/Piper I/O + Pi RPC agent, no custom Ollama loop.

Run with src/week_09/.venv/bin/python; uses the same microphone and speaker setup
as Weeks 09-10. Pi owns conversation state, grounding, routing and tool calls.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WEEK09 = HERE.parent / "week_09"
for directory in (str(REPO), str(WEEK09)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from src.week_09 import voice_assistant as voice09  # noqa: E402
from pi_rpc import PiRPCClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pi-model", default="ollama/gemma4:e2b-8k")
    parser.add_argument("--pi-continue", action="store_true", help="Resume last Pi session in this project")
    parser.add_argument("--pi-timeout", type=float, default=180.0)
    options, audio_args = parser.parse_known_args()

    # Keep the Week 09 loop unchanged: process_turn() calls these two helpers.
    # The Pi process and all its tools are initialized once, not per utterance.
    with PiRPCClient(model=options.pi_model, continue_session=options.pi_continue,
                     timeout_s=options.pi_timeout) as client:
        print("[startup] Persistent Pi RPC agent ready", flush=True)

        def ask_agent(question: str, _model: str) -> str:
            reply = client.ask(question)
            print(f"[pi_rpc] elapsed_s={reply.elapsed_s:.3f} "
                  f"tools={reply.tools} tool_s={reply.tool_seconds} "
                  f"input_tokens={reply.input_tokens} output_tokens={reply.output_tokens}",
                  flush=True)
            return reply.text

        voice09.ask_agent = ask_agent
        # Starting an extra Ollama client just for warm-up would be redundant;
        # first-turn loading is measured in Pi's stage instead.
        voice09.warm_up_model = lambda _model: None
        # Our repo ignores outputs/ but not src/week_09/outputs/. Keep recordings,
        # transcripts, and synthesized audio outside tracked source directories.
        if "--output-dir" not in audio_args:
            audio_args += ["--output-dir", str(REPO / "outputs" / "week_11" / "voice")]
        sys.argv = [sys.argv[0], *audio_args, "--model", options.pi_model]
        voice09.main()


if __name__ == "__main__":
    main()
