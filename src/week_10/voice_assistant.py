"""Week 10 push-to-talk voice assistant with persistent observation memory.

Reuse the full Week 09 record -> Whisper -> agent -> Piper -> speaker path.
Only the agent is changed; no duplicate hardware or audio implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WEEK09 = HERE.parent / "week_09"
for directory in (str(REPO), str(WEEK09)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from src.week_09 import voice_assistant as voice09
from src.week_10.agent import MemoryVoiceAgent

_agent: MemoryVoiceAgent | None = None


def ask_agent(question: str, model: str) -> str:
    global _agent
    if _agent is None or _agent.model != model:
        if _agent is not None:
            _agent.tools.close()
        _agent = MemoryVoiceAgent(model=model)
    return _agent.ask(question)


def main() -> None:
    # Week 09's process_turn calls voice09.ask_agent by name.
    # Replace only that callback; preserve all audio flags, timings and I/O.
    voice09.ask_agent = ask_agent
    try:
        voice09.main()
    finally:
        if _agent is not None:
            _agent.tools.close()


if __name__ == "__main__":
    main()
