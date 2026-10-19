"""Week 10 push-to-talk voice assistant with persistent observation memory.

Reuse the full Week 09 record -> Whisper -> agent -> Piper -> speaker path.
Only the agent is changed; no duplicate hardware or audio implementation.
"""
from __future__ import annotations

import argparse
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
_resume_last = False
_session_id: str | None = None


def ask_agent(question: str, model: str) -> str:
    global _agent
    if _agent is None or _agent.model != model:
        if _agent is not None:
            # Preserve the dialogue session if the model changes during a process.
            previous_id = _agent.session_id
            _agent.tools.close()
        else:
            previous_id = _session_id
        _agent = MemoryVoiceAgent(model=model, conversation_session_id=previous_id,
                                 resume_last=_resume_last if previous_id is None else False)
    return _agent.ask(question)


def main() -> None:
    global _resume_last, _session_id
    # Consume Week 10-only flags, then let Week 09 parse every audio flag.
    parser = argparse.ArgumentParser(add_help=False)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--resume-last", action="store_true")
    selection.add_argument("--session-id")
    options, remaining = parser.parse_known_args()
    _resume_last, _session_id = options.resume_last, options.session_id
    sys.argv = [sys.argv[0], *remaining]
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
