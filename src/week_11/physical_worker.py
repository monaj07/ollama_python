"""Week 11 JSONL bridge: Week 09 hardware + Week 10 observation memory.

Keep hardware dispatch in PhysicalTools and database semantics in RecordingTools.
The protocol intentionally exposes no shell, SQL, GPIO writes or deletions.
A Python worker remains alive throughout the Pi session, preserving warm YOLOE.
"""
from __future__ import annotations

import contextlib
import json
import signal
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WEEK10 = REPO / "src" / "week_10"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WEEK10))  # Week 10 modules use local imports.

from src.week_09.tools import PhysicalTools  # noqa: E402
from src.week_10.live_memory import RecordingTools  # noqa: E402
from src.week_10.memory_tools import (  # noqa: E402
    get_latest_away_report,
    get_observation_history_report,
)
from src.week_10.recent_memory import get_recent_observations_report  # noqa: E402


def dispatch(recording: RecordingTools, tool: str, args: object) -> dict:
    """Dispatch only allow-listed tools. No LLM-originated SQL or shell commands."""
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be a JSON object")
    if tool == "health":
        if args:
            raise ValueError("health takes no arguments")
        return {"ok": True, "status": "ready"}
    if tool == "begin_turn":
        if set(args) != {"session_id"}:
            raise ValueError("begin_turn requires session_id")
        session_id = args["session_id"]
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 128:
            raise ValueError("Invalid session_id")
        # Pi owns dialogue sessions. Their IDs label physical evidence in SQLite.
        recording.conversation_session_id = f"week11-pi:{session_id}"
        recording.begin_turn()
        return {"ok": True, "session_bound": True}
    if tool in ("detect_motion", "observe_scene"):
        result = recording.execute(tool, args)
        if tool == "observe_scene":
            # Memory writes are best-effort, and their status is never mistaken
            # for the result of the physical observation itself.
            result = {
                **result,
                "recording": recording.last_recording,
            }
            recording.begin_turn()  # Never reuse an old PIR reading next time.
        return result
    if tool == "memory_recent":
        if set(args) - {"limit"}:
            raise ValueError("Unknown memory_recent arguments")
        limit = args.get("limit", 5)
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("limit must be an integer from 1 to 10")
        return get_recent_observations_report(limit=limit)
    if tool in ("memory_history", "memory_away"):
        if args:
            raise ValueError(f"{tool} takes no arguments")
        return (get_observation_history_report() if tool == "memory_history"
                else get_latest_away_report())
    return {"ok": False, "error": "Unknown tool"}


def main() -> None:
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))
    hardware = PhysicalTools()
    recording = RecordingTools(hardware, "week11-pi:unbound")
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object")
                # stdout is exclusively the machine-readable protocol.
                with contextlib.redirect_stdout(sys.stderr):
                    result = dispatch(recording, request.get("tool"), request.get("args", {}))
                if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                    raise ValueError("Tool returned an invalid result")
            except Exception as exc:
                result = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        recording.close()


if __name__ == "__main__":
    main()
