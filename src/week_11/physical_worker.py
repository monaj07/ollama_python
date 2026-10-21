"""Week 11 JSONL bridge: Week 09 hardware + Week 10 observation memory.

Keep hardware dispatch in PhysicalTools and database semantics in RecordingTools.
The protocol intentionally exposes no shell, SQL, GPIO writes or deletions.
A Python worker remains alive throughout the Pi session, preserving warm YOLOE.
"""
from __future__ import annotations

import contextlib
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
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
from memory_store import DB_PATH  # noqa: E402


MEMORY_LIMITATION = (
    "Detections are unverified model outputs, not distinct people or proof of occupancy. "
    "A negative snapshot does not prove absence."
)


def query_memory(args: dict) -> dict:
    """One read-only, bounded interface to the existing Week 10 memory."""
    operation = args.get("operation")
    allowed = {
        "recent": {"operation", "limit"},
        "summary": {"operation"},
        "away": {"operation"},
        "events": {"operation", "limit"},
        "event": {"operation", "event_id"},
    }
    if not isinstance(operation, str) or operation not in allowed:
        raise ValueError("operation must be recent, summary, away, events or event")
    if set(args) - allowed[operation]:
        raise ValueError(f"Unexpected arguments for memory operation {operation}")

    if operation in ("recent", "events"):
        maximum = 10 if operation == "recent" else 12
        limit = args.get("limit", 1 if operation == "recent" else 12)
        if type(limit) is not int or not 1 <= limit <= maximum:
            raise ValueError(f"{operation} limit must be an integer from 1 to {maximum}")
        if operation == "recent":
            return get_recent_observations_report(limit=limit)
        report = get_observation_history_report()
        events = report.get("positive_events", [])[:limit]
        return {
            "ok": report.get("ok", False),
            "operation": "events", "scope": "all_saved_observations",
            "total_positive_events": report.get("person_positive_events", 0),
            "returned_events": len(events),
            "events_truncated": report.get("person_positive_events", 0) > len(events),
            "events": events, "limitations": MEMORY_LIMITATION,
        }

    if operation == "summary":
        report = get_observation_history_report()
        return {
            "ok": report.get("ok", False), "operation": "summary",
            "scope": "all_saved_observations",
            "total_events": report.get("total_events", 0),
            "person_positive_events": report.get("person_positive_events", 0),
            "person_negative_events": report.get("person_negative_events", 0),
            "events_without_person_result": report.get("events_without_person_result", 0),
            "pir_motion_events": report.get("pir_motion_events", 0),
            "first_person_detection_local": report.get("first_person_detection_local"),
            "last_person_detection_local": report.get("last_person_detection_local"),
            "limitations": MEMORY_LIMITATION,
        }

    if operation == "away":
        # Reuse the Week 10 implementation; its event listing is already bounded.
        return get_latest_away_report()

    event_id = args.get("event_id")
    if type(event_id) is not int or not 1 <= event_id <= 2147483647:
        raise ValueError("event_id must be a positive integer")
    # This fixed SELECT and bound integer are the only new SQL in Week 11.
    # Open SQLite in read-only mode, without schema initialization or writes.
    if not DB_PATH.is_file():
        return {"ok": False, "operation": "event", "error": "No memory database exists"}
    uri = f"{DB_PATH.resolve().as_uri()}?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT id, away_session_id, observed_at_utc, trigger, "
            "person_detected, detection_score, evidence_path, details_json "
            "FROM observations WHERE id = ? LIMIT 1", (event_id,)
        ).fetchone()
    if row is None:
        return {"ok": False, "operation": "event", "event_id": event_id,
                "error": "No observation exists with that event ID"}
    try:
        details = json.loads(row["details_json"] or "{}")
    except (ValueError, TypeError):
        details = {}
    if not isinstance(details, dict):
        details = {}
    seen = row["person_detected"]
    timestamp = datetime.fromisoformat(row["observed_at_utc"])
    return {
        "ok": True, "operation": "event", "event_id": row["id"],
        "away_session_id": row["away_session_id"],
        "time_local": timestamp.astimezone(ZoneInfo("Australia/Melbourne")).isoformat(timespec="seconds"),
        "trigger": row["trigger"],
        "person_detected": None if seen is None else bool(seen),
        "person_score": row["detection_score"],
        "evidence_path": row["evidence_path"],
        "person_count": details.get("person_count"),
        "camera_ok": details.get("camera_ok"),
        "pir_motion_signal_active": details.get("pir_signal_active"),
        "conversation_session_id": details.get("conversation_session_id"),
        "limitations": MEMORY_LIMITATION,
    }


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
    if tool == "query_memory":
        return query_memory(args)
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
