"""Retrieve Away Mode reports and remove active session data."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from memory_store import connect

LOCAL_TZ = ZoneInfo("Australia/Melbourne")
OBSERVATIONS_DIR = Path(__file__).resolve().parents[2] / "outputs/observations"

SCHEMAS = [{
    "type": "function",
    "function": {
        "name": "get_latest_away_report",
        "description": (
            "Retrieve stored evidence from the most recent Away Mode session, "
            "including motion events, person detections, times and confidence. "
            "Use for questions about what happened while the monitor was armed. "
            "Does not read live sensors or capture a new image."
        ),
        "parameters": {
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
        },
    },
}]


def local_time(utc_timestamp: str | None) -> str | None:
    if utc_timestamp is None:
        return None
    return datetime.fromisoformat(utc_timestamp).astimezone(LOCAL_TZ).isoformat(
        timespec="seconds"
    )


def delete_latest_away_report() -> dict:
    """Delete the latest active session, its observations and capture folders.

    Stop the event monitor before calling. Shared capture folders are retained.
    File cleanup follows the database commit; any cleanup failures are returned
    with their paths so they can be retried without deleting another session.
    """
    with closing(connect()) as db:
        with db:
            db.execute("BEGIN IMMEDIATE")
            session = db.execute("""
                SELECT id FROM away_sessions
                WHERE ended_at_utc IS NULL
                ORDER BY started_at_utc DESC, id DESC LIMIT 1
            """).fetchone()
            if session is None:
                return {"ok": False, "error": "No active Away Mode session is recorded."}

            session_id = session["id"]
            rows = db.execute("""
                SELECT evidence_path FROM observations
                WHERE away_session_id = ? AND evidence_path IS NOT NULL
            """, (session_id,)).fetchall()
            root = OBSERVATIONS_DIR.resolve()
            folders = set()
            for row in rows:
                source = Path(row["evidence_path"]).absolute()
                folder = source.parent
                # Each capture owns one direct child folder containing scene.jpg.
                # Never follow a symlink or delete outside the capture root.
                if (source.name != "scene.jpg" or source.is_symlink()
                        or folder.is_symlink() or folder.resolve().parent != root):
                    raise ValueError(f"Unsupported evidence path: {source}")
                folders.add(folder.resolve())

            other_rows = db.execute("""
                SELECT evidence_path FROM observations
                WHERE (away_session_id IS NULL OR away_session_id != ?)
                    AND evidence_path IS NOT NULL
            """, (session_id,)).fetchall()
            shared = folders & {
                Path(row["evidence_path"]).resolve().parent for row in other_rows
            }
            cursor = db.execute("""
                DELETE FROM observations
                WHERE away_session_id = ?
            """, (session_id,))
            deleted = cursor.rowcount
            db.execute("DELETE FROM away_sessions WHERE id = ?", (session_id,))

    removed = []
    cleanup_errors = []
    for folder in sorted(folders - shared):
        try:
            if folder.is_symlink():
                raise ValueError("Capture folder was replaced by a symlink")
            shutil.rmtree(folder)
        except FileNotFoundError:
            pass  # Assets may already have been removed manually.
        except (OSError, ValueError) as exc:
            cleanup_errors.append({"path": str(folder), "error": str(exc)})
        else:
            removed.append(str(folder))

    return {
        "ok": not cleanup_errors,
        "session_id": session_id,
        "session_deleted": True,
        "observations_deleted": deleted,
        "asset_directories_deleted": removed,
        "shared_asset_directories_retained": [str(folder) for folder in sorted(shared)],
        "asset_cleanup_errors": cleanup_errors,
    }


def get_latest_away_report() -> dict:
    """Summarise all events; bound the individual examples sent to the LLM."""
    with closing(connect()) as db:
        session = db.execute("""
            SELECT * FROM away_sessions
            ORDER BY started_at_utc DESC, id DESC LIMIT 1
        """).fetchone()
        if session is None:
            return {"ok": False, "error": "No Away Mode sessions are recorded."}

        session_id = session["id"]
        counts = db.execute("""
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(CASE WHEN trigger='pir_motion' THEN 1 ELSE 0 END), 0) AS motion,
                   COALESCE(SUM(CASE WHEN person_detected=1 THEN 1 ELSE 0 END), 0) AS positive,
                   COALESCE(SUM(CASE WHEN person_detected=0 THEN 1 ELSE 0 END), 0) AS negative,
                   COALESCE(SUM(CASE WHEN person_detected IS NULL THEN 1 ELSE 0 END), 0) AS unknown,
                   MAX(CASE WHEN person_detected=1 THEN detection_score END) AS max_score,
                   MIN(CASE WHEN person_detected=1 THEN observed_at_utc END) AS first_positive,
                   MAX(CASE WHEN person_detected=1 THEN observed_at_utc END) AS last_positive
            FROM observations WHERE away_session_id=?
        """, (session_id,)).fetchone()

        # Return recent event detail only; aggregate counts cover the entire session.
        rows = db.execute("""
            SELECT id, observed_at_utc, trigger, person_detected, detection_score
            FROM observations WHERE away_session_id=?
            ORDER BY observed_at_utc DESC, id DESC LIMIT 15
        """, (session_id,)).fetchall()

    return {
        "ok": True,
        "session_id": session_id,
        "session_active": session["ended_at_utc"] is None,
        "started_local": local_time(session["started_at_utc"]),
        "ended_local": local_time(session["ended_at_utc"]),
        "total_events": counts["total"],
        "pir_motion_events": counts["motion"],
        "person_positive_events": counts["positive"],
        "person_negative_events": counts["negative"],
        "events_without_camera_result": counts["unknown"],
        "max_person_confidence": counts["max_score"],
        "first_person_detection_local": local_time(counts["first_positive"]),
        "last_person_detection_local": local_time(counts["last_positive"]),
        "events_truncated": counts["total"] > len(rows),
        "events": [
            {
                "event_id": row["id"],
                "time_local": local_time(row["observed_at_utc"]),
                "trigger": row["trigger"],
                "person_detected": None if row["person_detected"] is None
                                   else bool(row["person_detected"]),
                "score": row["detection_score"],
            }
            for row in reversed(rows)
        ],
        "limitations": (
            "A person detection is an unverified model prediction, not an identity "
            "or proof of physical presence or proximity to a desk. A negative detection is not proof "
            "of absence. Motion events may refer to the same person. "
            "Away Mode indicates monitoring was armed, not verified user absence."
        ),
    }
