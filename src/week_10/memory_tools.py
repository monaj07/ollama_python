"""Retrieve recorded observations and manage Away Mode session data."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from memory_store import connect

LOCAL_TZ = ZoneInfo("Australia/Melbourne")
MAX_POSITIVE_EVENTS = 12
OBSERVATIONS_DIR = Path(__file__).resolve().parents[2] / "outputs/observations"


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
    """Return complete counts and a bounded list of positive detections."""
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

        positive_rows = db.execute("""
            SELECT id, observed_at_utc, detection_score
            FROM observations
            WHERE away_session_id=? AND person_detected=1
            ORDER BY observed_at_utc ASC, id ASC LIMIT ?
        """, (session_id, MAX_POSITIVE_EVENTS)).fetchall()

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
        "positive_events_truncated": counts["positive"] > len(positive_rows),
        "positive_events": [
            {
                "event_id": row["id"],
                "time_local": local_time(row["observed_at_utc"]),
                "score": row["detection_score"],
            }
            for row in positive_rows
        ],
        "limitations": (
            "Detections are unverified model outputs; a single person may produce "
            "several events. No detection does not prove absence. Observations "
            "do not establish identity or proximity to the desk."
        ),
    }


def get_observation_history_report() -> dict:
    """Summarise all saved observations, with Away Mode history explicitly scoped."""
    away = get_latest_away_report()
    with closing(connect()) as db:
        db.execute("BEGIN")  # Counts and examples use the same read snapshot.
        counts = db.execute("""
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(CASE WHEN trigger='pir_motion' THEN 1 ELSE 0 END), 0) AS motion,
                   COALESCE(SUM(CASE WHEN person_detected=1 THEN 1 ELSE 0 END), 0) AS positive,
                   COALESCE(SUM(CASE WHEN person_detected=0 THEN 1 ELSE 0 END), 0) AS negative,
                   COALESCE(SUM(CASE WHEN person_detected IS NULL THEN 1 ELSE 0 END), 0) AS unknown,
                   MAX(CASE WHEN person_detected=1 THEN detection_score END) AS max_score,
                   MIN(CASE WHEN person_detected=1 THEN observed_at_utc END) AS first_positive,
                   MAX(CASE WHEN person_detected=1 THEN observed_at_utc END) AS last_positive
            FROM observations
        """).fetchone()
        rows = db.execute("""
            SELECT id, away_session_id, observed_at_utc, trigger, detection_score
            FROM observations WHERE person_detected=1
            ORDER BY observed_at_utc DESC, id DESC LIMIT ?
        """, (MAX_POSITIVE_EVENTS,)).fetchall()
    return {
        "ok": counts["total"] > 0 or away["ok"],
        "scope": "all_saved_observations",
        "total_events": counts["total"],
        "pir_motion_events": counts["motion"],
        "person_positive_events": counts["positive"],
        "person_negative_events": counts["negative"],
        "events_without_person_result": counts["unknown"],
        "max_person_confidence": counts["max_score"],
        "first_person_detection_local": local_time(counts["first_positive"]),
        "last_person_detection_local": local_time(counts["last_positive"]),
        "positive_events_truncated": counts["positive"] > len(rows),
        "positive_events": [
            {"event_id": row["id"], "session_id": row["away_session_id"],
             "trigger": row["trigger"], "time_local": local_time(row["observed_at_utc"]),
             "score": row["detection_score"]}
            for row in rows
        ],
        "latest_away_session": away,
        "limitations": (
            "Top-level counts and times cover all saved observations, including live checks. "
            "Use latest_away_session only for questions specifically about that Away Mode session. "
            "Positive examples are newest first. Detections are unverified predictions, not "
            "identity or proof of occupancy. A negative snapshot does not prove absence."
        ),
    }

def get_latest_live_observation(conversation_session_id: str) -> dict | None:
    """Retrieve the latest stored live camera observation for one conversation."""
    if not isinstance(conversation_session_id, str) or not conversation_session_id:
        raise ValueError("A conversation session ID is required")

    with closing(connect()) as db:
        row = db.execute("""
            SELECT id, observed_at_utc, detection_score, evidence_path,
                   json_extract(details_json, '$.camera_ok') AS camera_ok,
                   json_extract(details_json, '$.person_count') AS person_count,
                   json_extract(details_json, '$.pir_signal_active') AS pir_signal_active
            FROM observations
            WHERE trigger = 'live_observation'
              AND json_valid(details_json) = 1
              AND json_extract(details_json, '$.conversation_session_id') = ?
            ORDER BY observed_at_utc DESC, id DESC
            LIMIT 1
        """, (conversation_session_id,)).fetchone()

    if row is None:
        return None
    camera_ok = row["camera_ok"] == 1
    count = row["person_count"]
    if not camera_ok or type(count) is not int or count < 0:
        count = None
    pir = row["pir_signal_active"]
    motion = bool(pir) if type(pir) is int and pir in (0, 1) else None
    return {
        "event_id": row["id"],
        "time_local": local_time(row["observed_at_utc"]),
        "camera_ok": camera_ok,
        "person_count": count,
        "detection_score": row["detection_score"] if count and count > 0 else None,
        "pir_signal_active": motion,
        "evidence_path": row["evidence_path"],
    }
