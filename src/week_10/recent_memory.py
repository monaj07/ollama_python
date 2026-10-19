"""Small read-only summary of the most recently recorded observations."""
from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from memory_store import recent_observations

MELBOURNE = ZoneInfo("Australia/Melbourne")


def get_recent_observations_report(limit: int = 5) -> dict:
    """Read recent events across sessions, including their negative detections."""
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("limit must be an integer between 1 and 10")
    rows = recent_observations(limit)
    events = []
    for row in rows:
        # Important: NULL, 0 and 1 have different meanings.
        detected = row["person_detected"]
        if detected not in (None, 0, 1):
            raise ValueError("Unexpected person_detected state")
        try:
            details = json.loads(row.get("details_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            details = {}
        if not isinstance(details, dict):
            details = {}
        stamp = datetime.fromisoformat(row["observed_at_utc"])
        if stamp.utcoffset() is None:
            raise ValueError("Observation timestamp requires a timezone")
        events.append({
            "event_id": row["id"],
            "session_id": row["away_session_id"],
            "time_local": stamp.astimezone(MELBOURNE).isoformat(timespec="seconds"),
            "trigger": row["trigger"],
            "person_detected": None if detected is None else bool(detected),
            "person_count": details.get("person_count"),
            "person_score": row["detection_score"],
            "camera_ok": details.get("camera_ok"),
        })
    return {
        "ok": bool(events),
        "count": len(events),
        "observations": events,  # newest first; first entry is MOST recent
        "limitations": (
            "Stored camera detections are unverified and may contain false positives "
            "or missed people. A person not detected does not prove absence. "
            "These are prior snapshots, not live camera measurements."
        ),
    }
