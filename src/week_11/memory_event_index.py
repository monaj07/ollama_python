"""Read-only, bounded, exact-path retrieval of person-positive observations.

The Pi LLM chooses which criteria to supply; filtering, event selection and
file-path retrieval are deterministic. No hardware calls, SQL generation or writes.
"""
from __future__ import annotations

import contextlib
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


LIMITATIONS = (
    "Person detections are tentative model outputs, not verified people or "
    "occupancy; detector scores are not calibrated probabilities."
)


def search_person_events(
    db_path: Path,
    *,
    limit: int = 12,
    min_score: float | None = None,
    event_ids: list[int] | None = None,
) -> dict:
    """Return newest matching positive events, including exact stored photo paths.

    min_score is inclusive. `limit` applies AFTER score/ID filtering.
    Only photos under outputs/observations are marked available or exposed.
    """
    if type(limit) is not int or not 1 <= limit <= 12:
        raise ValueError("events limit must be between 1 and 12")
    if min_score is not None and (
        type(min_score) not in (float, int)
        or not math.isfinite(min_score)
        or not 0 <= min_score <= 1
    ):
        raise ValueError("min_score must be a finite number from 0 to 1")
    if event_ids is not None and (
        type(event_ids) is not list
        or not 1 <= len(event_ids) <= 12
        or any(type(value) is not int or not 1 <= value <= 2147483647 for value in event_ids)
        or len(set(event_ids)) != len(event_ids)
    ):
        raise ValueError("event_ids must be 1-12 distinct positive integer IDs")

    db_path = Path(db_path)
    if not db_path.is_file():
        return {"ok": False, "operation": "events", "error": "No memory database exists"}

    filters = ["person_detected = 1"]
    params: list[int | float] = []
    if min_score is not None:
        filters.append("detection_score >= ?")
        params.append(min_score)
    if event_ids is not None:
        filters.append("id IN (" + ",".join("?" for _ in event_ids) + ")")
        params.extend(event_ids)
    where = " AND ".join(filters)
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        total_positive = db.execute(
            "SELECT COUNT(*) FROM observations WHERE person_detected = 1"
        ).fetchone()[0]
        total_matching = db.execute(
            "SELECT COUNT(*) FROM observations WHERE " + where, params
        ).fetchone()[0]
        rows = db.execute(
            "SELECT id, away_session_id, observed_at_utc, trigger, "
            "detection_score, evidence_path FROM observations WHERE " + where
            + " ORDER BY observed_at_utc DESC, id DESC LIMIT ?", [*params, limit]
        ).fetchall()

    approved_root = (db_path.parent.parent / "observations").resolve()
    results = []
    for row in rows:
        recorded_path = row["evidence_path"]
        path = Path(recorded_path).resolve() if recorded_path else None
        approved = path is not None and path.is_relative_to(approved_root)
        results.append({
            "event_id": row["id"],
            "session_id": row["away_session_id"],
            "trigger": row["trigger"],
            "time_local": datetime.fromisoformat(row["observed_at_utc"])
                .astimezone(ZoneInfo("Australia/Melbourne")).isoformat(timespec="seconds"),
            "score": row["detection_score"],
            "evidence_path": recorded_path if approved else None,
            "image_available": bool(approved and path.is_file()),
        })
    return {
        "ok": True, "operation": "events", "scope": "all_saved_observations",
        "total_positive_events": total_positive,
        "total_matching_events": total_matching,
        "returned_events": len(results),
        "events_truncated": total_matching > len(results),
        "events": results,
        "limitations": LIMITATIONS,
    }
