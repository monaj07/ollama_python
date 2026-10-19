
"""SQLite event memory for the Week 10 Raspberry Pi agent."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "outputs/week_10/events.sqlite3"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS away_sessions (
            id INTEGER PRIMARY KEY,
            started_at_utc TEXT NOT NULL,
            ended_at_utc TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS one_active_away_session
            ON away_sessions ((1))
            WHERE ended_at_utc IS NULL;

        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY,
            away_session_id INTEGER REFERENCES away_sessions(id),
            observed_at_utc TEXT NOT NULL,
            trigger TEXT NOT NULL,
            person_detected INTEGER CHECK(person_detected IN (0, 1)),
            detection_score REAL CHECK(detection_score BETWEEN 0 AND 1),
            evidence_path TEXT,
            details_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS observations_by_time
            ON observations(observed_at_utc);
    """)
    return db



def add_observation(
    trigger: str,
    *,
    person_detected: bool | None = None,
    detection_score: float | None = None,
    evidence_path: str | None = None,
    away_session_id: int | None = None,
    details: dict | None = None,
    observed_at_utc: str | None = None,
) -> int:
    if not trigger.strip():
        raise ValueError("trigger cannot be empty")
    if detection_score is not None and not 0 <= detection_score <= 1:
        raise ValueError("detection_score must be between 0 and 1")

    with closing(connect()) as db:
        with db:
            db.execute("BEGIN IMMEDIATE")

            # Associate with the active session automatically.
            if away_session_id is None:
                active = db.execute("""
                    SELECT id FROM away_sessions
                    WHERE ended_at_utc IS NULL
                """).fetchone()
                away_session_id = active["id"] if active else None

            if observed_at_utc is None:
                timestamp = datetime.now(timezone.utc).isoformat(
                    timespec="microseconds"
                )
            else:
                parsed = datetime.fromisoformat(observed_at_utc)
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    raise ValueError("observed_at_utc must include a timezone")
                timestamp = parsed.astimezone(timezone.utc).isoformat(
                    timespec="microseconds"
                )

            cursor = db.execute("""
                INSERT INTO observations
                (away_session_id, observed_at_utc, trigger,
                 person_detected, detection_score,
                 evidence_path, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                away_session_id,
                timestamp,
                trigger,
                None if person_detected is None else int(person_detected),
                detection_score,
                evidence_path,
                json.dumps(details or {}),
            ))

            return cursor.lastrowid


def observations_for_session(session_id: int) -> list[dict]:
    """Retrieve all events belonging to an away session."""
    with closing(connect()) as db:
        rows = db.execute("""
            SELECT * FROM observations
            WHERE away_session_id = ?
            ORDER BY observed_at_utc ASC, id ASC
        """, (session_id,)).fetchall()

        return [dict(row) for row in rows]


def recent_observations(limit: int = 5) -> list[dict]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    with closing(connect()) as db:
        rows = db.execute("""
            SELECT * FROM observations
            ORDER BY observed_at_utc DESC, id DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]


def get_active_away_session() -> dict | None:
    """Return the active session, if one exists."""
    with closing(connect()) as db:
        row = db.execute("""
            SELECT * FROM away_sessions
            WHERE ended_at_utc IS NULL
            LIMIT 1
        """).fetchone()
        return dict(row) if row else None


def arm_away_mode() -> int:
    """Start an away session. Reject duplicate activation."""
    with closing(connect()) as db:
        with db:
            db.execute("BEGIN IMMEDIATE")

            active = db.execute("""
                SELECT id FROM away_sessions
                WHERE ended_at_utc IS NULL
            """).fetchone()

            if active:
                raise RuntimeError(
                    f"Away Mode is already active (session {active['id']})"
                )

            now = datetime.now(timezone.utc).isoformat(
                timespec="microseconds"
            )
            cursor = db.execute("""
                INSERT INTO away_sessions (started_at_utc)
                VALUES (?)
            """, (now,))
            return cursor.lastrowid


def disarm_away_mode() -> dict:
    """End the currently active away session."""
    with closing(connect()) as db:
        with db:
            db.execute("BEGIN IMMEDIATE")

            active = db.execute("""
                SELECT id FROM away_sessions
                WHERE ended_at_utc IS NULL
            """).fetchone()

            if not active:
                raise RuntimeError("Away Mode is not active")

            now = datetime.now(timezone.utc).isoformat(
                timespec="microseconds"
            )
            db.execute("""
                UPDATE away_sessions
                SET ended_at_utc = ?
                WHERE id = ?
            """, (now, active["id"]))

            row = db.execute("""
                SELECT * FROM away_sessions WHERE id = ?
            """, (active["id"],)).fetchone()

            return dict(row)
