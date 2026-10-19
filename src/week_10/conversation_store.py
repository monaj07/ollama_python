"""Persistent dialogue memory alongside, but separate from, sensor observations.

Stores complete successful user/assistant text exchanges, while model prompts use
only the most recent few turns. Does not store sensor evidence blobs or audio.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from memory_store import DB_PATH

LOCAL_TZ = ZoneInfo("Australia/Melbourne")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _local(timestamp: str) -> str:
    return datetime.fromisoformat(timestamp).astimezone(LOCAL_TZ).isoformat(
        timespec="seconds"
    )


class ConversationStore:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path is not None else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            with db:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS conversation_sessions (
                        id TEXT PRIMARY KEY,
                        started_at_utc TEXT NOT NULL,
                        last_activity_at_utc TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS conversation_turns (
                        id INTEGER PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES conversation_sessions(id)
                            ON DELETE CASCADE,
                        turn_index INTEGER NOT NULL,
                        created_at_utc TEXT NOT NULL,
                        user_text TEXT NOT NULL,
                        assistant_text TEXT NOT NULL,
                        route TEXT NOT NULL,
                        model TEXT NOT NULL,
                        UNIQUE(session_id, turn_index)
                    );
                    CREATE INDEX IF NOT EXISTS conversation_turns_by_session
                        ON conversation_turns(session_id, turn_index);
                    CREATE INDEX IF NOT EXISTS conversation_sessions_by_recency
                        ON conversation_sessions(last_activity_at_utc DESC);
                """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def new_session(self) -> str:
        session_id = uuid4().hex
        now = _now()
        with closing(self._connect()) as db:
            with db:
                db.execute("""
                    INSERT INTO conversation_sessions
                        (id, started_at_utc, last_activity_at_utc)
                    VALUES (?, ?, ?)
                """, (session_id, now, now))
        return session_id

    def require_session(self, session_id: str) -> str:
        if not session_id or not isinstance(session_id, str):
            raise ValueError("A valid conversation session ID is required")
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT id FROM conversation_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Conversation session not found: {session_id}")
        return row["id"]

    def latest_session_id(self) -> str | None:
        with closing(self._connect()) as db:
            row = db.execute("""
                SELECT s.id FROM conversation_sessions s
                WHERE EXISTS (SELECT 1 FROM conversation_turns t WHERE t.session_id=s.id)
                ORDER BY s.last_activity_at_utc DESC, s.rowid DESC LIMIT 1
            """).fetchone()
        return row["id"] if row else None

    def append_turn(
        self, session_id: str, user_text: str, assistant_text: str,
        route: str, model: str,
    ) -> int:
        if not user_text.strip() or not assistant_text.strip():
            raise ValueError("Only complete, non-empty dialogue turns can be saved")
        now = _now()
        with closing(self._connect()) as db:
            with db:
                # Lock before computing turn_index, to avoid duplicate indices.
                db.execute("BEGIN IMMEDIATE")
                session = db.execute(
                    "SELECT id FROM conversation_sessions WHERE id=?", (session_id,)
                ).fetchone()
                if session is None:
                    raise ValueError(f"Conversation session not found: {session_id}")
                index = db.execute("""
                    SELECT COALESCE(MAX(turn_index), 0) + 1
                    FROM conversation_turns WHERE session_id=?
                """, (session_id,)).fetchone()[0]
                db.execute("""
                    INSERT INTO conversation_turns
                        (session_id, turn_index, created_at_utc,
                         user_text, assistant_text, route, model)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (session_id, index, now, user_text, assistant_text, route, model))
                db.execute("""
                    UPDATE conversation_sessions SET last_activity_at_utc=? WHERE id=?
                """, (now, session_id))
        return index

    def recent_turns(self, session_id: str, limit: int = 3) -> list[dict]:
        """Retrieve N *exchanges*, chronological; never raw sensor tool output."""
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        self.require_session(session_id)
        with closing(self._connect()) as db:
            rows = db.execute("""
                SELECT turn_index, user_text, assistant_text, route, model
                FROM conversation_turns WHERE session_id=?
                ORDER BY turn_index DESC LIMIT ?
            """, (session_id, limit)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def latest_route(self, session_id: str) -> str:
        turns = self.recent_turns(session_id, limit=1)
        return turns[-1]["route"] if turns else "general"

    def latest_live_observation(self, session_id: str) -> dict | None:
        """Find the last actual live measurement, excluding discussion turns."""
        self.require_session(session_id)
        with closing(self._connect()) as db:
            row = db.execute("""
                SELECT assistant_text, created_at_utc, route FROM conversation_turns
                WHERE session_id=? AND route IN ('live_person', 'live_visual')
                ORDER BY turn_index DESC LIMIT 1
            """, (session_id,)).fetchone()
        return dict(row) if row else None

    def recent_chat_report(
        self, *, exclude_session_id: str | None = None,
        max_sessions: int = 2, turns_per_session: int = 4,
    ) -> dict:
        """Bounded read-only transcript excerpts for explicit past-chat questions."""
        if not 1 <= max_sessions <= 5 or not 1 <= turns_per_session <= 10:
            raise ValueError("Invalid conversation retrieval limits")
        with closing(self._connect()) as db:
            sessions = db.execute("""
                SELECT s.id, s.started_at_utc,
                       (SELECT COUNT(*) FROM conversation_turns t
                        WHERE t.session_id=s.id) AS turn_count
                FROM conversation_sessions s
                WHERE (? IS NULL OR s.id != ?)
                  AND EXISTS (SELECT 1 FROM conversation_turns t WHERE t.session_id=s.id)
                ORDER BY s.last_activity_at_utc DESC, s.rowid DESC LIMIT ?
            """, (exclude_session_id, exclude_session_id, max_sessions)).fetchall()
            reports = []
            for session in sessions:
                rows = db.execute("""
                    SELECT turn_index, user_text, assistant_text, route
                    FROM conversation_turns WHERE session_id=?
                    ORDER BY turn_index DESC LIMIT ?
                """, (session["id"], turns_per_session)).fetchall()
                reports.append({
                    "session_id": session["id"],
                    "started_local": _local(session["started_at_utc"]),
                    "total_turns": session["turn_count"],
                    "shown_turns": len(rows),
                    "turns": [
                        {
                            "turn": row["turn_index"],
                            "user": row["user_text"][:280],
                            "assistant": row["assistant_text"][:380],
                            "route": row["route"],
                        }
                        for row in reversed(rows)
                    ],
                })
        return {
            "ok": bool(reports),
            "sessions": reports,
            "limitations": (
                "Only recent dialogue excerpts are shown; earlier turns can exist "
                "outside this bounded report. Do not infer unrecorded conversations."
            ),
        }
