"""Check the compact report against the recorded Week 10 example values."""
import os
import sqlite3
import tempfile
import unittest
import sys
from types import ModuleType
from unittest.mock import patch

# Only needed if running off-Pi without the Week 10 database module.
try:
    import memory_store
except ImportError:
    fake_store = ModuleType("memory_store")
    fake_store.connect = lambda: None
    sys.modules["memory_store"] = fake_store

import memory_tools


class MemoryReportTests(unittest.TestCase):
    def test_positive_scores_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "events.sqlite3")
            db = sqlite3.connect(db_path)
            db.executescript("""
                CREATE TABLE away_sessions(
                    id INTEGER PRIMARY KEY, started_at_utc TEXT, ended_at_utc TEXT);
                CREATE TABLE observations(
                    id INTEGER PRIMARY KEY, away_session_id INTEGER,
                    observed_at_utc TEXT, trigger TEXT, person_detected INTEGER,
                    detection_score REAL);
            """)
            db.execute(
                "INSERT INTO away_sessions VALUES(2, ?, ?)",
                ("2026-10-19T01:37:37+00:00", "2026-10-19T02:03:21+00:00"),
            )
            times = ["01:51:38", "01:51:55", "01:52:20", "01:52:29",
                     "01:52:45", "01:52:54", "01:53:10", "01:53:19",
                     "01:53:27", "01:53:37", "01:53:53"]
            scores = {4: 0.667, 6: 0.862, 7: 0.889, 10: 0.466}
            for event_id, stamp in zip(range(3, 14), times):
                db.execute(
                    "INSERT INTO observations VALUES (?, 2, ?, 'pir_motion', ?, ?)",
                    (event_id, f"2026-10-19T{stamp}+00:00",
                     int(event_id in scores), scores.get(event_id)),
                )
            db.commit()
            db.close()

            def connect():
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                return conn

            with patch("memory_tools.connect", side_effect=connect):
                report = memory_tools.get_latest_away_report()

            self.assertEqual(report["session_id"], 2)
            self.assertEqual(report["total_events"], 11)
            self.assertEqual(report["person_positive_events"], 4)
            self.assertEqual(report["person_negative_events"], 7)
            self.assertEqual(report["max_person_confidence"], 0.889)
            self.assertEqual(report["first_person_detection_local"],
                             "2026-10-19T12:51:55+11:00")
            self.assertEqual(
                [(r["event_id"], r["score"]) for r in report["positive_events"]],
                [(4, 0.667), (6, 0.862), (7, 0.889), (10, 0.466)],
            )
            self.assertNotIn("events", report)


if __name__ == "__main__":
    unittest.main()
