"""Offline regression tests for exact-path filtered memory search."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.week_11.memory_event_index import search_person_events


class MemoryEventIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        db_dir = self.root / "outputs" / "week_10"
        db_dir.mkdir(parents=True)
        self.db_path = db_dir / "events.sqlite3"
        self.obs_root = self.root / "outputs" / "observations"
        self.obs_root.mkdir()
        with sqlite3.connect(self.db_path) as db:
            db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, "
                       "away_session_id INTEGER, observed_at_utc TEXT, trigger TEXT, "
                       "person_detected INTEGER, detection_score REAL, evidence_path TEXT)")
            self._create(db, 7, 0.889, "2026-10-19T01:52:45+00:00")
            self._create(db, 19, 0.513, "2026-10-19T06:29:14+00:00")
            self._create(db, 22, 0.786, "2026-10-19T09:12:27+00:00")
            self._create(db, 23, 0.819, "2026-10-20T08:27:29+00:00")
            self._create(db, 28, 0.358, "2026-10-21T00:28:40+00:00")
            db.execute("INSERT INTO observations VALUES (29,NULL,?,?,0,NULL,NULL)",
                       ("2026-10-21T01:00:00+00:00", "live_observation"))

    def _create(self, db, id, score, timestamp):
        folder = self.obs_root / f"capture_{id}"
        folder.mkdir()
        path = folder / "scene.jpg"
        if id != 19:  # Simulate a missing file.
            path.write_bytes(b"photo")
        db.execute("INSERT INTO observations VALUES (?,NULL,?,?,1,?,?)",
                   (id, timestamp, "live_observation", score, str(path)))

    def test_filter_is_applied_before_limit(self):
        result = search_person_events(self.db_path, limit=1, min_score=0.7)
        self.assertEqual(result["total_positive_events"], 5)
        self.assertEqual(result["total_matching_events"], 3)
        self.assertTrue(result["events_truncated"])
        self.assertEqual([row["event_id"] for row in result["events"]], [23])

    def test_many_ids_return_authoritative_paths(self):
        result = search_person_events(self.db_path, event_ids=[23, 22, 19, 7])
        self.assertEqual([row["event_id"] for row in result["events"]], [23, 22, 19, 7])
        self.assertEqual(result["total_matching_events"], 4)
        self.assertEqual(result["events"][0]["evidence_path"], str(self.obs_root / "capture_23" / "scene.jpg"))
        self.assertTrue(result["events"][0]["image_available"])
        self.assertFalse(result["events"][2]["image_available"])

    def test_no_matching_results(self):
        result = search_person_events(self.db_path, min_score=0.99)
        self.assertEqual(result["events"], [])
        self.assertEqual(result["total_matching_events"], 0)

    def test_missing_database(self):
        result = search_person_events(self.root / "missing.sqlite3")
        self.assertFalse(result["ok"])

    def test_reject_invalid_args(self):
        for kwargs in [
            {"min_score": -0.1}, {"min_score": float("nan")},
            {"min_score": True}, {"event_ids": [23, 23]},
            {"event_ids": []}, {"event_ids": ["23"]},
            {"limit": True}, {"limit": 13},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                search_person_events(self.db_path, **kwargs)

    def test_unapproved_path_not_exposed(self):
        with sqlite3.connect(self.db_path) as db:
            db.execute("INSERT INTO observations VALUES (40,NULL,?,?,1,?,?)",
                       ("2026-10-21T02:00:00+00:00", "live_observation", 0.99, "/etc/passwd"))
        result = search_person_events(self.db_path, limit=1)
        self.assertEqual(result["events"][0]["event_id"], 40)
        self.assertIsNone(result["events"][0]["evidence_path"])
        self.assertFalse(result["events"][0]["image_available"])


if __name__ == "__main__":
    unittest.main()
