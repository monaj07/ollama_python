"""Regression checks for consolidated read-only Pi memory; no real GPIO/camera."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.week_11 import physical_worker as worker


class NoHardware:
    def __init__(self):
        self.calls = []

    def execute(self, name, args):
        self.calls.append((name, args))
        raise AssertionError("Memory queries must not activate physical tools")


class QueryMemoryTests(unittest.TestCase):
    def setUp(self):
        self.hardware = NoHardware()

    def query(self, **params):
        return worker.dispatch(self.hardware, "query_memory", params)

    def test_recent_defaults_to_one_and_does_not_use_hardware(self):
        with patch.object(worker, "get_recent_observations_report", return_value={
            "ok": True, "observations": [{"event_id": 29}],
        }) as fetch:
            got = self.query(operation="recent")
        fetch.assert_called_once_with(limit=1)
        self.assertEqual(got["observations"][0]["event_id"], 29)
        self.assertFalse(self.hardware.calls)

    def test_recent_accepts_bounded_limit(self):
        with patch.object(worker, "get_recent_observations_report", return_value={"ok": True}) as fetch:
            self.query(operation="recent", limit=10)
        fetch.assert_called_once_with(limit=10)

    def test_summary_is_compact_but_preserves_counts(self):
        report = {"ok": True, "total_events": 29, "person_positive_events": 12,
                  "person_negative_events": 15, "events_without_person_result": 2,
                  "pir_motion_events": 11, "positive_events": [{"event_id": 27}],
                  "latest_away_session": {"session_id": 2}, "scope": "all_saved_observations"}
        with patch.object(worker, "get_observation_history_report", return_value=report):
            got = self.query(operation="summary")
        self.assertEqual(got["total_events"], 29)
        self.assertEqual(got["person_positive_events"], 12)
        self.assertNotIn("positive_events", got)
        self.assertNotIn("latest_away_session", got)
        self.assertFalse(self.hardware.calls)

    def test_away_uses_existing_report(self):
        with patch.object(worker, "get_latest_away_report", return_value={
            "ok": True, "session_id": 2, "person_positive_events": 4,
        }) as fetch:
            got = self.query(operation="away")
        fetch.assert_called_once_with()
        self.assertEqual(got["person_positive_events"], 4)
        self.assertFalse(self.hardware.calls)

    def test_events_return_scores_and_are_explicitly_truncated(self):
        report = {"ok": True, "person_positive_events": 12,
                  "positive_events": [
                      {"event_id": i, "score": i / 100, "time_local": "2026-10-20T20:00:00+11:00"}
                      for i in range(12, 0, -1)
                  ]}
        with patch.object(worker, "get_observation_history_report", return_value=report):
            got = self.query(operation="events", limit=3)
        self.assertEqual([r["event_id"] for r in got["events"]], [12, 11, 10])
        self.assertEqual(got["total_positive_events"], 12)
        self.assertTrue(got["events_truncated"])
        self.assertFalse(self.hardware.calls)

    def test_events_default_returns_all_twelve(self):
        report = {"ok": True, "person_positive_events": 12,
                  "positive_events": [{"event_id": i} for i in range(12)]}
        with patch.object(worker, "get_observation_history_report", return_value=report):
            got = self.query(operation="events")
        self.assertEqual(got["returned_events"], 12)
        self.assertFalse(got["events_truncated"])

    def test_exact_event_uses_read_only_parameterized_query(self):
        with tempfile.TemporaryDirectory() as td:
            dbpath = Path(td) / "events.sqlite3"
            image_path = Path(td) / "scene.jpg"
            image_path.write_bytes(b"test image")
            with sqlite3.connect(dbpath) as db:
                db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, "
                           "away_session_id INTEGER, observed_at_utc TEXT, trigger TEXT, "
                           "person_detected INTEGER, detection_score REAL, evidence_path TEXT, details_json TEXT)")
                db.execute("INSERT INTO observations VALUES (10,2,?,?,?,?,?,?)", (
                    "2026-10-19T01:53:19+00:00", "pir_motion", 1, 0.466,
                    str(image_path), json.dumps({"person_count": 1, "camera_ok": True,
                                                      "conversation_session_id": "week11-pi:test"})))
            with patch.object(worker, "DB_PATH", dbpath):
                got = self.query(operation="event", event_id=10)
                missing = self.query(operation="event", event_id=999)
                self.assertIs(got["image_available"], True)
                image_path.unlink()
                deleted = self.query(operation="event", event_id=10)
                self.assertIs(deleted["image_available"], False)
                for empty_path in (None, ""):
                    with self.subTest(evidence_path=empty_path):
                        with sqlite3.connect(dbpath) as db:
                            db.execute("UPDATE observations SET evidence_path = ? WHERE id = 10",
                                       (empty_path,))
                        no_image = self.query(operation="event", event_id=10)
                        self.assertIs(no_image["image_available"], False)
            self.assertTrue(got["ok"])
            self.assertEqual(got["event_id"], 10)
            self.assertEqual(got["away_session_id"], 2)
            self.assertEqual(got["person_score"], 0.466)
            self.assertEqual(got["time_local"], "2026-10-19T12:53:19+11:00")
            self.assertEqual(got["conversation_session_id"], "week11-pi:test")
            self.assertEqual(got["evidence_path"], str(image_path))
            self.assertFalse(missing["ok"])
            self.assertFalse(self.hardware.calls)

    def test_invalid_operations_and_arguments_are_rejected(self):
        invalid = [
            {}, {"operation": "delete"}, {"operation": "recent", "limit": 11},
            {"operation": "recent", "limit": True},
            {"operation": "recent", "limit": "5"},
            {"operation": "events", "limit": 13},
            {"operation": "events", "event_id": 10},
            {"operation": "summary", "limit": 1},
            {"operation": "away", "event_id": 10},
            {"operation": "event"},
            {"operation": "event", "event_id": True},
            {"operation": "event", "event_id": -1},
            {"operation": "event", "event_id": "1"},
            {"operation": "event", "event_id": 1, "limit": 1},
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                worker.dispatch(self.hardware, "query_memory", args)
        self.assertFalse(self.hardware.calls)


if __name__ == "__main__":
    unittest.main()
