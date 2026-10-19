"""Week 10 refactor stage 1: recording isolation and structured past evidence."""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import live_memory
import memory_tools


class FakeTools:
    def __init__(self, *, error=None):
        self.error = error
        self.calls = []

    def execute(self, name, args):
        self.calls.append(name)
        if name == "detect_motion":
            return {"ok": True, "motion_signal_active": False}
        if self.error is not None:
            raise self.error
        return {
            "ok": True, "observed_at": "2026-10-19T05:03:52+00:00",
            "counts": {"person": 1},
            "detections": [{"label": "person", "confidence": 0.536}],
            "source": "/tmp/camera.jpg",
        }

    def close(self):
        pass


class RecordingIsolationTests(unittest.TestCase):
    def test_success_stores_one_event_and_preserves_tool_return(self):
        tools = FakeTools()
        wrapper = live_memory.RecordingTools(tools, "session-a")
        wrapper.begin_turn()
        wrapper.execute("detect_motion", {})
        with patch.object(live_memory, "add_observation", return_value=17) as save:
            result = wrapper.execute("observe_scene", {"targets": ["person"]})
        self.assertEqual(result["counts"]["person"], 1)
        self.assertEqual(wrapper.last_recording, {"memory_saved": True, "event_id": 17})
        self.assertEqual(save.call_count, 1)
        self.assertEqual(save.call_args.kwargs["details"]["conversation_session_id"], "session-a")
        self.assertEqual(save.call_args.kwargs["details"]["pir_signal_active"], False)

    def test_save_failure_does_not_make_good_camera_result_fail(self):
        wrapper = live_memory.RecordingTools(FakeTools(), "session-a")
        with patch.object(live_memory, "add_observation", side_effect=sqlite3.OperationalError("locked")):
            result = wrapper.execute("observe_scene", {})
        self.assertTrue(result["ok"])
        self.assertEqual(result["counts"]["person"], 1)
        self.assertFalse(wrapper.last_recording["memory_saved"])
        self.assertIn("locked", wrapper.last_recording["error"])

    def test_camera_error_is_not_masked_by_save_error(self):
        original = RuntimeError("camera unplugged")
        wrapper = live_memory.RecordingTools(FakeTools(error=original), "session-a")
        with patch.object(live_memory, "add_observation", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "camera unplugged"):
                wrapper.execute("observe_scene", {})
        self.assertFalse(wrapper.last_recording["memory_saved"])

    def test_sensor_without_camera_does_not_write(self):
        wrapper = live_memory.RecordingTools(FakeTools(), "session-a")
        with patch.object(live_memory, "add_observation") as save:
            wrapper.execute("detect_motion", {})
        save.assert_not_called()

    def test_recorded_answer_does_not_claim_live_measurement(self):
        record = {
            "time_local": "2026-10-19T16:03:52+11:00",
            "camera_ok": True, "person_count": 1,
            "detection_score": 0.536, "pir_signal_active": False,
        }
        text = live_memory.render_recorded_observation(record)
        self.assertIn("one earlier observation", text)
        self.assertIn("0.536", text)
        self.assertIn("no movement", text)
        self.assertIn("haven't taken a new reading", text)

    def test_camera_failure_remains_unknown_in_recollection(self):
        record = {"time_local": "2026-10-19T16:03:52+11:00",
                  "camera_ok": False, "person_count": None,
                  "detection_score": None, "pir_signal_active": None}
        text = live_memory.render_recorded_observation(record)
        self.assertIn("unavailable", text)
        self.assertNotIn("room is empty", text)


class StructuredRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "events.sqlite3"
        with sqlite3.connect(self.path) as db:
            db.execute("""CREATE TABLE observations (
                id INTEGER PRIMARY KEY, observed_at_utc TEXT NOT NULL,
                trigger TEXT NOT NULL, detection_score REAL,
                evidence_path TEXT, details_json TEXT NOT NULL
            )""")
            for sid, ts, n in (
                ("session-a", "2026-10-19T05:03:52+00:00", 1),
                ("session-b", "2026-10-19T05:04:52+00:00", 0),
            ):
                db.execute("""INSERT INTO observations
                    (observed_at_utc, trigger, detection_score, evidence_path, details_json)
                    VALUES (?, 'live_observation', ?, NULL, ?)""",
                    (ts, 0.536 if n else None, json.dumps({
                        "conversation_session_id": sid,
                        "camera_ok": True, "person_count": n,
                        "pir_signal_active": False,
                    })),
                )

    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def test_retrieves_structured_evidence_for_correct_session(self):
        with patch.object(memory_tools, "connect", self._connect):
            a = memory_tools.get_latest_live_observation("session-a")
            b = memory_tools.get_latest_live_observation("session-b")
            absent = memory_tools.get_latest_live_observation("absent")
        self.assertEqual(a["person_count"], 1)
        self.assertEqual(a["detection_score"], 0.536)
        self.assertEqual(a["pir_signal_active"], False)
        self.assertEqual(b["person_count"], 0)
        self.assertIsNone(b["detection_score"])
        self.assertIsNone(absent)


if __name__ == "__main__":
    unittest.main()
