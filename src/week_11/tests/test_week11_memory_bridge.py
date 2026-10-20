"""Unit tests with fake hardware and fake retrieval: no camera, GPIO or SQLite."""
import unittest
from unittest.mock import patch

from src.week_11 import physical_worker as worker


class FakeRecording:
    def __init__(self):
        self.conversation_session_id = "unbound"
        self.calls = []
        self.begin_count = 0
        self.last_recording = {"memory_saved": True, "event_id": 42}

    def begin_turn(self):
        self.begin_count += 1

    def execute(self, tool, args):
        self.calls.append((tool, args))
        if tool == "detect_motion":
            return {"ok": True, "motion_signal_active": False}
        if tool == "observe_scene":
            return {"ok": True, "counts": {"person": 1}}
        raise AssertionError(f"Unexpected hardware access: {tool}")


class Week11BridgeTests(unittest.TestCase):
    def setUp(self):
        self.recording = FakeRecording()

    def test_bind_turn_is_non_destructive(self):
        reply = worker.dispatch(self.recording, "begin_turn", {"session_id": "abc-123"})
        self.assertTrue(reply["ok"])
        self.assertEqual(self.recording.conversation_session_id, "week11-pi:abc-123")
        self.assertEqual(self.recording.begin_count, 1)
        self.assertEqual(self.recording.calls, [])

    def test_camera_response_includes_persistence_status(self):
        reply = worker.dispatch(self.recording, "observe_scene", {"targets": ["person"]})
        self.assertTrue(reply["ok"])
        self.assertEqual(reply["recording"], {"memory_saved": True, "event_id": 42})
        self.assertEqual(self.recording.begin_count, 1)
        self.assertEqual(self.recording.calls, [("observe_scene", {"targets": ["person"]})])

    def test_recent_memory_is_read_only(self):
        with patch.object(worker, "get_recent_observations_report", return_value={
            "ok": True, "observations": [{"event_id": 3}],
        }) as fetch:
            reply = worker.dispatch(self.recording, "memory_recent", {"limit": 3})
        fetch.assert_called_once_with(limit=3)
        self.assertEqual(reply["observations"][0]["event_id"], 3)
        self.assertEqual(self.recording.calls, [])

    def test_invalid_memory_limits_are_rejected(self):
        for value in (0, 11, True, "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker.dispatch(self.recording, "memory_recent", {"limit": value})
        self.assertEqual(self.recording.calls, [])

    def test_memory_history_never_calls_hardware(self):
        with patch.object(worker, "get_observation_history_report", return_value={
            "ok": True, "scope": "all_saved_observations",
        }):
            result = worker.dispatch(self.recording, "memory_history", {})
        self.assertEqual(result["scope"], "all_saved_observations")
        self.assertEqual(self.recording.calls, [])

    def test_unknown_commands_do_not_run_hardware(self):
        self.assertFalse(worker.dispatch(self.recording, "bash", {"command": "echo x"})["ok"])
        self.assertEqual(self.recording.calls, [])


if __name__ == "__main__":
    unittest.main()
