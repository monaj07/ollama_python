"""SQLite memory regression tests; no GPIO or camera hardware required."""
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/week_10"))
import memory_store


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        db_path = patch.object(memory_store, "DB_PATH", Path(temp.name) / "events.sqlite3")
        db_path.start()
        self.addCleanup(db_path.stop)

    def test_supplied_timestamp_is_normalized_with_either_session_selection(self):
        session_id = memory_store.arm_away_mode()
        for selected_session in (session_id, None):
            with self.subTest(away_session_id=selected_session):
                event_id = memory_store.add_observation(
                    "pir_motion",
                    away_session_id=selected_session,
                    observed_at_utc="2026-10-19T12:40:23.203627+11:00",
                )
                event = memory_store.recent_observations(1)[0]
                self.assertEqual(event["id"], event_id)
                self.assertEqual(event["away_session_id"], session_id)
                self.assertEqual(event["observed_at_utc"], "2026-10-19T01:40:23.203627+00:00")

    def test_missing_timestamp_uses_current_utc_with_or_without_session(self):
        session_id = memory_store.arm_away_mode()
        for selected_session in (session_id, None):
            with self.subTest(away_session_id=selected_session):
                before = datetime.now(timezone.utc)
                memory_store.add_observation("pir_motion", away_session_id=selected_session)
                after = datetime.now(timezone.utc)
                event = memory_store.recent_observations(1)[0]
                timestamp = datetime.fromisoformat(event["observed_at_utc"])
                self.assertEqual(timestamp.utcoffset(), timezone.utc.utcoffset(timestamp))
                self.assertLessEqual(before, timestamp)
                self.assertLessEqual(timestamp, after)
                self.assertEqual(event["away_session_id"], session_id)

        memory_store.disarm_away_mode()
        memory_store.add_observation("pir_motion")
        self.assertIsNone(memory_store.recent_observations(1)[0]["away_session_id"])

    def test_naive_timestamp_is_rejected_with_either_session_selection(self):
        session_id = memory_store.arm_away_mode()
        for selected_session in (session_id, None):
            with self.subTest(away_session_id=selected_session):
                with self.assertRaisesRegex(ValueError, "must include a timezone"):
                    memory_store.add_observation(
                        "pir_motion",
                        away_session_id=selected_session,
                        observed_at_utc="2026-10-19T01:40:23.203627",
                    )
        self.assertEqual(memory_store.recent_observations(), [])


if __name__ == "__main__":
    unittest.main()
