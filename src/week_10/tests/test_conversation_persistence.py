"""Week 10 persistent dialogue; uses isolated SQLite and a fake Ollama client."""
from __future__ import annotations

from contextlib import closing
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src/week_09", ROOT / "src/week_10"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.week_10.conversation_store import ConversationStore
from src.week_10 import agent as a


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content="Mock answer", tool_calls=[]))


class FakeTools:
    def close(self):
        pass


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "events.sqlite3"
        self.store = ConversationStore(self.db_path)

    def agent(self, **kwargs):
        return a.MemoryVoiceAgent(client=FakeClient(), tools=FakeTools(),
                                  conversation_store=self.store, **kwargs)

    def test_persists_complete_dialogue_after_restart(self):
        first = self.agent()
        first.ask("What is 2 + 17?")
        first.ask("Who are you?")
        self.assertEqual(len(self.store.recent_turns(first.session_id)), 2)
        fresh = ConversationStore(self.db_path)
        second = a.MemoryVoiceAgent(client=FakeClient(), tools=FakeTools(),
                                    conversation_store=fresh, resume_last=True)
        self.assertEqual(first.session_id, second.session_id)
        self.assertEqual(len(second.history), 4)
        self.assertEqual(second.history[0], {"role": "user", "content": "What is 2 + 17?"})
        self.assertEqual(second.history[-1], {"role": "assistant", "content": "Mock answer"})
        second.ask("How many times did I ask you something?")
        self.assertEqual(len(fresh.recent_turns(first.session_id)), 3)

    def test_new_session_is_separate_but_old_dialogue_retrievable(self):
        first = self.agent()
        first.ask("Talk about the garden")
        second = self.agent()
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(second.history, [])
        report = self.store.recent_chat_report(exclude_session_id=second.session_id)
        self.assertTrue(report["ok"])
        self.assertEqual(report["sessions"][0]["turns"][0]["user"], "Talk about the garden")
        second.ask("What did we talk about last time?")
        self.assertEqual(second.last_topic, "chat_history")
        self.assertIn("Talk about the garden", second.client.calls[-1]["messages"][-1]["content"])
        self.assertEqual(self.store.recent_turns(second.session_id)[0]["route"], "chat_history")

    def test_recent_turns_are_bounded_not_full_history(self):
        first = self.agent()
        for i in range(5):
            first.ask(f"What is {i} + 1?")
        second = self.agent(conversation_session_id=first.session_id)
        self.assertEqual(len(second.history), 6)
        self.assertEqual(second.history[0]["content"], "What is 2 + 1?")
        self.assertEqual(len(self.store.recent_turns(first.session_id, 10)), 5)
        self.assertEqual(second.client.calls, [])

    def test_last_topic_restored_for_followups(self):
        first = self.agent()
        with patch.object(a, "get_observation_history_report", return_value={
            "ok": True, "session_id": 2, "person_positive_events": 4,
        }):
            first.ask("Was there anyone in the room?")
        resumed = self.agent(conversation_session_id=first.session_id)
        self.assertEqual(resumed.last_topic, "memory_session")
        with patch.object(a, "get_observation_history_report", return_value={
            "ok": True, "session_id": 2, "person_positive_events": 4,
        }) as report:
            resumed.ask("How confident were those detections?")
        report.assert_called_once()
        self.assertEqual(resumed.last_topic, "memory_session")

    def test_general_questions_do_not_read_observations(self):
        instance = self.agent()
        with patch.object(a, "get_recent_observations_report", side_effect=AssertionError("unexpected")), \
             patch.object(a, "get_observation_history_report", side_effect=AssertionError("unexpected")):
            instance.ask("What is 2 + 17?")
        self.assertEqual(len(instance.client.calls), 1)
        self.assertEqual(instance.client.calls[0]["tools"], [])
        turns = self.store.recent_turns(instance.session_id)
        self.assertEqual(turns[0]["user_text"], "What is 2 + 17?")
        self.assertEqual(turns[0]["route"], "general")

    def test_no_partial_history_if_model_fails(self):
        client = FakeClient()
        instance = a.MemoryVoiceAgent(client=client, tools=FakeTools(),
                                      conversation_store=self.store)
        with patch.object(client, "chat", side_effect=RuntimeError("inference failed")):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                instance.ask("Hello")
        self.assertEqual(self.store.recent_turns(instance.session_id), [])
        self.assertEqual(instance.history, [])

    def test_bad_session_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            self.agent(conversation_session_id="unknown")

    def test_existing_sensor_tables_untouched(self):
        import sqlite3
        with closing(sqlite3.connect(self.db_path)) as db:
            with db:
                db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, trigger TEXT)")
                db.execute("INSERT INTO observations VALUES (1, 'pir_motion')")
        first = self.agent()
        first.ask("Who are you?")
        with closing(sqlite3.connect(self.db_path)) as db:
            self.assertEqual(db.execute("SELECT trigger FROM observations").fetchone()[0],
                             "pir_motion")
            self.assertEqual(db.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0], 1)

    def test_chat_report_is_bounded(self):
        first = self.agent()
        for i in range(7):
            first.ask("hello " + str(i))
        report = self.store.recent_chat_report(turns_per_session=3)
        self.assertTrue(report["ok"])
        self.assertEqual(report["sessions"][0]["total_turns"], 7)
        self.assertEqual(len(report["sessions"][0]["turns"]), 3)


if __name__ == "__main__":
    unittest.main()
