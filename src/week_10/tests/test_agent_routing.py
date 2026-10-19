"""No sensor or Ollama server required; run in Week 09 venv on the Pi."""
from __future__ import annotations

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

from src.week_10 import agent as a
from src.week_10 import recent_memory as rm
from src.week_10.conversation_store import ConversationStore


class FakeClient:
    def __init__(self):
        self.calls = []
    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content="Test answer", tool_calls=[]))


class FakeTools:
    def close(self):
        pass


class RouteTests(unittest.TestCase):
    def test_general_never_requests_memory(self):
        for q in ["What is 2+17?", "Who are you?", "How old are you?"]:
            with self.subTest(q=q):
                self.assertEqual(a.route_question(q, "memory_session"), "general")

    def test_recency(self):
        self.assertEqual(a.route_question("What did you see most recently?"), "memory_recent")
        self.assertEqual(a.route_question("What was your last observation?"), "memory_recent")

    def test_historical_and_followup(self):
        self.assertEqual(a.route_question("Was there anyone in the room?"), "memory_session")
        self.assertEqual(a.route_question("How confident were those detections?", "memory_session"), "memory_session")
        self.assertEqual(a.route_question("Was there anyone in the room?", "memory_recent"), "memory_session")
        self.assertEqual(a.route_question("At what time was the first person detected?"), "memory_session")
        self.assertEqual(a.route_question("How many times did the camera detect a person?"), "memory_session")

    def test_conversation_history_route(self):
        self.assertEqual(a.route_question("What did we talk about last time?"), "chat_history")
        self.assertEqual(a.route_question("What was our previous conversation?"), "chat_history")
        self.assertEqual(a.route_question("What is 2 + 17?", "chat_history"), "general")

    def test_live(self):
        for q in ["Is anyone in the room right now?", "What can you see?", "Who is here currently?"]:
            # 'Who is here currently?' has explicit currently, so LIVE.
            with self.subTest(q=q):
                self.assertEqual(a.route_question(q, "memory_session"), "live")


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        store = ConversationStore(Path(self.tempdir.name) / "test.sqlite3")
        self.client = FakeClient()
        self.agent = a.MemoryVoiceAgent(client=self.client, tools=FakeTools(),
                                        conversation_store=store)

    def test_general_cannot_read_db_or_call_tools(self):
        with patch.object(a, "get_observation_history_report", side_effect=AssertionError("SQL should not be read")), \
             patch.object(a, "get_recent_observations_report", side_effect=AssertionError("SQL should not be read")):
            answer = self.agent.ask("What is 2+17?")
        self.assertEqual(answer, "Test answer")
        self.assertEqual(self.client.calls[-1]["tools"], [])

    def test_recent_is_stored_not_live(self):
        data = {"ok": True, "observations": [{"event_id": 13, "person_detected": False}]}
        with patch.object(a, "get_recent_observations_report", return_value=data), \
             patch.object(a.VoiceAgent, "ask", side_effect=AssertionError("Live sensor invoked")):
            self.agent.ask("What did you see most recently?")
        self.assertIn('"event_id":13', self.client.calls[-1]["messages"][-1]["content"])
        self.assertEqual(self.client.calls[-1]["tools"], [])

    def test_followup_uses_away_session(self):
        with patch.object(a, "get_recent_observations_report", return_value={"ok": True, "observations": []}):
            self.agent.ask("What did you see most recently?")
        with patch.object(a, "get_observation_history_report", return_value={"ok": True, "session_id": 2, "person_positive_events": 4}):
            self.agent.ask("Was there anyone in the room?")
        self.assertIn('"person_positive_events":4', self.client.calls[-1]["messages"][-1]["content"])
        self.assertEqual(len(self.agent.history), 4)

    def test_chat_history_routes_to_sqlite_not_camera(self):
        with patch.object(a.VoiceAgent, "ask", side_effect=AssertionError("Camera invoked")):
            self.agent.ask("What did we discuss last time?")
        # No previous session: a truthful fallback, with no Ollama call.
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.agent.last_topic, "chat_history")

    def test_non_person_live_request_delegates_to_week09(self):
        with patch.object(a.VoiceAgent, "ask", return_value="Camera checked the scene") as live, \
             patch.object(a, "get_observation_history_report", side_effect=AssertionError("SQL invoked")):
            got = self.agent.ask("What can you see?")
        self.assertEqual(got, "Camera checked the scene")
        live.assert_called_once_with("What can you see?")


class RecentReportTests(unittest.TestCase):
    def test_negative_latest_observation_remains_negative(self):
        rows = [
            {"id": 13, "away_session_id": 2, "observed_at_utc": "2026-10-19T01:53:53+00:00",
             "trigger": "pir_motion", "person_detected": 0, "detection_score": None,
             "details_json": '{"person_count": 0, "camera_ok": true}'},
            {"id": 10, "away_session_id": 2, "observed_at_utc": "2026-10-19T01:53:19+00:00",
             "trigger": "pir_motion", "person_detected": 1, "detection_score": 0.466,
             "details_json": '{"person_count": 1, "camera_ok": true}'},
        ]
        with patch.object(rm, "recent_observations", return_value=rows):
            report = rm.get_recent_observations_report(5)
        self.assertFalse(report["observations"][0]["person_detected"])
        self.assertEqual(report["observations"][0]["event_id"], 13)
        self.assertEqual(report["observations"][1]["person_score"], 0.466)
        self.assertEqual(report["observations"][0]["time_local"], "2026-10-19T12:53:53+11:00")


class VoiceAdapterTests(unittest.TestCase):
    def test_cli_resume_flag_preserves_week09_audio_flags(self):
        from src.week_10 import voice_assistant as adapter
        with patch.object(adapter.voice09, "main") as run_voice, \
             patch.object(sys, "argv", ["voice_assistant.py", "--resume-last",
                                        "--mic-device", "plughw:0,0", "--duration", "5"]):
            adapter._agent = None
            adapter.main()
            run_voice.assert_called_once()
            self.assertEqual(sys.argv[1:], ["--mic-device", "plughw:0,0", "--duration", "5"])
            self.assertTrue(adapter._resume_last)
        adapter._resume_last = False
        adapter._agent = None

    def test_push_to_talk_callback_reuses_agent_between_questions(self):
        from src.week_10 import voice_assistant as adapter

        instances = []

        class FakeMemoryAgent:
            def __init__(self, model, **kwargs):
                self.model = model
                self.session_id = "stub-session"
                self.questions = []
                self.tools = FakeTools()
                instances.append(self)

            def ask(self, question):
                self.questions.append(question)
                return "spoken answer"

        with patch.object(adapter, "MemoryVoiceAgent", FakeMemoryAgent):
            adapter._agent = None
            self.assertEqual(adapter.ask_agent("What did you see?", "gemma4:e2b"), "spoken answer")
            self.assertEqual(adapter.ask_agent("Was anyone detected?", "gemma4:e2b"), "spoken answer")
            self.assertEqual(len(instances), 1)
            self.assertEqual(instances[0].questions, ["What did you see?", "Was anyone detected?"])
        adapter._agent = None


if __name__ == "__main__":
    unittest.main()
