"""Regression tests: no memory access or extra model call for unrelated queries."""
import unittest
import sys
from types import SimpleNamespace, ModuleType

# Permit routing tests without Ollama or Pi-specific dependencies.
try:
    import ollama
except ImportError:
    fake_ollama = ModuleType("ollama")
    fake_ollama.Client = object
    sys.modules["ollama"] = fake_ollama
try:
    import memory_store
except ImportError:
    fake_store = ModuleType("memory_store")
    fake_store.connect = lambda: None
    sys.modules["memory_store"] = fake_store
from unittest.mock import patch

from agent import ConversationState, ask, needs_memory


REPORT = {
    "ok": True,
    "session_id": 2,
    "total_events": 11,
    "person_positive_events": 4,
    "person_negative_events": 7,
    "positive_events": [
        {"event_id": i, "score": score}
        for i, score in [(4, 0.667), (6, 0.862), (7, 0.889), (10, 0.466)]
    ],
}


class FakeClient:
    def __init__(self, reply="Test reply"):
        self.reply = reply
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=self.reply))


class AgentRoutingTests(unittest.TestCase):
    def test_relevant_questions(self):
        state = ConversationState()
        for question in (
            "Was anyone near my desk while I was away?",
            "How many times did the camera detect a person?",
            "At what time was the first person detected?",
            "How confident were those detections?",
        ):
            self.assertTrue(needs_memory(question, state), question)

    def test_unrelated_questions(self):
        state = ConversationState(last_topic="memory")
        for question in ("Who are you?", "How old are you?", "What is 2+17?"):
            self.assertFalse(needs_memory(question, state), question)

    @patch("agent.get_latest_away_report")
    def test_arithmetic_never_calls_memory(self, get_report):
        client = FakeClient("19")
        answer = ask("What is 2+17?", client)
        self.assertEqual(answer, "19")
        get_report.assert_not_called()
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["tools"], [])

    @patch("agent.get_latest_away_report", return_value=REPORT)
    def test_memory_then_followup_then_general(self, get_report):
        client = FakeClient()
        state = ConversationState()
        ask("Was anyone near my desk while I was away?", client, state)
        ask("Were they the same person?", client, state)
        ask("What is 2+17?", client, state)
        self.assertEqual(get_report.call_count, 2)
        self.assertEqual(len(client.calls), 3)
        self.assertIn('"person_positive_events":4', client.calls[0]["messages"][-1]["content"])
        self.assertEqual(client.calls[2]["messages"][-1]["content"], "What is 2+17?")
        self.assertEqual(len(state.history), 6)
        self.assertFalse(any("READ-ONLY SQLITE EVIDENCE" in m["content"] for m in state.history))

    @patch("agent.get_latest_away_report", return_value={"ok": False})
    def test_no_sessions_does_not_hallucinate(self, get_report):
        client = FakeClient()
        answer = ask("Was anyone near my desk while I was away?", client)
        self.assertIn("no recorded Away Mode session", answer)
        self.assertEqual(len(client.calls), 0)


if __name__ == "__main__":
    unittest.main()
