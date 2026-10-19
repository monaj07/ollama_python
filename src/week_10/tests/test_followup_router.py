"""Structured routing contract plus an optional on-device semantic evaluation."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from followup_router import FOCUS_TO_INTENT, ROUTING_SCHEMA, resolve_live_followup


class FollowupRouterTests(unittest.TestCase):
    def resolve(self, content, question="Please have another look."):
        client = Mock()
        client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content=content))
        return resolve_live_followup(
            question, client=client, model="test-model",
            history=[{"role": "user", "content": "Check for people in the room."}],
            last_topic="live_person",
        )

    def test_uses_model_decision_for_unlisted_wording(self):
        for focus, intent in FOCUS_TO_INTENT.items():
            with self.subTest(focus=focus):
                self.assertEqual(self.resolve(json.dumps({"focus": focus})), intent)

    def test_invalid_decisions_require_clarification(self):
        for content in ("", None, "not JSON", "[]", "null", '{"intent":"delete"}',
                        '{"intent":"general"}',
                        '{"intent":true}', '{"intent":[]}',
                        '{"focus":"delete"}', '{"focus":[]}', '{"focus":true}',
                        '{"focus":"scene_state","tool":"shell"}',
                        '{"intent":"repeat_observation","tool":"shell"}'):
            with self.subTest(content=content):
                self.assertEqual(self.resolve(content), "clarify")

    def test_supplies_bounded_dialogue_and_no_sensor_tools(self):
        history = [{"role": "user", "content": str(index)} for index in range(20)]
        client = Mock()
        client.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content='{"focus":"scene_state"}'),
        )
        resolve_live_followup("A new measurement, please.", client=client, model="test-model",
                              history=history, last_topic="live_person")
        request = client.chat.call_args.kwargs
        self.assertEqual(request["format"], ROUTING_SCHEMA)
        self.assertEqual(request["tools"], [])
        self.assertFalse(request["think"])
        context = json.loads(request["messages"][-1]["content"])
        self.assertEqual(context["recent_dialogue"], history[-6:])
        self.assertEqual(context["current_request"], "A new measurement, please.")
        self.assertEqual(context["previous_topic"], "live_person")

    def test_routing_failure_does_not_fabricate_a_decision(self):
        client = Mock()
        client.chat.side_effect = RuntimeError("Ollama unavailable")
        with self.assertRaisesRegex(RuntimeError, "Ollama unavailable"):
            resolve_live_followup("Try once more.", client=client, model="test-model",
                                  history=[], last_topic="live_person")


@unittest.skipUnless(os.environ.get("OLLAMA_ROUTER_EVAL") == "1",
                     "Set OLLAMA_ROUTER_EVAL=1 to evaluate the real local model")
class LocalModelEvaluation(unittest.TestCase):
    def setUp(self):
        from ollama import Client
        self.client = Client(host="http://127.0.0.1:11434", timeout=200)
        self.addCleanup(self.client._client.close)

    def test_corrections_override_repeated_sensor_answers(self):
        client = self.client
        stale = (
            "I haven't taken a fresh reading on this turn. The earlier observation at "
            '2026-10-19T17:03:37+11:00 was: "The camera did not detect a person in its '
            'latest frame. The PIR currently reports no movement. That does not prove '
            'the room is empty." That describes the earlier measurement, not the room now.'
        )
        history = [
            {"role": "user", "content": "Do you see anyone in the room?"},
            {"role": "assistant", "content": "The camera did not detect anyone."},
            {"role": "user", "content": "What is three plus nineteen?"},
            {"role": "assistant", "content": stale},
            {"role": "user", "content": "No, I asked why these three plus nineteen."},
            {"role": "assistant", "content": stale},
        ]
        for question, expected in (
            ("No, I asked why these three plus nineteen.", ("change_topic", "clarify")),
            ('I asked, "What\'s 3+19?"', "change_topic"),
            ("You misunderstood; I meant the distance between Earth and the Moon.", "change_topic"),
            ("My question was about multiplication, not who is present.", "change_topic"),
            ("No, explain the earlier detector score.", "discuss_observation"),
        ):
            with self.subTest(question=question):
                decision = resolve_live_followup(
                    question, client=client, model=os.environ.get("OLLAMA_MODEL", "gemma4:e2b"),
                    history=history, last_topic="live_person_discussion",
                )
                allowed = expected if isinstance(expected, tuple) else (expected,)
                self.assertIn(decision, allowed)

    def test_followup_meaning_on_the_real_model(self):
        client = self.client
        history = [
            {"role": "user", "content": "Is anyone in the room?"},
            {"role": "assistant", "content": "The camera detected no people. The PIR reported no movement."},
        ]
        for question, expected in (
            ("Can you double check?", "repeat_observation"),
            ("Please obtain another independent reading.", "repeat_observation"),
            ("Has the situation changed since your last look?", "repeat_observation"),
            ("Really? No one?", "repeat_observation"),
            ("That seems unlikely; are you certain the space is unoccupied?", "repeat_observation"),
            ("Explain how you reached that conclusion.", "discuss_observation"),
            ("Can you double check that two plus three is five?", "change_topic"),
            ("What does PIR mean?", "explain_sensor"),
            ("Could there be somebody you missed?", "repeat_observation"),
            ("How does that sensor detect movement?", "explain_sensor"),
        ):
            with self.subTest(question=question):
                self.assertEqual(resolve_live_followup(
                    question, client=client, model=os.environ.get("OLLAMA_MODEL", "gemma4:e2b"),
                    history=history, last_topic="live_person",
                ), expected)

    def test_logged_sequence_on_the_real_model(self):
        client = self.client
        summary = "The camera did not detect a person. The PIR reports no movement. This does not prove absence."
        history = [
            {"role": "user", "content": "Do you see anyone in the room?"},
            {"role": "assistant", "content": summary},
        ]
        for question in ("Okay, how about now?", "Really? No one?", "How about now?"):
            with self.subTest(question=question):
                self.assertEqual(resolve_live_followup(
                    question, client=client, model=os.environ.get("OLLAMA_MODEL", "gemma4:e2b"),
                    history=history, last_topic="live_person",
                ), "repeat_observation")
                history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": summary},
                ])

    def test_corrected_arithmetic_is_answered_after_a_stuck_conversation(self):
        import tempfile
        from src.week_10.agent import MemoryVoiceAgent
        from src.week_10.conversation_store import ConversationStore
        import memory_store
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.sqlite3'
            store = ConversationStore(path)
            session_id = store.new_session()
            stale = 'The earlier camera did not detect a person. The PIR reported no movement.'
            for question in ('Do you see anyone in the room?',
                             'No, I asked why these three plus nineteen.',
                             'I asked, "What\'s 3+19?"'):
                store.append_turn(session_id, question, stale, 'live_person_discussion', 'test-model')
            tools = Mock()
            tools.execute.side_effect = AssertionError('Arithmetic must not invoke sensors')
            with patch.object(memory_store, 'DB_PATH', path):
                agent = MemoryVoiceAgent(client=self.client, tools=tools,
                    model=os.environ.get('OLLAMA_MODEL', 'gemma4:e2b'),
                    conversation_store=store, resume_last=True,
                    trace_path=Path(directory) / 'trace.jsonl')
                with patch.object(agent, '_discuss_live_observation',
                                  side_effect=AssertionError('Correction must not replay sensor evidence')):
                    # This transcript is ambiguous. Either solve the identifiable
                    # addition or request clarification; do not require a guess.
                    answer = agent.ask('No, I asked why these three plus nineteen.')
                    self.assertIn(agent.last_topic, ('general', 'live_person_clarify'))
                    if not re.search(r'\b22\b|twenty[- ]two', answer.lower()):
                        self.assertRegex(answer.lower(), r'\?|\b(?:clarify|rephrase|restate)\b')
                    self.assertNotIn(stale, answer)

                    # Once the task is stated clearly, clarification alone is
                    # insufficient: the arithmetic must actually be answered.
                    answer = agent.ask('I asked, "What\'s 3+19?"')
                    self.assertRegex(answer.lower(), r'\b22\b|twenty[- ]two')
                    self.assertEqual(agent.last_topic, 'general')
            tools.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
