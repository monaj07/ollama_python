"""Regression tests for host-enforced Week 10 live occupancy sensing.

No Pi hardware or Ollama needed. In the Pi environment run with unittest discover.
"""
from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / 'src/week_09', ROOT / 'src/week_10'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.week_10 import agent as a
from src.week_10.conversation_store import ConversationStore
import memory_store
from followup_router import FOCUS_TO_INTENT


class FakeClient:
    def __init__(self):
        self.calls = []
        self.intent = 'repeat_observation'

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        focus = next((key for key, value in FOCUS_TO_INTENT.items() if value == self.intent), self.intent)
        content = json.dumps({'focus': focus}) if kwargs.get('format') else 'Mock explanation.'
        return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))


class FakePhysicalTools:
    def __init__(self, *, pir=None, camera=None):
        self.pir = pir if pir is not None else {"ok": True, "motion_signal_active": False}
        self.camera = camera if camera is not None else {
            "ok": True, "counts": {"person": 1},
            "detections": [{"label": "person", "confidence": 0.81}],
        }
        self.calls = []

    def execute(self, name, args):
        self.calls.append((name, args))
        return self.pir if name == 'detect_motion' else self.camera

    def close(self):
        pass


class LiveSensorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tools = FakePhysicalTools()
        self.client = FakeClient()
        self.store = ConversationStore(Path(self.tmp.name) / 'events.sqlite3')
        db_patch = patch.object(memory_store, 'DB_PATH', self.store.db_path)
        db_patch.start()
        self.addCleanup(db_patch.stop)
        self.agent = a.MemoryVoiceAgent(
            client=self.client,
            tools=self.tools,
            conversation_store=self.store,
        )
        # Stub Week 09 has no trace path; trace persistence isn't under test.
        self.trace_patch = patch.object(self.agent, 'trace', create=True)
        self.trace_patch.start()
        self.addCleanup(self.trace_patch.stop)

    def test_actual_user_questions_both_route_live(self):
        for q in [
            'Do you see anyone in their room right now?',
            'Do you see anyone in the room?',
            'Can you see anyone in the room?',
            'Does the camera see anyone?',
            'Is anyone in the room right now?',
            'Are there any people in the room?',
            'Who is here currently?',
        ]:
            with self.subTest(question=q):
                self.assertEqual(a.route_question(q, 'general'), 'live')
                self.assertTrue(a.is_live_person_question(q))

    def test_live_person_followup_refreshes_both_sensors(self):
        first = self.agent.ask('Do you see anyone in their room?')
        second = self.agent.ask('How about now?')
        self.assertIn('camera reported 1 person', first.lower())
        self.assertIn('camera reported 1 person', second.lower())
        self.assertEqual(self.tools.calls, [
            ('detect_motion', {}),
            ('observe_scene', {'targets': ['person'], 'conf': 0.2}),
            ('detect_motion', {}),
            ('observe_scene', {'targets': ['person'], 'conf': 0.2}),
        ])
        self.assertEqual(len(self.client.calls), 1)
        self.assertIn('format', self.client.calls[0])
        turns = self.store.recent_turns(self.agent.session_id, 10)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[-1]['user_text'], 'How about now?')
        self.assertEqual(turns[-1]['route'], 'live_person')

    def test_double_check_refreshes_the_actual_spoken_person_query(self):
        self.tools.camera = {'ok': True, 'counts': {'person': 0}, 'detections': []}
        with patch.object(a, 'get_observation_history_report', side_effect=AssertionError('Historical report read')), \
             patch.object(a, 'get_recent_observations_report', side_effect=AssertionError('Historical report read')):
            first = self.agent.ask('Is there anyone who drove right now?')
            self.tools.camera = {
                'ok': True, 'counts': {'person': 1},
                'detections': [{'label': 'person', 'confidence': 0.81}],
            }
            second = self.agent.ask('Can you double check?')
        self.assertIn('did not detect a person', first)
        self.assertIn('camera reported 1 person', second.lower())
        self.assertNotIn('driving', second)
        self.assertEqual([name for name, _ in self.tools.calls],
                         ['detect_motion', 'observe_scene'] * 2)
        self.assertEqual(len(self.client.calls), 1)
        self.assertIn('format', self.client.calls[0])
        turns = self.store.recent_turns(self.agent.session_id, 10)
        self.assertEqual(turns[-1]['user_text'], 'Can you double check?')
        self.assertEqual(turns[-1]['route'], 'live_person')

    def test_contextual_questions_are_deferred_without_a_phrase_list(self):
        for question in ('Can you double check?', 'Please obtain another independent reading.',
                         'Has the situation changed since your last look?',
                         'I doubt that conclusion.', 'Explain how you reached that.'):
            for topic, previous, expected in (
                ('live_person', 'How about now?', 'live_followup'),
                ('live_visual', 'What can you see?', 'live_followup'),
                ('live', 'Is there anyone who drove right now?', 'live_followup'),
                ('general', 'What is 2 plus 3?', 'general'),
                ('live', 'Turn the light on', 'general'),
            ):
                with self.subTest(question=question, topic=topic):
                    self.assertEqual(a.route_question(question, topic, previous), expected)
            self.assertNotIn(a.route_question(
                question, 'memory_session', 'Was anyone here while I was away?',
            ), ('live', 'live_followup'))

    def test_repeated_live_followups_survive_history_truncation(self):
        self.agent.ask('Do you see anyone in the room?')
        for question in ('How about now?', 'Can you double check?',
                         'Could you double-check?', 'Please check again.', 'How about now?'):
            self.agent.ask(question)
        self.assertEqual(len(self.tools.calls), 12)
        self.assertEqual(self.agent.last_topic, 'live_person')
        self.assertEqual(self.agent.history[-2]['content'], 'How about now?')
        self.assertEqual(len(self.store.recent_turns(self.agent.session_id, 10)), 6)

    def test_live_followup_survives_restart_with_resume_last(self):
        self.agent.ask('Do you see anyone in the room?')
        fresh_tools = FakePhysicalTools()
        resumed = a.MemoryVoiceAgent(
            client=FakeClient(), tools=fresh_tools,
            conversation_store=self.store, resume_last=True,
        )
        with patch.object(resumed, 'trace', create=True):
            result = resumed.ask('Can you double check?')
        self.assertIn('camera reported 1 person', result.lower())
        self.assertEqual([name for name, _ in fresh_tools.calls],
                         ['detect_motion', 'observe_scene'])

    def test_followups_not_inherited_after_unrelated_questions(self):
        self.assertEqual(
            a.route_question('How about now?', 'general', 'What is 2+17?'),
            'general',
        )
        self.assertEqual(
            a.route_question('How about now?', 'live_person', 'How about now?'),
            'live_followup',
        )
        self.assertEqual(
            a.route_question('How about now?', 'live', 'Turn the light on'),
            'general',
        )
        self.assertEqual(
            a.route_question('What about after that?', 'live',
                             'Do you see anyone in the room?'),
            'live_followup',
        )
        self.assertEqual(
            a.route_question('How about now?', 'memory_session',
                             'Was anyone here while I was away?'),
            'general',
        )

    def test_visual_followup_reuses_safe_sensing_question(self):
        with patch.object(a.VoiceAgent, 'ask', return_value='Visual result') as ask_live:
            self.assertEqual(self.agent.ask('What can you see?'), 'Visual result')
            self.assertEqual(self.agent.ask('What about now?'), 'Visual result')
            self.assertEqual(self.agent.ask('Can you double check?'), 'Visual result')
            self.assertEqual(ask_live.call_count, 3)
            self.assertEqual(ask_live.call_args.args, ('What can you see right now?',))

    def test_explicit_topic_change_does_not_repeat_sensors(self):
        self.agent.ask('Do you see anyone in the room?')
        self.client.intent = 'change_topic'
        answer = self.agent.ask('What is 2 plus 3?')
        self.assertEqual(answer, 'Mock explanation.')
        self.assertEqual(len(self.tools.calls), 2)
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(self.agent.last_topic, 'general')
        self.assertIn('No sensors were checked', self.client.calls[-1]['messages'][-1]['content'])
        self.agent.ask('How about now?')
        self.assertEqual(len(self.tools.calls), 2)
        self.assertNotIn('format', self.client.calls[-1])

    def test_corrected_request_resets_answer_context_but_keeps_transcript(self):
        self.agent.ask('Do you see anyone in the room?')
        stale = 'The earlier camera did not detect anyone. The PIR reported no movement.'
        self.agent._remember('What is three plus nineteen?', stale, 'live_person_discussion')
        self.agent._remember('No, I meant a calculation.', stale, 'live_person_discussion')
        self.client.intent = 'change_topic'
        correction = 'I asked, "What\'s 3+19?"'
        self.agent.ask(correction)
        messages = self.client.calls[-1]['messages']
        self.assertEqual(len(messages), 2)
        self.assertIn(correction, messages[-1]['content'])
        self.assertNotIn(stale, str(messages))
        self.assertEqual(len(self.agent.history), 2)
        self.assertEqual(self.agent.last_topic, 'general')
        turns = self.store.recent_turns(self.agent.session_id, 10)
        self.assertEqual(len(turns), 4)
        self.assertEqual(turns[-1]['route'], 'general_new_topic')
        resumed = a.MemoryVoiceAgent(
            client=FakeClient(), tools=FakePhysicalTools(),
            conversation_store=self.store, resume_last=True,
        )
        self.assertEqual(resumed.history, self.agent.history)
        self.assertEqual(resumed.last_topic, 'general')
        resumed.ask('Why?')
        self.assertNotIn(stale, str(resumed.client.calls[-1]['messages']))
        self.assertEqual(len(resumed.client.calls), 1)

    def test_unsuccessful_topic_change_does_not_discard_existing_context(self):
        self.agent.ask('Do you see anyone in the room?')
        before = list(self.agent.history)
        self.client.intent = 'change_topic'
        original_chat = self.client.chat
        def fail_answer(**kwargs):
            if kwargs.get('format'):
                return original_chat(**kwargs)
            raise RuntimeError('generation failed')
        with patch.object(self.client, 'chat', side_effect=fail_answer):
            with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                self.agent.ask('What is 3 plus 19?')
        self.assertEqual(self.agent.history, before)
        self.assertEqual(self.agent.last_topic, 'live_person')
        self.assertEqual(len(self.store.recent_turns(self.agent.session_id)), 1)

    def test_clarification_of_garbled_arithmetic_recovers_on_clear_restatement(self):
        stale = 'The camera did not detect a person. The PIR reports no movement.'
        self.agent._remember('Is anyone in the room?', stale, 'live_person')
        responses = iter([
            SimpleNamespace(message=SimpleNamespace(content='{"focus":"new_subject"}')),
            SimpleNamespace(message=SimpleNamespace(content=(
                'I do not have enough context about "these three plus nineteen." '
                'Could you clarify what you mean?'))),
            SimpleNamespace(message=SimpleNamespace(content='3 + 19 = 22.')),
        ])
        with patch.object(self.agent.client, 'chat', side_effect=lambda **kwargs: next(responses)):
            clarification = self.agent.ask('No, I asked why these three plus nineteen.')
            self.assertIn('clarify', clarification)
            answer = self.agent.ask('I asked, "What\'s 3+19?"')
        self.assertEqual(answer, '3 + 19 = 22.')
        self.assertEqual(self.agent.last_topic, 'general')
        self.assertEqual(self.tools.calls, [])
        self.assertEqual(len(self.store.recent_turns(self.agent.session_id)), 3)

    def test_sensor_knowledge_gets_an_answer_and_preserves_observation_context(self):
        self.agent.ask('Do you see anyone in the room?')
        self.client.intent = 'explain_sensor'
        self.assertEqual(self.agent.ask('What does PIR mean?'), 'Mock explanation.')
        self.assertEqual(len(self.tools.calls), 2)
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(self.agent.last_topic, 'live_person_discussion')
        self.client.intent = 'repeat_observation'
        self.agent.ask('How about now?')
        self.assertEqual(len(self.tools.calls), 4)

    def test_logged_sequence_retains_live_context_during_discussion(self):
        self.tools.camera = {'ok': True, 'counts': {'person': 0}, 'detections': []}
        self.agent.ask('Do you see anyone in the room?')
        self.agent.ask('Okay, how about now?')
        self.client.intent = 'discuss_observation'
        reply = self.agent.ask('Really? No one?')
        self.assertIn("haven't taken a fresh reading", reply)
        self.assertIn('earlier observation', reply)
        self.assertEqual(len(self.tools.calls), 4)
        # No free-form answer call can invent a new reading for this discussion.
        self.assertTrue(all('format' in call for call in self.client.calls))
        self.assertEqual(self.agent.last_topic, 'live_person_discussion')
        self.client.intent = 'repeat_observation'
        self.tools.camera = {'ok': True, 'counts': {'person': 1}, 'detections': []}
        reply = self.agent.ask('How about now?')
        self.assertIn('camera reported 1 person', reply.lower())
        self.assertEqual([name for name, _ in self.tools.calls],
                         ['detect_motion', 'observe_scene'] * 3)

    def test_challenged_presence_can_refresh_both_sensors(self):
        self.agent.ask('Do you see anyone in the room?')
        self.agent.ask('Really? No one?')
        self.assertEqual([name for name, _ in self.tools.calls],
                         ['detect_motion', 'observe_scene'] * 2)
        self.assertEqual(self.agent.last_topic, 'live_person')

    def test_discussion_context_survives_restart_and_history_truncation(self):
        self.agent.ask('Do you see anyone in the room?')
        self.client.intent = 'discuss_observation'
        for _ in range(5):
            self.agent.ask('Explain what that result means.')
        self.assertEqual(len(self.agent.history), 6)
        self.assertEqual(len(self.tools.calls), 2)
        self.assertEqual(len(self.client.calls), 5)
        observation = self.store.latest_live_observation(self.agent.session_id)
        self.assertEqual(observation['route'], 'live_person')
        fresh_tools = FakePhysicalTools()
        resumed = a.MemoryVoiceAgent(
            client=FakeClient(), tools=fresh_tools,
            conversation_store=self.store, resume_last=True,
        )
        self.assertEqual(resumed.last_topic, 'live_person_discussion')
        with patch.object(resumed, 'trace', create=True):
            resumed.ask('How about now?')
        self.assertEqual([name for name, _ in fresh_tools.calls],
                         ['detect_motion', 'observe_scene'])

    def test_old_general_routing_label_clarifies_without_erasing_context(self):
        self.agent.ask('Do you see anyone in the room?')
        self.client.intent = 'general'
        reply = self.agent.ask('Really? No one?')
        self.assertIn('restate', reply)
        self.assertEqual(self.agent.last_topic, 'live_person_clarify')
        self.assertEqual(len(self.tools.calls), 2)
        self.client.intent = 'repeat_observation'
        self.agent.ask('How about now?')
        self.assertEqual(len(self.tools.calls), 4)

    def test_uncertain_decision_clarifies_before_taking_new_readings(self):
        self.agent.ask('Do you see anyone in the room?')
        self.client.intent = 'clarify'
        answer = self.agent.ask('Are you sure about that?')
        self.assertIn('restate', answer)
        self.assertEqual(len(self.tools.calls), 2)
        self.assertEqual(len(self.client.calls), 1)
        self.client.intent = 'repeat_observation'
        self.agent.ask('A new measurement, please.')
        self.assertEqual(len(self.tools.calls), 4)

    def test_two_sensors_run_even_if_pir_says_no_motion(self):
        with patch.object(a.VoiceAgent, 'ask', side_effect=AssertionError('Legacy model-controlled loop')):
            response = self.agent.ask('Do you see anyone in their room right now?')
        self.assertEqual(self.tools.calls, [
            ('detect_motion', {}),
            ('observe_scene', {'targets': ['person'], 'conf': 0.2}),
        ])
        self.assertIn('camera reported 1 person', response.lower())
        self.assertIn('pir currently reports no movement', response.lower())
        self.assertEqual(self.client.calls, [])
        # The conversational exchange is still saved to SQLite.
        turns = self.store.recent_turns(self.agent.session_id, 10)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]['route'], 'live_person')

    def test_no_person_is_not_proof_empty(self):
        self.tools.camera = {'ok': True, 'counts': {'person': 0}, 'detections': []}
        response = self.agent.ask('Do you see anyone in the room?')
        self.assertIn('did not detect a person', response)
        self.assertIn('does not prove the room is empty', response)
        self.assertEqual(len(self.tools.calls), 2)

    def test_camera_failure_does_not_create_false_negative(self):
        self.tools.camera = {'ok': False, 'error': 'Camera not available'}
        response = self.agent.ask('Do you see anyone in the room?')
        self.assertIn("couldn't obtain a valid camera", response)
        self.assertIn("can't determine whether anyone is present", response)
        self.assertEqual(len(self.tools.calls), 2)

    def test_unrelated_and_historical_queries_do_not_activate_live_sensors(self):
        self.assertEqual(a.route_question('What is 2+17?', 'memory_session'), 'general')
        self.assertEqual(a.route_question('Was anyone in the room?', 'general'), 'memory_session')
        self.assertEqual(a.route_question('What did you see most recently?', 'live'), 'memory_recent')
        self.assertEqual(a.route_question('Did you see anyone yesterday?', 'general'), 'memory_session')


if __name__ == '__main__':
    unittest.main()
