"""Actual SQLite persistence and retrieval, using temporary files and fake sensors."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from ollama import ChatResponse

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / 'src/week_09', ROOT / 'src/week_10'):
    sys.path.insert(0, str(path))

import memory_store as store
from memory_tools import get_observation_history_report
from live_memory import RecordingTools
from src.week_10.agent import MemoryVoiceAgent
from src.week_10.conversation_store import ConversationStore


def camera_result(count=1, *, stamp='2026-10-19T05:31:21.871399+00:00', score=0.853):
    return {
        'ok': True, 'observed_at': stamp,
        'counts': {'person': count}, 'person_count_threshold': 0.3,
        'detections': [{'label': 'person', 'confidence': score}] if count else [],
        'source': '/captured/scene.jpg',
    }


def chat_response(content='', tool_calls=None):
    """Match Ollama's message and timing contract, including Pi instrumentation."""
    return ChatResponse(
        model='test-model', done=True,
        message={'role': 'assistant', 'content': content, 'tool_calls': tool_calls or []},
        total_duration=3_000_000, load_duration=1_000_000,
        prompt_eval_duration=1_000_000, eval_duration=1_000_000,
        prompt_eval_count=1, eval_count=1,
    )


class LiveMemoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'events.sqlite3'
        patched = patch.object(store, 'DB_PATH', self.path)
        patched.start()
        self.addCleanup(patched.stop)
        self.physical = Mock()
        self.tools = RecordingTools(self.physical, 'conversation-123')

    def test_positive_snapshot_keeps_capture_time_score_image_and_pir(self):
        self.physical.execute.side_effect = [
            {'ok': True, 'motion_signal_active': False}, camera_result(),
        ]
        self.tools.execute('detect_motion', {})
        self.tools.execute('observe_scene', {'targets': ['person']})
        row = store.recent_observations()[0]
        self.assertEqual(row['trigger'], 'live_observation')
        self.assertEqual(row['observed_at_utc'], '2026-10-19T05:31:21.871399+00:00')
        self.assertEqual(row['person_detected'], 1)
        self.assertEqual(row['detection_score'], 0.853)
        self.assertEqual(row['evidence_path'], '/captured/scene.jpg')
        self.assertIsNone(row['away_session_id'])
        details = json.loads(row['details_json'])
        self.assertIs(details['pir_signal_active'], False)
        self.assertEqual(details['conversation_session_id'], 'conversation-123')
        self.assertEqual(details['timestamp_source'], 'camera_receipt')

    def test_negative_and_failed_camera_remain_distinct_and_pir_is_not_reused(self):
        self.physical.execute.side_effect = [
            {'ok': True, 'motion_signal_active': True}, camera_result(0),
            {'ok': False, 'error': 'offline'},
        ]
        self.tools.execute('detect_motion', {})
        self.tools.execute('observe_scene', {'targets': ['person']})
        self.tools.begin_turn()
        self.tools.execute('observe_scene', {'targets': ['person']})
        rows = store.recent_observations()
        self.assertIsNone(rows[0]['person_detected'])
        self.assertIsNone(rows[0]['detection_score'])
        self.assertIsNone(json.loads(rows[0]['details_json'])['pir_signal_active'])
        self.assertEqual(rows[1]['person_detected'], 0)
        self.assertIsNone(rows[1]['detection_score'])

    def test_exception_records_unknown_once_and_propagates(self):
        self.physical.execute.side_effect = RuntimeError('camera offline')
        with self.assertRaisesRegex(RuntimeError, 'camera offline'):
            self.tools.execute('observe_scene', {'targets': ['person']})
        rows = store.recent_observations()
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]['person_detected'])

    def test_live_camera_call_from_week09_loop_is_also_saved(self):
        self.physical.execute.return_value = {
            'ok': True, 'observed_at': '2026-10-19T05:31:00+00:00',
            'counts': {'cup': 1}, 'detections': [{'label': 'cup', 'confidence': 0.8}],
        }
        client = Mock()
        client.chat.side_effect = [
            chat_response(tool_calls=[
                {'function': {'name': 'observe_scene', 'arguments': {'targets': ['cup']}}},
            ]),
            chat_response(content='One cup.'),
        ]
        agent = MemoryVoiceAgent(client=client, tools=self.physical,
                                 conversation_store=ConversationStore(self.path),
                                 trace_path=self.path.parent / 'trace.jsonl')
        self.assertEqual(agent.ask('What can you see?'), 'One cup.')
        rows = store.recent_observations()
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]['person_detected'])
        self.assertEqual(json.loads(rows[0]['details_json'])['targets'], ['cup'])

    def test_history_finds_last_live_positive_after_restart_and_many_negatives(self):
        session = store.arm_away_mode()
        store.add_observation('pir_motion', person_detected=True, detection_score=0.466,
                              observed_at_utc='2026-10-19T01:53:19+00:00')
        store.disarm_away_mode()
        self.physical.execute.return_value = camera_result(stamp='2026-10-19T05:29:54.131281+00:00', score=0.754)
        self.tools.execute('observe_scene', {'targets': ['person']})
        self.physical.execute.return_value = camera_result()
        self.tools.execute('observe_scene', {'targets': ['person']})
        for minute in range(6):
            store.add_observation('live_observation', person_detected=False,
                                  observed_at_utc=f'2026-10-19T06:0{minute}:00+00:00')

        # All retrieval here uses newly opened connections, not conversation text.
        report = get_observation_history_report()
        self.assertEqual(report['last_person_detection_local'], '2026-10-19T16:31:21+11:00')
        self.assertEqual(report['positive_events'][0]['score'], 0.853)
        self.assertEqual(report['latest_away_session']['session_id'], session)
        self.assertEqual(report['latest_away_session']['last_person_detection_local'],
                         '2026-10-19T12:53:19+11:00')
        self.assertEqual(report['person_positive_events'], 3)
        self.assertEqual(report['total_events'], 9)
        self.assertTrue(all(row['person_detected'] == 0 for row in store.recent_observations()))

        resumed = MemoryVoiceAgent(client=Mock(), tools=Mock(),
                                   conversation_store=ConversationStore(self.path))
        with patch.object(resumed, '_one_llm_call', return_value='Latest saved detection.') as answer:
            resumed.ask('When was the last time you saw someone in the room?')
        evidence = answer.call_args.args[1]
        self.assertEqual(evidence['scope'], 'all_saved_observations')
        self.assertEqual(evidence['last_person_detection_local'], '2026-10-19T16:31:21+11:00')

    def test_live_positive_is_retrievable_without_any_away_session(self):
        self.physical.execute.return_value = camera_result()
        self.tools.execute('observe_scene', {'targets': ['person']})
        report = get_observation_history_report()
        self.assertTrue(report['ok'])
        self.assertFalse(report['latest_away_session']['ok'])
        self.assertEqual(report['last_person_detection_local'], '2026-10-19T16:31:21+11:00')


if __name__ == '__main__':
    unittest.main()
