import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import ollama
except ImportError:
    sys.modules['ollama'] = types.SimpleNamespace(Client=object)
from agent import VoiceAgent, requires_occupancy_evidence, guard_occupancy_answer


def msg(content='', calls=None):
    return types.SimpleNamespace(content=content, tool_calls=calls)

def call(name, args):
    return types.SimpleNamespace(function=types.SimpleNamespace(name=name, arguments=args))

class EvidencePolicyTests(unittest.TestCase):
    def agent(self, responses, motion=False, people=0, fail_camera=False):
        client = Mock()
        client.chat.side_effect = [types.SimpleNamespace(message=r) for r in responses]
        tools = Mock()
        def execute(name, args):
            if name == 'detect_motion':
                return {'ok': True, 'motion_signal_active': motion}
            if name == 'observe_scene':
                if fail_camera:
                    return {'ok': False, 'error': 'Camera offline'}
                return {'ok': True, 'counts': {'person': people}}
            return {'ok': False, 'error': 'bad tool'}
        tools.execute.side_effect = execute
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return VoiceAgent(client=client, tools=tools, trace_path=Path(directory.name)/'trace.jsonl'), client, tools

    def test_policy_classifier(self):
        self.assertTrue(requires_occupancy_evidence('Is anybody in the room right now?'))
        self.assertTrue(requires_occupancy_evidence('How many people are in the room?'))
        self.assertFalse(requires_occupancy_evidence('Do you see anyone in the room?'))
        self.assertFalse(requires_occupancy_evidence('What is 2 plus 17?'))
        self.assertTrue(requires_occupancy_evidence('Do you see anyone in the room moving?'))

    def test_pir_first_camera_completed(self):
        a,c,t = self.agent([msg(calls=[call('detect_motion',{})]), msg('No one is here.')])
        answer = a.ask('Is anybody in the room right now?')
        self.assertIn('can\'t confirm', answer)
        self.assertEqual([x.args[0] for x in t.execute.call_args_list], ['detect_motion','observe_scene'])
        history = c.chat.call_args_list[1].kwargs['messages']
        self.assertEqual(len([m for m in history if isinstance(m,dict) and m.get('role')=='tool']), 2)

    def test_no_calls_forced_both(self):
        a,c,t = self.agent([msg('Looks empty'), msg('Still empty')])
        answer = a.ask('How many people are in the room?')
        self.assertIn('can\'t confirm', answer)
        self.assertEqual([x.args[0] for x in t.execute.call_args_list], ['detect_motion','observe_scene'])

    def test_already_both_no_duplicate(self):
        a,c,t = self.agent([msg(calls=[call('detect_motion',{}), call('observe_scene', {'targets':['person'],'conf':0.2})]), msg('One candidate')], people=1)
        self.assertEqual(a.ask('Is anyone in the room?'), 'One candidate')
        self.assertEqual(t.execute.call_count,2)

    def test_visual_only_not_force_pir(self):
        a,c,t = self.agent([msg(calls=[call('observe_scene', {'targets':['person']})]),msg('None')])
        self.assertEqual(a.ask('Can you see anyone?'), 'None')
        self.assertEqual([x.args[0] for x in t.execute.call_args_list], ['observe_scene'])

    def test_camera_failure_not_misrepresented(self):
        a,c,t = self.agent([msg(calls=[call('detect_motion',{})]), msg('Empty room')], fail_camera=True)
        answer = a.ask('Is anybody in the room?')
        self.assertIn("couldn't get reliable data from the camera", answer)
        self.assertEqual(t.execute.call_count,2)

if __name__=='__main__': unittest.main()
