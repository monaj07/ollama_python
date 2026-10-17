"""Unit tests run without Ollama, GPIO or camera hardware."""
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

from agent import VoiceAgent
from tools import validate_call


def message(content='', tool_calls=None):
    return types.SimpleNamespace(content=content, tool_calls=tool_calls)


def call(name, args):
    return types.SimpleNamespace(function=types.SimpleNamespace(name=name, arguments=args))


class TestAgent(unittest.TestCase):
    def make_agent(self, responses):
        client = Mock()
        client.chat.side_effect = [types.SimpleNamespace(message=x) for x in responses]
        tools = Mock()
        tools.execute.side_effect = lambda name, args: {'ok': True, 'source': name}
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        agent = VoiceAgent('gemma4:e2b', client=client, tools=tools,
                           trace_path=Path(tempdir.name) / 'events.jsonl')
        return agent, client, tools

    def test_plain_question(self):
        agent, client, tools = self.make_agent([message('19')])
        self.assertEqual(agent.ask('What is 2+17?'), '19')
        tools.execute.assert_not_called()
        self.assertEqual(client.chat.call_count, 1)

    def test_two_tools_one_round(self):
        agent, client, tools = self.make_agent([
            message(tool_calls=[call('detect_motion', {}),
                                call('observe_scene', {'targets': ['person']})]),
            message('Movement detected; one person visible.'),
        ])
        self.assertIn('person', agent.ask('Is anyone here?'))
        self.assertEqual(tools.execute.call_count, 2)
        tool_msgs = client.chat.call_args_list[1].kwargs['messages']
        self.assertEqual(len([m for m in tool_msgs if isinstance(m, dict) and m.get('role') == 'tool']), 2)

    def test_invalid_args_do_not_execute(self):
        agent, client, tools = self.make_agent([
            message(tool_calls=[call('detect_motion', {'pin': 18})]),
            message('I could not check motion.'),
        ])
        self.assertIn('could not', agent.ask('Any motion?'))
        tools.execute.assert_not_called()

    def test_repeated_call_not_executed_again(self):
        agent, client, tools = self.make_agent([
            message(tool_calls=[call('detect_motion', {})]),
            message(tool_calls=[call('detect_motion', {})]),
            message('One sensor reading was available.'),
        ])
        agent.ask('Motion?')
        self.assertEqual(tools.execute.call_count, 1)

    def test_no_unbounded_loop(self):
        agent, client, tools = self.make_agent([
            message(tool_calls=[call('detect_motion', {})]),
            message(tool_calls=[call('observe_scene', {'targets':['person']})]),
            message(tool_calls=[call('observe_scene', {'targets':['cup']})]),
            message('Completed'),
        ])
        self.assertEqual(agent.ask('Check'), 'Completed')
        self.assertEqual(client.chat.call_count, 4)
        self.assertEqual(client.chat.call_args_list[3].kwargs['tools'], [])

    def test_validation(self):
        with self.assertRaises(ValueError):
            validate_call('observe_scene', {'targets': ['person'], 'conf': True})
        with self.assertRaises(ValueError):
            validate_call('observe_scene', {'targets': ['person', 'PERSON']})
        with self.assertRaises(ValueError):
            validate_call('observe_scene', {'targets': ['person'], 'shell': 'ls'})
        self.assertEqual(validate_call('observe_scene', '{"targets": ["person"]}'),
                         {'targets': ['person'], 'conf': 0.2})


if __name__ == '__main__':
    unittest.main()
