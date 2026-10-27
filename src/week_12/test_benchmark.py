"""Run with: python3 -m unittest discover -s src/week_12 -p 'test_*.py' -v"""
import unittest
from unittest.mock import patch
import benchmark


class BenchmarkTests(unittest.TestCase):
    def test_tool_schemas_have_no_required_arguments(self):
        for item in benchmark.TOOLS:
            self.assertEqual(item["function"]["parameters"]["properties"], {})

    def test_failure_is_unknown_not_absent(self):
        case = benchmark.SENSOR_RESULTS["failure_normalized"]
        self.assertEqual(case["presence"], "unknown")
        self.assertIsNone(case["camera"]["person_detected"])

    def test_agent_calls_two_tools_and_keeps_results(self):
        def fake_chat(_model, messages):
            if len(messages) == 1:
                return {"message": {"role": "assistant", "tool_calls": [
                    {"function": {"name": "check_presence", "arguments": {}}},
                    {"function": {"name": "memory_recent", "arguments": {}}},
                ]}}
            # Ensure mock errors / historical records are actually returned.
            observations = [m["content"] for m in messages if m["role"] == "tool"]
            self.assertTrue(any("timed out" in x for x in observations))
            self.assertTrue(any("earlier today" in x for x in observations))
            return {"message": {"role": "assistant", "content": "unknown"}}

        with patch.object(benchmark, "chat", side_effect=fake_chat):
            result = benchmark.agent_case("mock", "failure_raw")
        self.assertTrue(result["required_tools_ok"])
        self.assertEqual(result["answer"], "unknown")
        self.assertEqual(len(result["rounds"]), 2)

    def test_missing_memory_tool_is_detected(self):
        def fake_chat(_model, messages):
            if len(messages) == 1:
                return {"message": {"role": "assistant", "tool_calls": [
                    {"function": {"name": "check_presence", "arguments": {}}},
                ]}}
            return {"message": {"role": "assistant", "content": "done"}}

        with patch.object(benchmark, "chat", side_effect=fake_chat):
            result = benchmark.agent_case("mock", "failure_raw")
        self.assertFalse(result["required_tools_ok"])


if __name__ == "__main__":
    unittest.main()
