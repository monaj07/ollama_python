"""Non-hardware tests for the Week 10 finalisation benchmark."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import evaluate_routes, load_cases, parse_voice_log


class BenchmarkTests(unittest.TestCase):
    def test_valid_unique_fixture_ids(self):
        cases = load_cases()
        self.assertGreaterEqual(len(cases), 20)
        self.assertEqual(len({x["id"] for x in cases}), len(cases))

    def test_evaluate_routes_reports_failures_not_hides(self):
        cases = [
            {"id": "one", "question": "hello", "expected_route": "general"},
            {"id": "two", "question": "check now", "expected_route": "live"},
        ]
        result = evaluate_routes(cases, lambda q, *_: "general")
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["total"], 2)
        self.assertEqual([r["id"] for r in result["misroutes"]], ["two"])

    def test_voice_log_parsing(self):
        log = """[user] Is anyone there?
[route] live
[policy-tool] detect_motion({})
[policy-tool] observe_scene({})
[total_s] 9.12
[user] What is 2+17?
[route] general
[total_s] 2.46
"""
        result = parse_voice_log(log)
        self.assertEqual(result["turn_count"], 2)
        self.assertEqual(result["routes"], {"live": 1, "general": 1})
        self.assertEqual(result["physical_tool_calls"]["observe_scene"], 1)
        self.assertEqual(result["agent_latency_s"]["median"], 5.79)

    def test_empty_log(self):
        result = parse_voice_log("no turn here")
        self.assertEqual(result["turn_count"], 0)
        self.assertIsNone(result["agent_latency_s"]["median"])


if __name__ == "__main__":
    unittest.main()
