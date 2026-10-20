"""Protocol smoke tests; these do not activate sensors or load YOLOE."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "physical_worker.py"


def exchange(*requests):
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    completed = subprocess.run(
        [sys.executable, str(WORKER)], input=payload, text=True,
        capture_output=True, check=True, timeout=10,
    )
    return [json.loads(line) for line in completed.stdout.splitlines()]


class PhysicalWorkerTests(unittest.TestCase):
    def test_health_and_repeated_requests(self):
        results = exchange({"tool": "health"}, {"tool": "health", "args": {}})
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result == {"ok": True, "status": "ready"} for result in results))

    def test_unsupported_tool_rejected(self):
        result, = exchange({"tool": "bash", "args": {"cmd": "whoami"}})
        self.assertIs(result["ok"], False)
        self.assertIn("Unknown tool", result["error"])

    def test_invalid_pir_arguments_rejected(self):
        result, = exchange({"tool": "detect_motion", "args": {"unexpected": 42}})
        self.assertIs(result["ok"], False)
        self.assertIn("ValueError", result["error"])


if __name__ == "__main__":
    unittest.main()
