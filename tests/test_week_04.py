"""Offline regressions for reported Week 4 answers and scoring."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1] / "src" / "week_04"
sys.path.insert(0, str(ROOT))
import ollama_benchmark as benchmark
import compare_benchmarks as comparison


class BenchmarkTests(unittest.TestCase):
    def test_reported_results_match_all_failure_excerpts(self):
        data = json.loads((ROOT / "benchmark_cases.json").read_text())
        self.assertEqual(data["name"], "edge-ai-mini-v2")
        cases = {case["id"]: case for case in data["cases"]}
        failures = json.loads((ROOT / "results/reported_failures.json").read_text())["failures"]
        results = json.loads((ROOT / "results/reported_results.json").read_text())["models"]
        self.assertEqual(len(failures), 58)
        for model in results:
            rows = [r for r in failures if r["model"] == model["model"]]
            for row in rows:
                self.assertEqual(benchmark.grade(row, cases[row["id"]]),
                                 {key: row[key] for key in benchmark.grade(row, cases[row["id"]])})
            q = model["quality_summary"]
            self.assertEqual(32 - len(rows), q["passed"])
            self.assertEqual(q["passed"] + sum(r["correct"] for r in rows), q["correct_answers"])
            self.assertEqual(q["passed"] + sum(comparison.content_correct(r) for r in rows),
                             model["content_diagnostic"]["correct_answers"])

    def test_content_diagnostic_does_not_change_typed_scoring(self):
        case = {"expected": 42}
        row = {"content": '{"humidity_percentage":"42"}', "done": True, "stop_reason": "stop"}
        row.update(benchmark.grade(row, case))
        row["expected"] = 42
        self.assertFalse(row["correct"])
        self.assertFalse(row["valid_format"])
        self.assertTrue(comparison.content_correct(row))
        for text in ('The answer is 42.', '{"answer":12,"answer":42}', '{"answer":NaN}',
                     '{"answer":true}', '{"a":42,"b":12}'):
            self.assertFalse(comparison.content_correct({**row, "content": text}))
        self.assertFalse(comparison.content_correct({**row, "completed": False}))

    def test_offline_rescoring_preserves_responses_and_speed(self):
        case = {"id": "number", "expected": 42}
        row = {"id": "number", "expected": 42, "category": "test", "content": "42",
               "done": True, "stop_reason": "stop", "wall_time_s": 1}
        report = {"dataset": "original", "dataset_sha256": "original-hash",
                  "speed_summary": {"tokens_per_second": 7.85}, "quality_runs": [row]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(report))
            before = path.read_bytes()
            updated = benchmark.rescore(path, [case], "scoring-hash")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(updated["speed_summary"], report["speed_summary"])
            self.assertEqual(updated["dataset_sha256"], "original-hash")
            self.assertEqual(updated["quality_summary"]["answer_accuracy_percent"], 100)
            self.assertEqual(updated["quality_summary"]["accuracy_percent"], 0)
            with self.assertRaises(ValueError):
                benchmark.rescore(path, [{**case, "expected": 99}], "scoring-hash")


if __name__ == "__main__":
    unittest.main()
