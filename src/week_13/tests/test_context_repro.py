"""Offline structural tests: run with `python3 -m unittest discover ...`."""

import importlib.util
import unittest
from pathlib import Path

WEEK = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, WEEK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prefix = load_module("prefix_cache_repro")
context = load_module("context_policy_repro")


class PrefixCacheStructure(unittest.TestCase):
    def test_changed_suffix_keeps_system_prefix(self):
        cases = prefix.different_suffix_cases(15)
        self.assertEqual(cases[0][1], cases[1][1])
        self.assertNotEqual(cases[0][2], cases[1][2])
        self.assertNotEqual(cases[0][1], cases[2][1])

    def test_longer_common_prefix(self):
        short = prefix.different_suffix_cases(15)
        long = prefix.different_suffix_cases(64)
        self.assertGreater(len(long[0][1]), len(short[0][1]))

    def test_cache_stats(self):
        row = prefix.metrics("identical", "Run 2", {
            "prompt_eval_count": 274,
            "prompt_eval_cached_count": 270,
            "prompt_eval_duration": 94_000_000,
            "message": {"content": "READY"},
        })
        self.assertEqual(row["uncached_tokens"], 4)
        self.assertEqual(row["prefill_s"], 0.094)


class ContextPolicyStructure(unittest.TestCase):
    def setUp(self):
        self.history = context.build_history()
        self.system = {"role": "system", "content": "Test"}
        self.ready = {"role": "user", "content": "Reply ACK."}
        self.first = {"role": "assistant", "content": "ACK"}
        self.question = {"role": "user", "content": "What was the code?"}

    def build(self, policy):
        return context.build_context(
            policy, self.system, self.history, self.ready, self.first, self.question
        )

    def test_full_retains_old_fact(self):
        self.assertIn(context.SECRET, str(self.build("full")))

    def test_recent_discards_old_fact(self):
        self.assertNotIn(context.SECRET, str(self.build("recent")))

    def test_summary_preserves_old_fact_in_reconstructed_history(self):
        summarized = self.build("summary")
        self.assertIn(context.SECRET, str(summarized))
        self.assertLess(len(summarized), len(self.build("full")))

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            self.build("unsupported")


if __name__ == "__main__":
    unittest.main()
