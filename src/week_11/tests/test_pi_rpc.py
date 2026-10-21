"""Offline protocol tests: exercise the real PiRPCClient against a fake RPC child."""
from pathlib import Path
import sys
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from pi_rpc import PiRPCClient, PiRPCError


class TestPiRPC(unittest.TestCase):
    def client(self, **kw):
        return PiRPCClient(command=[sys.executable, str(Path(__file__).with_name("fake_pi_rpc.py"))],
                           cwd=BASE, timeout_s=3, startup_timeout_s=3, **kw)

    def test_two_turns_share_a_single_process(self):
        with self.client() as pi:
            a = pi.ask("hello")
            b = pi.ask("again")
        self.assertEqual(a.text, "answer 1")
        self.assertEqual(b.text, "answer 2")
        self.assertGreaterEqual(a.elapsed_s, 0)

    def test_tool_round_waits_for_final_text(self):
        with self.client() as pi:
            result = pi.ask("tool")
        self.assertEqual(result.text, "answer 1")
        self.assertEqual(result.tools, ["query_memory"])
        self.assertAlmostEqual(result.tool_seconds["query_memory"], .25)
        self.assertEqual(result.input_tokens, 14)
        self.assertEqual(result.output_tokens, 5)
        self.assertNotIn("Don't speak", result.text)

    def test_truncated_response_is_not_spoken(self):
        with self.client() as pi:
            with self.assertRaisesRegex(PiRPCError, "stopReason=length"):
                pi.ask("truncated")

    def test_rejected_response_is_not_spoken(self):
        with self.client() as pi:
            with self.assertRaisesRegex(PiRPCError, "mock rejection"):
                pi.ask("reject")

    def test_timeout_does_not_reuse_unknown_agent_state(self):
        pi = PiRPCClient(command=[sys.executable, str(Path(__file__).with_name("fake_pi_rpc.py"))],
                         cwd=BASE, timeout_s=.15, startup_timeout_s=3)
        try:
            pi.start()
            with self.assertRaisesRegex(PiRPCError, "timed out"):
                pi.ask("hang")
            with self.assertRaisesRegex(PiRPCError, "restart the client"):
                pi.ask("next")
        finally:
            pi.close()

    def test_compact_system_prompt_is_included_in_default_rpc_command(self):
        prompt = BASE / "voice_system.md"
        pi = PiRPCClient(cwd=BASE, system_prompt=prompt)
        self.assertIn("--system-prompt", pi.argv)
        pos = pi.argv.index("--system-prompt")
        self.assertEqual(pi.argv[pos + 1], str(prompt.resolve()))
        self.assertNotIn("--no-context-files", pi.argv)

    def test_context_files_can_be_disabled_for_ablation(self):
        pi = PiRPCClient(cwd=BASE, system_prompt=BASE / "voice_system.md", no_context_files=True)
        self.assertIn("--no-context-files", pi.argv)

    def test_missing_prompt_is_rejected_before_process_start(self):
        with self.assertRaisesRegex(ValueError, "System prompt file not found"):
            PiRPCClient(cwd=BASE, system_prompt=BASE / "missing.md")

    def test_conflicting_session_options(self):
        with self.assertRaises(ValueError):
            self.client(no_session=True, continue_session=True)


if __name__ == "__main__":
    unittest.main()
