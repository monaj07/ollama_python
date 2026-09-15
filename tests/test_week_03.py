"""Run with: .venv/bin/python -m unittest discover -s tests -v"""

import builtins
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import ollama


NOTEBOOK = Path(__file__).resolve().parents[1] / "src" / "week_03.ipynb"


def response(content="Done", tool_calls=None):
    return ollama.ChatResponse(
        model="test",
        message={"role": "assistant", "content": content, "tool_calls": tool_calls},
        total_duration=1,
        load_duration=1,
        eval_count=1,
        eval_duration=1,
    )


class MemoryNotebookTests(unittest.TestCase):
    def setUp(self):
        self.cells = json.loads(NOTEBOOK.read_text())["cells"]
        self.ns = {}
        exec("".join(self.cells[2]["source"]), self.ns)
        self.db = self.ns["Database"](":memory:")
        self.addCleanup(self.db.close)
        self.ns["db"] = self.db
        for index in (8, 10):
            exec("".join(self.cells[index]["source"]), self.ns)
        self.db.remember("I like carrot juice.", "preference")

    def run_chat(self, prompts, responses):
        with (
            patch.object(builtins, "input", side_effect=[*prompts, "exit"]),
            patch.object(ollama, "chat", side_effect=responses) as chat,
            redirect_stdout(io.StringIO()),
        ):
            exec("".join(self.cells[12]["source"]), self.ns)
        return chat.call_args_list

    def test_retrieves_even_when_model_never_requests_a_tool(self):
        calls = self.run_chat(
            ["What do I like?", "What do you know about me in your memory?"],
            [response(), response()],
        )
        runs = self.db.connection.execute(
            "SELECT tool_name, status FROM tool_runs ORDER BY id"
        ).fetchall()
        self.assertEqual([tuple(row) for row in runs], [("search_memory", "success")] * 2)
        for call in calls:
            result = call.kwargs["messages"][-1]
            self.assertEqual(result["role"], "tool")
            self.assertEqual(result["tool_name"], "search_memory")
            self.assertEqual(json.loads(result["content"])[0]["text"], "I like carrot juice.")

    def test_next_turn_retrieves_newly_saved_memory(self):
        calls = self.run_chat(
            ["Remember that I like mangoes.", "What do I like?"],
            [
                response("", [{"function": {
                    "name": "remember",
                    "arguments": {"text": "I like mangoes.", "category": "preference"},
                }}]),
                response("Saved"),
                response(),
            ],
        )
        memories = json.loads(calls[-1].kwargs["messages"][-1]["content"])
        self.assertEqual([row["text"] for row in memories], [
            "I like carrot juice.", "I like mangoes.",
        ])

    def test_history_keeps_complete_tool_exchanges(self):
        calls = self.run_chat(["What do I like?"] * 5, [response()] * 5)
        self.assertEqual(
            sum(message["role"] == "user" for message in calls[1].kwargs["messages"]),
            2,
        )
        for call in calls:
            messages = call.kwargs["messages"][1:]  # Exclude the system instruction.
            self.assertEqual(messages[0]["role"], "user")
            for index, message in enumerate(messages):
                if message["role"] == "tool":
                    self.assertEqual(messages[index - 1]["role"], "assistant")
                    self.assertTrue(messages[index - 1]["tool_calls"])

    def test_search_accepts_broad_queries_and_categories(self):
        for query in ("", "%", "  ", "preference", "CARROT"):
            validated = self.ns["SearchMemoryInput"].model_validate({"query": query})
            self.assertEqual(len(self.db.search_memory(**validated.model_dump())), 1)
        self.assertEqual(self.db.search_memory("mango"), [])
        self.assertEqual(self.db.search_memory("' OR 1=1 --"), [])


if __name__ == "__main__":
    unittest.main()
