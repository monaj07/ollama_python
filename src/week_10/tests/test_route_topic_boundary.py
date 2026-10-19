"""Independent questions after a live check bypass the follow-up model."""
import unittest
from agent import route_question


class TopicBoundaryTests(unittest.TestCase):
    def route(self, question):
        return route_question(question, "live_person", "Do you see anyone in the room?")

    def test_clear_independent_questions(self):
        for question in ("What is 2 + 17?", "Calculate (4 * 3) + 2",
                         "Tell me about Roman history", "Explain photosynthesis",
                         "Describe Jupiter"):
            with self.subTest(question=question):
                self.assertEqual(self.route(question), "general")

    def test_ambiguous_followups_still_classified(self):
        for question in ("How about now?", "Are you sure?",
                         "Explain that result", "Tell me about the camera"):
            with self.subTest(question=question):
                self.assertEqual(self.route(question), "live_followup")

    def test_explicit_live_and_history(self):
        self.assertEqual(self.route("What did you see most recently?"), "memory_recent")
        self.assertEqual(self.route("Was anyone here while I was away?"), "memory_session")
        self.assertEqual(self.route("Do you see anyone in the room?"), "live")


if __name__ == "__main__":
    unittest.main()
