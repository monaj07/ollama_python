"""Offline checks of the boundary between proposed actions and LED outputs."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1] / "src" / "week_05"
sys.path.insert(0, str(ROOT))
from led_tools import LEDController
from model_decision_test import validate_decision


class Output:
    def __init__(self, name, events):
        self.name, self.events, self.value = name, events, False

    def on(self):
        self.value = True
        self.events.append((self.name, True))

    def off(self):
        self.value = False
        self.events.append((self.name, False))


def response(*calls):
    return SimpleNamespace(message=SimpleNamespace(tool_calls=[
        SimpleNamespace(function=SimpleNamespace(name=name, arguments=args))
        for name, args in calls
    ]))


class LEDBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.green = Output("green", self.events)
        self.red = Output("red", self.events)
        self.controller = LEDController(self.green, self.red)

    def test_rejected_requests_do_not_write_outputs(self):
        self.controller.execute("set_led", {"colour": "green"})
        before = self.events.copy()
        for name, args in (
            ("run_shell", {"command": "anything"}),
            ("set_led", {"colour": "blue"}),
            ("set_led", {"colour": "red", "pin": 27}),
            ("set_led", {}),
            ("set_led", {"colour": True}),
        ):
            with self.subTest(name=name, args=args), self.assertRaises(ValueError):
                self.controller.execute(name, args)
            self.assertEqual(self.events, before)
            self.assertTrue(self.green.value)
            self.assertFalse(self.red.value)

    def test_switching_clears_both_outputs_before_enabling_red(self):
        self.controller.set_led("green")
        self.events.clear()
        self.assertEqual(self.controller.set_led("red"), {"commanded_state": "red"})
        self.assertEqual(self.events, [("green", False), ("red", False), ("red", True)])
        self.controller.off()
        self.assertFalse(self.green.value)
        self.assertFalse(self.red.value)

    def test_multiple_actions_are_rejected_before_any_execution(self):
        calls = (("set_led", {"colour": "green"}), ("set_led", {"colour": "red"}))
        with self.assertRaises(ValueError):
            validate_decision(response(*calls))
        self.assertEqual(self.events, [])

    def test_missing_or_invalid_decision_is_rejected(self):
        for result in (response(), response(("other", {})),
                       response(("set_led", {"colour": "blue"}))):
            with self.assertRaises(ValueError):
                validate_decision(result)
        self.assertEqual(validate_decision(response(("set_led", {"colour": "off"}))).colour, "off")


if __name__ == "__main__":
    unittest.main()
