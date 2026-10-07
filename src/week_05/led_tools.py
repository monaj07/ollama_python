"""Allowlisted LED action; importing this module never opens GPIO devices."""
from time import sleep
from typing import Literal

from pydantic import BaseModel, ConfigDict


class SetLEDArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    colour: Literal["green", "red", "off"]


class LEDController:
    def __init__(self, green, red):
        self.green = green
        self.red = red
        self.state = "off"

    def off(self):
        self.green.off()
        self.red.off()
        self.state = "off"

    def set_led(self, colour: str) -> dict:
        args = SetLEDArgs.model_validate({"colour": colour})
        self.off()
        if args.colour == "green":
            self.green.on()
        elif args.colour == "red":
            self.red.on()
        self.state = args.colour
        return {"commanded_state": self.state}

    def execute(self, tool_name: str, arguments: dict) -> dict:
        if tool_name != "set_led":
            raise ValueError(f"Unknown tool: {tool_name}")
        args = SetLEDArgs.model_validate(arguments)
        return self.set_led(args.colour)


def main():
    from gpiozero import LED

    with LED(17, initial_value=False) as green, LED(27, initial_value=False) as red:
        controller = LEDController(green, red)
        try:
            for colour in ("green", "red", "off"):
                print("EXECUTED:", controller.execute("set_led", {"colour": colour}))
                sleep(2)
            for name, arguments in (
                ("set_led", {"colour": "blue"}),
                ("set_led", {"colour": "green", "pin": 17}),
                ("set_led", {}),
                ("run_shell", {"command": "echo hello"}),
            ):
                try:
                    controller.execute(name, arguments)
                except ValueError as error:
                    print(f"REJECTED: {name} {arguments} ({type(error).__name__})")
                else:
                    raise AssertionError(f"Invalid request was accepted: {name}")
        except KeyboardInterrupt:
            pass
        finally:
            controller.off()


if __name__ == "__main__":
    main()
