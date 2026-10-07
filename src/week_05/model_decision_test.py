"""Inspect and validate local tool decisions without touching GPIO."""
import os
from time import perf_counter

from led_tools import SetLEDArgs

MODEL = os.environ.get("MODEL", "gemma4:e2b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
SYSTEM_PROMPT = """
You control a red and a green indicator LED through set_led.

For each user instruction, call set_led exactly once.
Use green for ready or successful.
Use red for a problem or failure.
Use off to clear the indicator.
Follow an explicit request for green, red, or off.
For ambiguous or unsupported requests, choose off.

Do not claim to have measured device health.
Do not invent tools or GPIO pins.
"""
TOOLS = [{
    "type": "function",
    "function": {
        "name": "set_led",
        "description": (
            "Select green, red, or off. Green and red are mutually exclusive; "
            "off turns both LEDs off."
        ),
        "parameters": SetLEDArgs.model_json_schema(),
    },
}]


def validate_decision(response) -> SetLEDArgs:
    calls = response.message.tool_calls or []
    if len(calls) != 1:
        raise ValueError(f"Expected exactly one tool call; received {len(calls)}")
    function = calls[0].function
    if function.name != "set_led":
        raise ValueError(f"Unknown tool: {function.name}")
    return SetLEDArgs.model_validate(function.arguments)


def request_decision(client, instruction):
    return client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        tools=TOOLS,
        think=False,
        stream=False,
        keep_alive="10m",
        options={"temperature": 0, "num_ctx": 2048, "num_predict": 128},
    )


def main():
    import httpx
    import ollama

    client = ollama.Client(host=OLLAMA_HOST, timeout=120.0)
    print(f"Model: {MODEL}")
    print("Dry run: LEDs will not change. Type quit to exit.")
    while True:
        try:
            instruction = input("\nInstruction: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            break
        if instruction.lower() in {"quit", "exit"}:
            break
        if not instruction:
            continue
        started = perf_counter()
        try:
            response = request_decision(client, instruction)
            print("Model message:", response.message.model_dump())
            args = validate_decision(response)
            print(f"VALIDATED: set_led(colour={args.colour!r})")
            print("DRY RUN: hardware execution skipped.")
        except ValueError as error:
            print(f"REJECTED: {error}")
        except (ollama.ResponseError, httpx.HTTPError) as error:
            print(f"REQUEST FAILED: {error}")
        finally:
            print(f"Elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
