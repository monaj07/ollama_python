import json
import time

import ollama
from pydantic import BaseModel, ConfigDict

from motion_tools import MotionTools


MODEL = "gemma4:e2b"
MAX_TOOL_ROUNDS = 3

SYSTEM_PROMPT = """
You are a concise assistant connected to a Raspberry Pi PIR sensor.

Before answering questions about current movement or room occupancy,
call detect_motion to obtain a fresh observation.

For general questions unrelated to physical observations, do not call it.

Interpret motion_signal_active carefully:
- true: the sensor's motion output was active at the observation time.
- false: the sensor's motion output was inactive at that time.
Neither result establishes whether the room is occupied or empty.

The sensor has a hardware hold time. An active signal can persist
briefly after movement stops. Describe readings as observations,
not guarantees about conditions at the time your answer arrives.

If a tool returns ok=false, explain that the reading is unavailable.
Do not invent observations.

You have no other sensors, other than the PIR sensor.
Answer in at most three short sentences.
"""


class MotionArguments(BaseModel):
    # This tool accepts no arguments, including no GPIO pin argument.
    model_config = ConfigDict(extra="forbid", strict=True)


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "detect_motion",
        "description": (
            "Read the current PIR motion output and observation timestamp. "
            "This detects movement signals, not guaranteed occupancy."
        ),
        "parameters": MotionArguments.model_json_schema(),
    },
}


def answer_question(client, hardware, question):
    # Each question starts fresh, so previous observations cannot be reused.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for round_number in range(MAX_TOOL_ROUNDS + 1):
        response = client.chat(
            model=MODEL,
            messages=messages,
            tools=[TOOL_SCHEMA] if round_number < MAX_TOOL_ROUNDS else [],
            think=False,
            keep_alive="10m",
            options={
                "temperature": 0,
                "num_ctx": 2048,
                "num_predict": 256,
            },
        )

        message = response.message
        messages.append(message.model_dump(exclude_none=True))
        calls = message.tool_calls or []

        if not calls:
            text = (message.content or "").strip()
            if not text:
                raise RuntimeError("The model returned an empty answer.")
            return text

        if round_number >= MAX_TOOL_ROUNDS:
            raise RuntimeError("Tool-round limit reached.")

        # This small agent permits one validated call per round.
        if len(calls) != 1:
            raise ValueError("Expected one tool call in this round.")

        call = calls[0].function

        if call.name != "detect_motion":
            raise ValueError(f"Unregistered tool: {call.name}")

        MotionArguments.model_validate(call.arguments)

        print(f"[tool] {call.name}({call.arguments})", flush=True)

        try:
            result = hardware.detect_motion()
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}

        print(f"[result] {json.dumps(result)}", flush=True)

        messages.append({
            "role": "tool",
            "tool_name": call.name,
            "content": json.dumps(result),
        })

    raise RuntimeError("Agent loop ended without an answer.")


def main():
    client = ollama.Client(
        host="http://127.0.0.1:11434",
        timeout=180.0,
    )
    hardware = MotionTools()

    try:
        print("Allowing 60 seconds for the PIR to settle...", flush=True)
        time.sleep(60)
        print(f"Ready. Model: {MODEL}. Type quit to exit.", flush=True)

        while True:
            question = input("\nYou: ").strip()
            if question.lower() in {"quit", "exit"}:
                break
            if not question:
                continue

            started = time.perf_counter()
            try:
                answer = answer_question(client, hardware, question)
                print(f"Agent: {answer}")
            except Exception as exc:
                print(f"Request failed: {exc}")

            print(f"[elapsed] {time.perf_counter() - started:.2f} s")

    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    finally:
        hardware.close()


if __name__ == "__main__":
    main()