"""Button-triggered local inference, validated LED action, and JSONL tracing."""
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from led_tools import LEDController
from model_decision_test import MODEL, OLLAMA_HOST, request_decision, validate_decision

TRACE_PATH = Path(__file__).with_name("ai_button_events.jsonl")


def write_event(event):
    with TRACE_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def main():
    import ollama
    from gpiozero import Button, LED

    client = ollama.Client(host=OLLAMA_HOST, timeout=120.0)
    with (
        LED(17, initial_value=False) as green,
        LED(27, initial_value=False) as red,
        Button(22, pull_up=True, bounce_time=0.05) as button,
    ):
        controller = LEDController(green, red)
        print(f"Model: {MODEL}")
        print(f"Trace: {TRACE_PATH}")
        print("Type an instruction, then press and release the button.")
        print("Type quit or press Ctrl+C to exit.")
        try:
            while True:
                instruction = input("\nInstruction: ").strip()
                if instruction.lower() in {"quit", "exit"}:
                    break
                if not instruction:
                    continue
                print("Ready — press and release the button.")
                button.wait_for_release()
                button.wait_for_press()
                pressed_at = perf_counter()
                controller.off()
                print("Triggered — release the button to start inference.")
                button.wait_for_release()
                event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": MODEL,
                    "instruction": instruction,
                }
                request_started = perf_counter()
                print("Thinking…")
                try:
                    response = request_decision(client, instruction)
                    event["model_message"] = response.message.model_dump()
                    for field in (
                        "load_duration", "prompt_eval_duration", "eval_duration",
                        "prompt_eval_count", "eval_count",
                    ):
                        event[field] = getattr(response, field, None)
                    args = validate_decision(response)
                    result = controller.execute("set_led", args.model_dump())
                    event.update(status="executed", result=result)
                    print(f"Executed: set_led(colour={args.colour!r})")
                except Exception as error:
                    controller.off()
                    event.update(
                        status="failed", error_type=type(error).__name__,
                        error=str(error), result={"commanded_state": "off"},
                    )
                    print(f"Failed: {error}")
                    print("Both LEDs switched off.")
                event["request_and_action_s"] = perf_counter() - request_started
                event["press_to_action_s"] = perf_counter() - pressed_at
                write_event(event)
                print(f"Request + action: {event['request_and_action_s']:.2f}s")
        except (KeyboardInterrupt, EOFError):
            print("\nStopped.")
        finally:
            controller.off()


if __name__ == "__main__":
    main()
