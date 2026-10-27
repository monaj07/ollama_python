#!/usr/bin/env python3
"""Week 12 offline (mock-tool) Ollama agent model comparison. Standard library only.

No GPIO, camera, audio, SQLite, or other real tools are invoked.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request

API = "http://127.0.0.1:11434/api"
MODELS = [
    "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M",
    "gemma4:e2b",
    "qwen3.5:2b",
    "qwen3.5:4b",
]
OPTIONS = {"num_ctx": 2048, "num_thread": 2, "num_predict": 256,
           "temperature": 0, "seed": 42}


def tool(name, description):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {}}}}


TOOLS = [
    tool("check_presence", "Check current human presence using both PIR and camera."),
    tool("observe_scene", "Capture a fresh image and identify visible objects."),
    tool("memory_recent", "Retrieve historical observations from saved records."),
]

ROUTING = [
    ("Check current presence using PIR and camera.", ["check_presence"]),
    ("What objects are visible through the camera?", ["observe_scene"]),
    ("Show the most recent saved observations.", ["memory_recent"]),
    ("Was anybody detected earlier today? Use stored records, not sensors.", ["memory_recent"]),
    ("Explain the difference between PIR and camera detection.", []),
]

QUESTION = ("Is anyone here right now? Also, was anyone detected in previous "
            "observations? Use live sensors for the first question and stored "
            "records for the second.")

SENSOR_RESULTS = {
    "normal": {"present": False, "pir_motion": False, "camera_person": False},
    "failure_raw": {"present": None, "pir_motion": None,
                    "camera_person": None,
                    "error": "PIR sensor timed out; camera is unavailable."},
    # New, explicitly labelled *proposed* contract. Not part of the historical runs.
    "failure_normalized": {
        "presence": "unknown",
        "pir": {"status": "timeout", "motion_detected": None},
        "camera": {"status": "unavailable", "person_detected": None},
    },
}
MEMORY_RESULT = {"observations": [
    {"person_detected": True, "when": "earlier today"}
]}


def post(endpoint, data):
    request = urllib.request.Request(
        f"{API}/{endpoint}", data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def seconds(value, field):
    return round(value.get(field, 0) / 1e9, 4)


def timing(response):
    return {
        "prompt_tokens": response.get("prompt_eval_count", 0),
        "generated_tokens": response.get("eval_count", 0),
        "cached_prompt_tokens": response.get("prompt_eval_cached_count", 0),
        "load_s": seconds(response, "load_duration"),
        "prefill_s": seconds(response, "prompt_eval_duration"),
        "decode_s": seconds(response, "eval_duration"),
        "total_s": seconds(response, "total_duration"),
    }


def chat(model, messages, num_predict=256):
    options = {**OPTIONS, "num_predict": num_predict}
    return post("chat", {"model": model, "messages": messages, "tools": TOOLS,
                         "stream": False, "think": False,
                         "options": options, "keep_alive": "10m"})


def preload(model):
    # Do not include model loading in warm-agent totals.
    post("generate", {"model": model, "keep_alive": 0})
    return post("generate", {"model": model, "prompt": "",
                             "options": OPTIONS, "keep_alive": "10m"})


def route_cases(model):
    rows = []
    for prompt, expected in ROUTING:
        response = chat(model, [{"role": "user", "content": prompt}],
                        num_predict=128)
        calls = [c["function"]["name"] for c in
                 response["message"].get("tool_calls", [])]
        row = {"prompt": prompt, "expected_tools": expected, "actual_tools": calls,
               "pass": calls == expected, "timing": timing(response)}
        rows.append(row)
        print(f"  {'PASS' if row['pass'] else 'FAIL'} "
              f"expected={expected} actual={calls} "
              f"total={row['timing']['total_s']:.2f}s", flush=True)
    return rows


def agent_case(model, scenario):
    results = {"check_presence": SENSOR_RESULTS[scenario],
               "memory_recent": MEMORY_RESULT}
    messages = [{"role": "user", "content": QUESTION}]
    calls, rounds, final_answer = [], [], None
    for _ in range(4):
        response = chat(model, messages)
        rounds.append(timing(response))
        message = response["message"]
        requested = message.get("tool_calls", [])
        messages.append(message)
        if not requested:
            final_answer = message.get("content", "")
            break
        for call in requested:
            name = call["function"]["name"]
            calls.append(name)
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(results.get(
                                 name, {"error": "Unknown or unavailable tool"}))})
    required_tools_ok = (len(calls) == 2 and
                         sorted(calls) == ["check_presence", "memory_recent"])
    return {"scenario": scenario, "tools": calls,
            "required_tools_ok": required_tools_ok,
            "answer": final_answer,
            "rounds": rounds,
            "api_total_s": round(sum(r["total_s"] for r in rounds), 4),
            # Deliberately no automated grounding score: prose needs review.
            "grounding_review": "manual_review_required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Exact names from `ollama list`")
    parser.add_argument("--stage", choices=["routing", "agent", "all"],
                        default="all")
    parser.add_argument("--scenario", choices=list(SENSOR_RESULTS),
                        default="failure_raw")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/week_12/benchmark.json"))
    args = parser.parse_args()
    records = []
    for model in args.models:
        print(f"\n=== {model} ===", flush=True)
        start = time.perf_counter()
        preload_response = preload(model)
        record = {"model": model, "preload": timing(preload_response),
                  "preload_wall_s": round(time.perf_counter()-start, 4),
                  "options": dict(OPTIONS)}
        try:
            if args.stage in ("routing", "all"):
                record["routing"] = route_cases(model)
                print("Routing:", sum(x["pass"] for x in record["routing"]),
                      "/", len(record["routing"]), flush=True)
            if args.stage in ("agent", "all"):
                record["agent"] = agent_case(model, args.scenario)
                case = record["agent"]
                print("Agent tools:", case["tools"],
                      "| required tools:", case["required_tools_ok"])
                print("Final answer:", case["answer"])
                print("Warm API total:", case["api_total_s"], "s")
        finally:
            post("generate", {"model": model, "keep_alive": 0})
        records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "kind": "week12_model_selection", "stage": args.stage,
        "scenario": args.scenario, "records": records,
        "note": "Only routing and required tool names are automatically graded; "
                "final-answer factuality requires independent review."
    }, indent=2) + "\n", encoding="utf-8")
    print("\nSaved", args.output)


if __name__ == "__main__":
    main()
