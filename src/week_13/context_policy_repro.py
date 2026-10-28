#!/usr/bin/env python3
"""Measure full/recent/summary context policies with a local Ollama model.

The 'summary' is manually constructed: this tests *using* a summary, not the
quality or cost of *generating* one. No hardware or SQLite records are used.
"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M"
SECRET = "VEX-7319-LUMA"  # Synthetic evaluation marker, not a credential.
OPTIONS = {
    "num_ctx": 2048,
    "num_thread": 2,
    "num_predict": 48,
    "temperature": 0,
    "seed": 42,
}


def post(base_url, endpoint, payload):
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Ollama HTTP {error.code}: {error.read()[:300]!r}") from error
    if "error" in data:
        raise RuntimeError(f"Ollama: {data['error']}")
    return data


def build_history():
    history = [
        {"role": "user", "content": f"The maintenance code is {SECRET}."},
        {"role": "assistant", "content": "Recorded."},
    ]
    for i in range(1, 11):
        history.extend([
            {
                "role": "user",
                "content": (
                    f"Status update {i}: Pump operating normally. "
                    "Temperature stable, pressure stable, no alarms. "
                    "The monitoring system reports normal operation "
                    "and no immediate maintenance is required."
                ),
            },
            {"role": "assistant", "content": "Noted."},
        ])
    return history


def build_context(strategy, system, history, ready, first_reply, question):
    tail = history[-4:]
    if strategy == "full":
        return [system] + history + [ready, first_reply, question]
    if strategy == "recent":
        return [system] + tail + [ready, first_reply, question]
    if strategy == "summary":
        summary = [
            {
                "role": "user",
                "content": f"Earlier conversation summary: Maintenance code = {SECRET}.",
            },
            {"role": "assistant", "content": "Understood."},
        ]
        return [system] + summary + tail + [ready, first_reply, question]
    raise ValueError(f"Unknown context strategy: {strategy}")


def chat(base_url, model, messages):
    return post(base_url, "chat", {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": OPTIONS,
        "keep_alive": "10m",
    })


def run(strategy, base_url, model):
    system = {
        "role": "system",
        "content": (
            "You are a maintenance assistant. Answer using only facts "
            "in the supplied conversation. If a fact is missing, "
            "reply UNKNOWN. Never invent equipment codes."
        ),
    }
    history = build_history()
    ready = {"role": "user", "content": "Reply ACK."}
    question = {
        "role": "user",
        "content": "What was the maintenance code? Reply with the code only, or UNKNOWN.",
    }

    # Reset context independently before each strategy, then preload weights.
    post(base_url, "generate", {"model": model, "keep_alive": 0})
    post(base_url, "generate", {
        "model": model, "prompt": "", "options": OPTIONS, "keep_alive": "10m"
    })
    initial = [system] + history + [ready]
    first = chat(base_url, model, initial)
    messages = build_context(strategy, system, history, ready, first["message"], question)
    result = chat(base_url, model, messages)

    prompt = result.get("prompt_eval_count", 0)
    cached = result.get("prompt_eval_cached_count", 0)
    answer = result.get("message", {}).get("content", "").strip()
    return {
        "strategy": strategy,
        "answer": answer,
        "correct": answer == SECRET,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "uncached_tokens": prompt - cached,
        "load_s": round(result.get("load_duration", 0) / 1e9, 3),
        "prefill_s": round(result.get("prompt_eval_duration", 0) / 1e9, 3),
        "decode_s": round(result.get("eval_duration", 0) / 1e9, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    results = []
    for strategy in ("full", "recent", "summary"):
        print(f"\n=== {strategy.upper()} ===", flush=True)
        row = run(strategy, args.base_url, args.model)
        results.append(row)
        for key, value in row.items():
            print(f"{key}: {value}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "model": args.model,
            "results": results,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
