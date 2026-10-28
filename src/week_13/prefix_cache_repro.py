#!/usr/bin/env python3
"""Reproduce four prefix-cache experiments with the Ollama chat API.

Runs locally. No sensors, SQLite, or third-party Python packages required.
The exact cache counts are implementation-dependent; see README.md.
"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M"


def request(base_url, endpoint, payload):
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Ollama HTTP {error.code}: {error.read()[:300]!r}") from error
    if "error" in result:
        raise RuntimeError(f"Ollama: {result['error']}")
    return result


def options(num_predict):
    return {
        "num_ctx": 2048,
        "num_thread": 2,
        "num_predict": num_predict,
        "temperature": 0,
        "seed": 42,
    }


def prepare(base_url, model, params):
    """Unload old runner state, then warm weights with empty prompt."""
    request(base_url, "generate", {"model": model, "keep_alive": 0})
    request(base_url, "generate", {
        "model": model, "prompt": "", "options": params, "keep_alive": "10m"
    })


def chat(base_url, model, messages, params):
    return request(base_url, "chat", {
        "model": model,
        "messages": messages,
        "options": params,
        "think": False,
        "stream": False,
        "keep_alive": "10m",
    })


def make_rules(count, phrase):
    return "\n".join(f"Rule {i}: {phrase}" for i in range(1, count + 1))


def different_suffix_cases(rule_count):
    rules = make_rules(
        rule_count,
        "Ground responses in observations; distinguish unavailable sensors "
        "from negative detections.",
    )
    prefix_a = "You are a Raspberry Pi environment agent.\n" + rules
    prefix_b = "You are an embedded AI research assistant.\n" + rules
    items = [
        ("A: Initial", prefix_a, "Reply only READY."),
        ("B: Same prefix, different user message", prefix_a, "Reply only DONE."),
    ]
    if rule_count == 15:
        items.append(("C: Changed prefix", prefix_b, "Reply only OK."))
    return items


def metrics(experiment, label, response):
    prompt_count = response.get("prompt_eval_count", 0)
    cached_count = response.get("prompt_eval_cached_count", 0)
    return {
        "experiment": experiment,
        "case": label,
        "prompt_tokens": prompt_count,
        "cached_tokens": cached_count,
        "uncached_tokens": prompt_count - cached_count,
        "generated_tokens": response.get("eval_count", 0),
        "load_s": round(response.get("load_duration", 0) / 1e9, 3),
        "prefill_s": round(response.get("prompt_eval_duration", 0) / 1e9, 3),
        "decode_s": round(response.get("eval_duration", 0) / 1e9, 3),
        "response": response.get("message", {}).get("content", ""),
    }


def run(experiment, base_url, model):
    params = options(48 if experiment == "continuation" else 16)
    prepare(base_url, model, params)
    rows = []
    if experiment in ("changed-short", "changed-long"):
        count = 15 if experiment == "changed-short" else 64
        for label, system, question in different_suffix_cases(count):
            reply = chat(base_url, model, [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ], params)
            rows.append(metrics(experiment, label, reply))
    elif experiment == "identical":
        rules = make_rules(
            15,
            "Ground responses in observations; distinguish unavailable sensors "
            "from negative detections.",
        )
        messages = [
            {"role": "system", "content": rules},
            {"role": "user", "content": "Reply with READY."},
        ]
        for label in ("Run 1", "Run 2"):
            rows.append(metrics(experiment, label, chat(base_url, model, messages, params)))
    elif experiment == "continuation":
        rules = make_rules(
            15,
            "Report actual sensor status; never interpret sensor failure as absence.",
        )
        messages = [
            {"role": "system", "content": rules},
            {"role": "user", "content": "What does a PIR sensor timeout mean?"},
        ]
        for index in range(2):
            reply = chat(base_url, model, messages, params)
            rows.append(metrics(experiment, f"Turn {index + 1}", reply))
            messages.append(reply["message"])
            if index == 0:
                messages.append({
                    "role": "user",
                    "content": "And what if the camera is also unavailable?",
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("changed-short", "changed-long", "identical", "continuation", "all"),
        default="all",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    experiments = ("changed-short", "identical", "continuation", "changed-long") \
        if args.experiment == "all" else (args.experiment,)
    all_rows = []
    for experiment in experiments:
        print(f"\n=== {experiment} ===", flush=True)
        rows = run(experiment, args.base_url, args.model)
        all_rows.extend(rows)
        for row in rows:
            print(
                f"{row['case']}: prompt={row['prompt_tokens']}, "
                f"cached={row['cached_tokens']}, "
                f"uncached={row['uncached_tokens']}, "
                f"prefill={row['prefill_s']:.3f}s, "
                f"load={row['load_s']:.3f}s; "
                f"reply={row['response'][:85]!r}",
                flush=True,
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "model": args.model,
            "experiments": list(experiments),
            "results": all_rows,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
