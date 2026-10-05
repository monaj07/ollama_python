#!/usr/bin/env python3
"""Usage: python3 ollama_benchmark.py MODEL RUNS [SPEED_PROMPT]."""
import argparse
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path


def equivalent(actual, expected):
    if isinstance(expected, str):
        def normalize(s):
            return " ".join(unicodedata.normalize("NFKD", s).casefold().split())
        return isinstance(actual, str) and normalize(actual) == normalize(expected)
    if type(expected) in (int, float):
        try:
            return type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9)
        except OverflowError:
            return False
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            equivalent(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            equivalent(a, b) for a, b in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def strict_json(text):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError(f"Non-JSON number: {value}")
    return json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_constant)


def grade(row, case):
    flags = {"passed": False, "correct": False, "valid_format": False,
             "completed": False, "abstained": False}
    if "error" in row:
        return {**flags, "status": "request_error"}
    valid = correct = abstained = False
    try:
        text = row["content"].strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
        value = strict_json(fenced[1] if fenced else text)
        valid = isinstance(value, dict) and set(value) == {"answer"}
        valid = valid and fenced is None
        if isinstance(value, dict) and "answer" in value:
            correct = any(equivalent(value["answer"], gold)
                          for gold in [case["expected"], *case.get("aliases", [])])
            abstained = any(equivalent(value["answer"], x) for x in ("UNKNOWN", "NONE"))
    except (ValueError, TypeError):
        pass
    complete = bool(row["done"] and row["stop_reason"] == "stop" and row["content"].strip())
    status = ("truncated" if row["stop_reason"] == "length" else
              "empty_answer" if not row["content"].strip() else "incomplete" if not complete else
              "invalid_format" if not valid else "wrong_answer" if not correct else "pass")
    return {"passed": bool(correct and complete and valid), "correct": bool(correct and complete),
            "valid_format": valid, "completed": complete, "abstained": abstained, "status": status}


def summarize_quality(rows):
    n = len(rows)
    completed = [r for r in rows if r["completed"]]
    percentage = lambda key: round(100 * sum(r[key] for r in rows) / n, 1) if n else None
    return {"attempts": n, "passed": sum(r["passed"] for r in rows),
            "accuracy_percent": percentage("passed"),
            "answer_accuracy_percent": percentage("correct"),
            "format_valid_percent": percentage("valid_format"),
            "completed_answers": len(completed),
            "false_abstentions": sum(r["abstained"] and not r["requires_abstention"] for r in rows),
            "status_counts": {s: sum(r["status"] == s for r in rows)
                              for s in sorted({r["status"] for r in rows})},
            "median_response_s": statistics.median(r["wall_time_s"] for r in completed) if completed else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("runs", type=int, help="number of warm speed runs; one initial loading run is added")
    parser.add_argument("--quality-runs", type=int, default=1, help="repetitions per quality case (default: 1)")
    parser.add_argument("prompt", nargs="?", help="override the speed prompt only")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("benchmark_cases.json"))
    parser.add_argument("--think", choices=["auto", "off", "on"], default="auto")
    parser.add_argument("--tokens", type=int, default=512, help="quality token budget, including thinking")
    parser.add_argument("--speed-tokens", type=int, default=128)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--transport", choices=["sdk", "curl"], default="sdk")
    args = parser.parse_args()
    if min(args.runs, args.quality_runs, args.tokens, args.speed_tokens) < 1:
        parser.error("runs and token budgets must be positive")
    dataset_bytes = args.dataset.read_bytes()
    data = json.loads(dataset_bytes)
    cases = data["cases"]
    for case in cases:
        if not all(key in case for key in ("id", "category", "prompt", "expected")):
            parser.error("each case needs id, category, prompt and expected")
    if not cases or len({c["id"] for c in cases}) != len(cases):
        parser.error("dataset cases must be nonempty with unique IDs")
    client = None
    if args.transport == "sdk":
        import ollama
        client = ollama.Client(host=args.host, timeout=900)

    def request(endpoint, payload):
        if client:
            if endpoint in ("show", "version"):
                # Read raw metadata: SDK versions may omit fields/endpoints.
                import httpx
                response = httpx.request("GET" if endpoint == "version" else "POST",
                    f"{args.host.rstrip('/')}/api/{endpoint}",
                    json=None if endpoint == "version" else payload, timeout=900)
                response.raise_for_status()
                return response.json()
            return getattr(client, endpoint)(**payload).model_dump(mode="json")
        result = subprocess.run([
            "curl", "--fail-with-body", "-sS", "--max-time", "900",
            "-H", "Content-Type: application/json", "-X", "GET" if endpoint == "version" else "POST",
            *([] if endpoint == "version" else ["--data-binary", "@-"]),
            f"{args.host.rstrip('/')}/api/{endpoint}"], input=json.dumps(payload),
            text=True, capture_output=True, timeout=910)
        if result.returncode:
            raise RuntimeError(result.stdout.strip() or result.stderr.strip())
        output = json.loads(result.stdout)
        if output.get("error"):
            raise RuntimeError(output["error"])
        return output

    info = request("show", {"model": args.model})
    supported = (info.get("thinking") or {}).get("values", [])
    if supported and args.think != "auto" and not any(
            v is (args.think == "on") for v in supported):
        parser.error(f"--think {args.think} is unsupported; server advertises {supported}")
    thinking = False if args.think == "off" else True if args.think == "on" else (
        False if False in supported else None)
    options = {"num_ctx": 2048, "temperature": 0}
    server_version = request("version", {}).get("version")
    request("generate", {"model": args.model, "keep_alive": 0, "stream": False})

    def generate(messages, limit):
        payload = {"model": args.model, "messages": messages, "stream": False,
                   "keep_alive": "10m", "options": {**options, "num_predict": limit}}
        if thinking is not None:
            payload["think"] = thinking
        start = time.perf_counter()
        try:
            output = request("chat", payload)
            message = output.get("message") or {}
            raw = message.get("content") or ""
            thoughts = message.get("thinking") or ""
            content = raw
            # Older runners may put tagged thinking in content instead.
            if "<think>" in content:
                thoughts += "\n".join(re.findall(r"<think>(.*?)</think>", content, re.S))
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
                if "<think>" in content:
                    content, unfinished = content.split("<think>", 1)
                    thoughts += unfinished
            seconds = (output.get("eval_duration") or 0) / 1e9
            count = output.get("eval_count") or 0
            row = {"content": content.strip(), "raw_content": raw, "thinking": thoughts,
                   "done": bool(output.get("done")), "stop_reason": output.get("done_reason"),
                   "total_time_s": (output.get("total_duration") or 0) / 1e9,
                   "load_time_s": (output.get("load_duration") or 0) / 1e9,
                   "generation_time_s": seconds, "generated_tokens": count,
                   "tokens_per_second": count / seconds if seconds > 0 else None}
        except Exception as exc:
            row = {"error": str(exc)}
        return {**row, "wall_time_s": time.perf_counter() - start}

    speed = []
    for run in range(args.runs + 1):
        print(f"Speed {run}/{args.runs} ({'cold' if run == 0 else 'warm'}): {args.model}", file=sys.stderr)
        row = generate([{"role": "user", "content": args.prompt or data["speed_prompt"]}], args.speed_tokens)
        speed.append({"run": run, "cold": run == 0, **row})
    warm = [r for r in speed[1:] if "error" not in r and r["done"] and
            r["stop_reason"] in ("stop", "length") and r["generation_time_s"] > 0 and r["generated_tokens"] > 0]
    quality = []
    for run in range(1, args.quality_runs + 1):
        ordered = list(cases)
        random.Random(42 + run).shuffle(ordered)
        for case in ordered:
            print(f"Quality {run}/{args.quality_runs}: {case['id']}", file=sys.stderr)
            messages = [{"role": "system", "content": data["system"]},
                        *case.get("history", []), {"role": "user", "content": case["prompt"]}]
            row = generate(messages, args.tokens)
            quality.append({"run": run, "id": case["id"], "category": case["category"],
                            "requires_abstention": case.get("requires_abstention", False),
                            "expected": case["expected"], **row, **grade(row, case)})
    report = {
        "model": args.model, "ollama_version": server_version, "model_details": info.get("details"),
        "dataset": data["name"], "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "settings": {"transport": args.transport, "think_requested": args.think,
                     "think_sent": thinking, "supported_thinking": supported,
                     **options, "quality_tokens": args.tokens, "speed_tokens": args.speed_tokens,
                     "speed_runs": args.runs, "quality_runs": args.quality_runs, "order_seed": 42},
        "speed_summary": {
            "warm_runs_count": len(warm),
            "invalid_warm_runs": args.runs - len(warm),
            "tokens_per_second": (sum(r["generated_tokens"] for r in warm) /
                                  sum(r["generation_time_s"] for r in warm)) if warm else None,
            "median_wall_time_s": statistics.median(r["wall_time_s"] for r in warm) if warm else None,
            "min_tokens_per_second": min(r["tokens_per_second"] for r in warm) if warm else None,
            "max_tokens_per_second": max(r["tokens_per_second"] for r in warm) if warm else None,
            "error_count": sum("error" in r for r in speed)},
        "quality_summary": {"unique_cases": len(cases), **summarize_quality(quality)},
        "by_category": {c: summarize_quality([r for r in quality if r["category"] == c])
                        for c in sorted({r["category"] for r in quality})},
        "speed_runs": speed, "quality_runs": quality}
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        sys.exit(1)
