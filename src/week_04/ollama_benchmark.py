#!/usr/bin/env python3
"""Benchmark MODEL RUNS [SPEED_PROMPT], or --rescore REPORT.json offline."""
import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

SCORER_VERSION = "typed-answer-values-v2"


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
    text = row["content"].strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    candidate = fenced[1].strip() if fenced else text
    try:
        value = strict_json(candidate)
        valid = isinstance(value, dict) and set(value) == {"answer"}
        valid = valid and fenced is None
        candidate = value["answer"] if isinstance(value, dict) and "answer" in value else value
    except (ValueError, TypeError):
        pass
    correct = any(equivalent(candidate, gold) for gold in [case["expected"], *case.get("aliases", [])])
    abstained = any(equivalent(candidate, x) for x in ("UNKNOWN", "NONE"))
    complete = bool(row["done"] and row["stop_reason"] == "stop" and row["content"].strip())
    status = ("truncated" if row["stop_reason"] == "length" else
              "refusal" if row.get("refusal") else
              "empty_answer" if not row["content"].strip() else "incomplete" if not complete else
              "invalid_format" if not valid else "wrong_answer" if not correct else "pass")
    return {"passed": bool(correct and complete and valid), "correct": bool(correct and complete),
            "valid_format": valid, "completed": complete, "abstained": abstained, "status": status}


def summarize_quality(rows):
    n = len(rows)
    completed = [r for r in rows if r["completed"]]
    percentage = lambda key: round(100 * sum(r[key] for r in rows) / n, 1) if n else None
    return {"attempts": n, "passed": sum(r["passed"] for r in rows),
            "correct_answers": sum(r["correct"] for r in rows),
            "accuracy_percent": percentage("passed"),
            "answer_accuracy_percent": percentage("correct"),
            "format_valid_percent": percentage("valid_format"),
            "completed_answers": len(completed),
            "false_abstentions": sum(r["abstained"] and not r["requires_abstention"] for r in rows),
            "status_counts": {s: sum(r["status"] == s for r in rows)
                              for s in sorted({r["status"] for r in rows})},
            "median_response_s": statistics.median(r["wall_time_s"] for r in completed) if completed else None}


def quality_metrics(rows):
    return {"quality_summary": {"unique_cases": len({r["id"] for r in rows}), **summarize_quality(rows)},
            "by_category": {c: summarize_quality([r for r in rows if r["category"] == c])
                            for c in sorted({r["category"] for r in rows})}}


def rescore(path, cases, scoring_hash):
    report = json.loads(path.read_text())
    by_id = {c["id"]: c for c in cases}
    for row in report["quality_runs"]:
        case = by_id.get(row["id"])
        if case is None or not equivalent(row["expected"], case["expected"]):
            raise ValueError(f"Dataset gold answer changed or missing for {row['id']}; cannot rescore")
        row.update({"requires_abstention": case.get("requires_abstention", False), **grade(row, case)})
    report.update({"scorer_version": SCORER_VERSION, "rescored": True,
                   "scoring_dataset_sha256": scoring_hash, **quality_metrics(report["quality_runs"])})
    # Original dataset identity, responses, settings and timings remain intact.
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?")
    parser.add_argument("runs", nargs="?", type=int, help="number of warm speed runs; one initial loading run is added")
    parser.add_argument("--quality-runs", type=int, default=1, help="repetitions per quality case (default: 1)")
    parser.add_argument("prompt", nargs="?", help="override the speed prompt only")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("benchmark_cases.json"))
    parser.add_argument("--think", choices=["auto", "off", "on"], default="auto")
    parser.add_argument("--tokens", type=int, default=512, help="quality token budget, including thinking")
    parser.add_argument("--speed-tokens", type=int, default=128)
    parser.add_argument("--host", help="Ollama server URL or OpenAI API base URL")
    parser.add_argument("--transport", choices=["sdk", "curl"], default="sdk")
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--reasoning", choices=["none", "low", "medium", "high", "xhigh", "max"], default="none")
    parser.add_argument("--rescore", type=Path, help="rescore a saved report offline; no API/model calls")
    args = parser.parse_args()
    dataset_bytes = args.dataset.read_bytes()
    data = json.loads(dataset_bytes)
    scoring_hash = hashlib.sha256(dataset_bytes).hexdigest()
    cases = data["cases"]
    for case in cases:
        if not all(key in case for key in ("id", "category", "prompt", "expected")):
            parser.error("each case needs id, category, prompt and expected")
    if not cases or len({c["id"] for c in cases}) != len(cases):
        parser.error("dataset cases must be nonempty with unique IDs")
    if args.rescore:
        print(json.dumps(rescore(args.rescore, cases, scoring_hash), indent=2, ensure_ascii=False, allow_nan=False))
        return
    if args.model is None or args.runs is None:
        parser.error("MODEL and RUNS are required unless --rescore is used")
    args.host = args.host or ("http://localhost:11434" if args.provider == "ollama" else "https://api.openai.com/v1")
    if min(args.runs, args.quality_runs, args.tokens, args.speed_tokens) < 1:
        parser.error("runs and token budgets must be positive")
    if args.provider == "openai" and args.transport != "sdk":
        parser.error("Use the Python entry point with --provider openai (SDK transport)")
    if args.provider == "openai" and args.think != "auto":
        parser.error("For OpenAI use --reasoning none instead of --think off")
    if args.provider == "openai" and min(args.tokens, args.speed_tokens) < 16:
        parser.error("OpenAI token budgets must be at least 16")
    client = None
    if args.provider == "ollama" and args.transport == "sdk":
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

    supported, thinking, server_version = [], None, None
    if args.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            parser.error("Set OPENAI_API_KEY in this shell before running the OpenAI benchmark")
        from openai import OpenAI
        api = OpenAI(timeout=900, max_retries=0, base_url=args.host)
        info = {"details": api.models.retrieve(args.model).model_dump(mode="json")}
        options = {"num_ctx": None, "temperature": 0 if args.reasoning == "none" else None}
    else:
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
            if args.provider == "openai":
                parameters = {"model": args.model, "input": messages, "max_output_tokens": limit,
                              "reasoning": {"effort": args.reasoning}, "store": False, "stream": False}
                if args.reasoning == "none":
                    parameters["temperature"] = 0
                response = api.responses.create(**parameters)
                result = response.model_dump(mode="json")
                if result.get("error") or result.get("status") not in ("completed", "incomplete"):
                    raise RuntimeError(str(result.get("error") or result.get("status")))
                usage = result.get("usage") or {}
                reason = (result.get("incomplete_details") or {}).get("reason")
                refusal = "\n".join(c.get("refusal", "") for item in result.get("output", [])
                                    for c in (item.get("content") or []) if c.get("type") == "refusal")
                output = {"message": {"content": response.output_text or ""}, "done": True,
                          "done_reason": "refusal" if refusal else "length" if reason == "max_output_tokens"
                          else "stop" if result["status"] == "completed" else reason or "incomplete",
                          "eval_count": usage.get("output_tokens") or 0}
            else:
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
                   "total_time_s": (output.get("total_duration") or 0) / 1e9 if args.provider == "ollama" else None,
                   "load_time_s": (output.get("load_duration") or 0) / 1e9 if args.provider == "ollama" else None,
                   "generation_time_s": seconds if args.provider == "ollama" else None, "generated_tokens": count,
                   "tokens_per_second": count / seconds if seconds > 0 else None}
            if args.provider == "openai":
                row.update({"api_status": result["status"], "response_model": result.get("model"),
                            "usage": usage, "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
                            "refusal": refusal})
        except Exception as exc:
            row = {"error": str(exc)}
        wall = time.perf_counter() - start
        return {**row, "wall_time_s": wall, "end_to_end_tokens_per_second":
                row.get("generated_tokens", 0) / wall if "error" not in row and wall > 0 else None}

    speed = []
    for run in range(args.runs + 1):
        label = "cold" if args.provider == "ollama" else "initial API request"
        print(f"Speed {run}/{args.runs} ({label if run == 0 else 'repeat'}): {args.model}", file=sys.stderr)
        row = generate([{"role": "user", "content": args.prompt or data["speed_prompt"]}], args.speed_tokens)
        speed.append({"run": run, "cold": run == 0 if args.provider == "ollama" else None, **row})
    warm = [r for r in speed[1:] if "error" not in r and r["done"] and
            r["stop_reason"] in ("stop", "length") and r["wall_time_s"] > 0 and r["generated_tokens"] > 0 and
            (args.provider == "openai" or r["generation_time_s"] > 0)]
    engine = [r for r in warm if r["generation_time_s"] is not None]
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
        "model": args.model, "provider": args.provider, "ollama_version": server_version, "model_details": info.get("details"),
        "dataset": data["name"], "dataset_sha256": scoring_hash, "scorer_version": SCORER_VERSION,
        "settings": {"transport": args.transport, "think_requested": args.think if args.provider == "ollama" else None,
                     "think_sent": thinking, "supported_thinking": supported,
                     "reasoning_effort": args.reasoning if args.provider == "openai" else None,
                     **options, "quality_tokens": args.tokens, "speed_tokens": args.speed_tokens,
                     "speed_runs": args.runs, "quality_runs": args.quality_runs, "order_seed": 42},
        "speed_summary": {
            "warm_runs_count": len(warm),
            "invalid_warm_runs": args.runs - len(warm),
            "tokens_per_second": (sum(r["generated_tokens"] for r in engine) /
                                  sum(r["generation_time_s"] for r in engine)) if engine else None,
            "end_to_end_tokens_per_second": (sum(r["generated_tokens"] for r in warm) /
                                             sum(r["wall_time_s"] for r in warm)) if warm else None,
            "median_wall_time_s": statistics.median(r["wall_time_s"] for r in warm) if warm else None,
            "min_tokens_per_second": min(r["tokens_per_second"] for r in engine) if engine else None,
            "max_tokens_per_second": max(r["tokens_per_second"] for r in engine) if engine else None,
            "error_count": sum("error" in r for r in speed)},
        **quality_metrics(quality),
        "speed_runs": speed, "quality_runs": quality}
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        sys.exit(1)
