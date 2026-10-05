#!/usr/bin/env python3
"""Compare saved reports: python3 compare_benchmarks.py REPORT.json ...

Keep beside the updated ollama_benchmark.py. No inference or SDK required.
Content is a relaxed diagnostic: accept numeric/boolean strings and an
unambiguous single-field wrapper. Strict and typed scores remain unchanged.
No correct-value substring search, arbitrary prose judging or schema repair.
Use --format csv or --format json for machine-readable output.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

from ollama_benchmark import SCORER_VERSION, equivalent, strict_json


def content_correct(row):
    if row.get("error") or not row.get("completed"):
        return False
    if row["correct"]:
        return True
    text = row["content"].strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    try:
        value = strict_json(fenced[1].strip() if fenced else text)
    except ValueError:
        value = text
    gold = row["expected"]
    if equivalent(value, gold):
        return True
    if isinstance(value, dict):
        if "answer" in value:
            value = value["answer"]
        elif len(value) == 1:
            value = next(iter(value.values()))
    if isinstance(value, str) and type(gold) in (int, float, bool):
        try:
            value = strict_json(value.strip().casefold())
        except ValueError:
            pass
    return equivalent(value, gold)


def compare(paths):
    reports = [json.loads(p.read_text()) for p in paths]
    if any(r.get("scorer_version") != SCORER_VERSION for r in reports):
        raise ValueError("Rescore all reports with the updated benchmark script first")
    signatures = {(r["dataset"], r["dataset_sha256"], r["settings"]["quality_tokens"],
                   r["settings"]["speed_tokens"], r["settings"]["quality_runs"]) for r in reports}
    if len(signatures) != 1:
        raise ValueError("Reports must use the same dataset hash, token budgets and quality repetitions")
    if len({(r.get("provider"), r["model"]) for r in reports}) != len(reports):
        raise ValueError("Supply one report per model/provider; exclude duplicate original/rescored files")
    result = []
    for path, r in zip(paths, reports):
        rows, q, s = r["quality_runs"], r["quality_summary"], r["speed_summary"]
        if not rows or len(rows) != q["attempts"]:
            raise ValueError(f"{path}: expected a full benchmark report, including every quality run")
        correct = sum(content_correct(row) for row in rows)
        result.append({"model": r["model"], "provider": r.get("provider", "ollama"),
                       "dataset": r["dataset"], "source": str(path), "attempts": len(rows),
                       "content_correct": correct,
                       "content_percent": round(100 * correct / len(rows), 1),
                       "typed_percent": q["answer_accuracy_percent"],
                       "strict_percent": q["accuracy_percent"],
                       "format_percent": q["format_valid_percent"],
                       "generation_tok_s": s["tokens_per_second"],
                       "end_to_end_tok_s": s.get("end_to_end_tokens_per_second"),
                       "median_quality_s": q["median_response_s"]})
    return sorted(result, key=lambda r: (-r["content_percent"], -r["strict_percent"], r["model"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--format", choices=["md", "csv", "json"], default="md")
    args = parser.parse_args()
    rows = compare(args.reports)
    if args.format == "json":
        print(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    else:
        print("| Model | Content* | Typed values | Strict passes | Generation tok/s | Median quality s |")
        print("|---|---:|---:|---:|---:|---:|")
        fmt = lambda value: "N/A" if value is None else f"{value:.2f}"
        for r in rows:
            print(f"| {r['model']} | {r['content_percent']:.1f}% | {r['typed_percent']:.1f}% | "
                  f"{r['strict_percent']:.1f}% | {fmt(r['generation_tok_s'])} | {fmt(r['median_quality_s'])} |")
        print("\n*Content ignores the specified type/wrapper mistakes; it is a diagnostic, not schema compliance.")
        print("Generation throughput is unavailable for APIs; compare response latency across providers.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        sys.exit(1)
