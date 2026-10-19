"""Week 10 final benchmark: routing accuracy and latency-log summary.

No GPIO, Ollama inference or camera access is performed by this script. The
--routes mode imports agent.route_question to evaluate real production routing.

Usage (repository root):
 PYTHONPATH=src/week_10 src/week_09/.venv/bin/python src/week_10/benchmark.py --routes
 python3 src/week_10/benchmark.py --log outputs/week_10/voice_run.log
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "routing_cases.json"


def load_cases(path: str | Path = FIXTURES) -> list[dict]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expected a nonempty list of routing cases")
    seen = set()
    required = {"id", "question", "expected_route"}
    for case in cases:
        if not isinstance(case, dict) or not required.issubset(case):
            raise ValueError("Malformed routing case")
        if not isinstance(case["id"], str) or case["id"] in seen:
            raise ValueError("Case IDs must be unique strings")
        seen.add(case["id"])
        for field in ("question", "expected_route", "last_topic", "last_question"):
            if field in case and not isinstance(case[field], str):
                raise ValueError(f"{field} must be text")
    return cases


def evaluate_routes(cases: list[dict], route_fn) -> dict:
    results = []
    for case in cases:
        observed = route_fn(
            case["question"],
            case.get("last_topic", "general"),
            case.get("last_question", ""),
        )
        expected = case["expected_route"]
        results.append({
            "id": case["id"],
            "question": case["question"],
            "last_topic": case.get("last_topic", "general"),
            "expected": expected,
            "observed": observed,
            "correct": observed == expected,
        })
    n = len(results)
    passed = sum(r["correct"] for r in results)
    return {
        "total": n,
        "correct": passed,
        "accuracy": round(passed / n, 4) if n else None,
        "misroutes": [r for r in results if not r["correct"]],
        "by_expected_route": {
            label: {
                "total": sum(r["expected"] == label for r in results),
                "correct": sum(r["expected"] == label and r["correct"] for r in results),
            }
            for label in sorted({r["expected"] for r in results})
        },
        "results": results,
        "limitations": (
            "Static first-stage routing only. 'live_followup' requires a separate "
            "Ollama classifier; this benchmark does not test its decision or any "
            "real sensor/SQL response."
        ),
    }


USER = re.compile(r"^\[user\]\s*(.*)")
ROUTE = re.compile(r"^\[route\]\s*(\S+)")
TOOL = re.compile(r"^\[(?:policy-tool|tool)\]\s*(\w+)")
TOTAL = re.compile(r"^\[total_s\]\s*([\d.]+)")


def parse_voice_log(text: str) -> dict:
    """Parse terminal traces; never reconstruct questions from a private DB."""
    turns = []
    turn = None
    for line in text.splitlines():
        user_match = USER.match(line.strip())
        if user_match:
            if turn is not None:
                turns.append(turn)
            turn = {"question": user_match.group(1), "route": None, "tools": [], "agent_s": None}
            continue
        if turn is None:
            continue
        route_match = ROUTE.match(line.strip())
        if route_match:
            turn["route"] = route_match.group(1)
        tool_match = TOOL.match(line.strip())
        if tool_match:
            turn["tools"].append(tool_match.group(1))
        total_match = TOTAL.match(line.strip())
        if total_match:
            turn["agent_s"] = float(total_match.group(1))
    if turn is not None:
        turns.append(turn)
    latencies = [t["agent_s"] for t in turns if t["agent_s"] is not None]
    def pct(values, p):
        if not values:
            return None
        ordered = sorted(values)
        # Nearest-rank percentile (well-defined even for small samples).
        idx = max(0, (len(ordered) * p + 99) // 100 - 1)
        return round(ordered[min(len(ordered) - 1, idx)], 3)
    return {
        "turn_count": len(turns),
        "routes": dict(Counter(t["route"] or "missing" for t in turns)),
        "physical_tool_calls": dict(Counter(
            name for t in turns for name in t["tools"]
            if name in ("detect_motion", "observe_scene")
        )),
        "agent_latency_s": {
            "n": len(latencies),
            "median": round(statistics.median(latencies), 3) if latencies else None,
            "p95_nearest_rank": pct(latencies, 95),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "turns": turns,
        "limitations": (
            "Extracts logged agent wall time, not end-to-end voice latency or model "
            "generation tokens. Missing or truncated logs cannot be verified. "
            "To evaluate tool correctness, compare per-turn tool lists with user intent."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--routes", action="store_true", help="Test production route_question()")
    mode.add_argument("--log", type=Path, help="Summarise a captured Week 10 terminal log")
    parser.add_argument("--cases", type=Path, default=FIXTURES)
    parser.add_argument("--output", type=Path, help="Write machine-readable JSON results")
    parser.add_argument("--strict", action="store_true", help="Exit 1 for any routing mismatch")
    args = parser.parse_args(argv)
    if args.routes:
        # Lazy import: --log works with stock Python and no Pi dependencies.
        from agent import route_question
        report = evaluate_routes(load_cases(args.cases), route_question)
        print(f"Routing: {report['correct']}/{report['total']} ({report['accuracy']:.1%})")
        for item in report["misroutes"]:
            print(f"  MISMATCH {item['id']}: expected={item['expected']} observed={item['observed']}: {item['question']}")
    else:
        report = parse_voice_log(args.log.read_text(encoding="utf-8"))
        print(f"Voice log: {report['turn_count']} turns; routes={report['routes']}; "
              f"agent latency={report['agent_latency_s']}")
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")
    if args.routes and args.strict and report["misroutes"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
