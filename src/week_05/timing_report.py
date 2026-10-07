"""Summarise successful JSONL traces without Ollama or GPIO dependencies."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def report(path, last=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    successful = [row for row in rows if row.get("status") == "executed"]
    if last is not None:
        successful = successful[-last:]
    groups = defaultdict(list)
    for row in successful:
        groups[row["model"]].append(row)
    if not groups:
        print("No successful requests in this selection.")
        return
    print("| Model | N | Request + action (s) | Load (s) | Prompt (s) | Generation (s) | Generation tok/s |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for model, events in groups.items():
        values = []
        for key, scale in (
            ("request_and_action_s", 1), ("load_duration", 1e9),
            ("prompt_eval_duration", 1e9), ("eval_duration", 1e9),
        ):
            samples = [e[key] / scale for e in events if isinstance(e.get(key), (int, float))]
            # Do not silently average a partially missing metric across a different cohort.
            values.append(f"{mean(samples):.3f}" if len(samples) == len(events) else "n/a")
        tokens = [e for e in events if isinstance(e.get("eval_count"), (int, float))
                  and isinstance(e.get("eval_duration"), (int, float)) and e["eval_duration"] > 0]
        speed = (sum(e["eval_count"] for e in tokens) / (sum(e["eval_duration"] for e in tokens) / 1e9)
                 if len(tokens) == len(events) else None)
        speed_text = f"{speed:.2f}" if speed is not None else "n/a"
        print(f"| {model} | {len(events)} | {' | '.join(values)} | {speed_text} |")
    print("\nMeans describe only the selected successful requests; no correctness score is inferred.")
    print("For a before/after comparison, use separate trace copies or inspect each request.")
    for row in successful:
        hold = (row.get("press_to_action_s", 0) - row["request_and_action_s"]
                if isinstance(row.get("press_to_action_s"), (int, float)) else None)
        hold_text = f", press/release overhead {hold:.3f}s" if hold is not None else ""
        print(f"{row.get('timestamp', '?')} {row['model']}: {row.get('instruction', '')!r} "
              f"→ {row['request_and_action_s']:.3f}s{hold_text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path,
                        default=Path(__file__).with_name("ai_button_events.jsonl"))
    parser.add_argument("--last", type=int, help="Select the last N successful requests across all models")
    args = parser.parse_args()
    if args.last is not None and args.last < 1:
        parser.error("--last must be positive")
    report(args.path, args.last)


if __name__ == "__main__":
    main()
