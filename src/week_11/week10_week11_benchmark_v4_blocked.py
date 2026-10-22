#!/usr/bin/env python3
"""Week 10 custom agent versus Week 11 Pi RPC, blocked warm-state benchmark.

Run from the monaj07/ollama_python repository root. Only Python standard library.
The harness invokes existing --text --no-play voice entry points and parses
emitted [timings] JSON, retaining all stdout/stderr for auditing. It cannot
fully equalise different agent prompt/tool policies or test multi-turn state.

Runs each backend consecutively after its own task-specific warmups, so switching
backends does not contaminate the measured warm results.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

CASES = {
    "arithmetic": "What is 2 + 17?",
    "positive_count": "How many person-positive observation events have been recorded?",
    "last_positive": "When did you last detect a person?",
    "presence": "Is there anyone in the room right now?",
}
STAGES = ("record_s", "stt_s", "llm_s", "tts_s", "play_s", "total_s")
COLUMNS = ["task", "backend", "repeat", "question", "answer", "status", "correct",
           "wall_s", *STAGES, "pi_tools", "pi_tool_s", "pi_tokens_in", "pi_tokens_out",
           "exit_code", "log_file"]


def parse_timings(output: str) -> dict:
    """Read the last complete [timings] JSON block, even if pretty-printed."""
    decoder = json.JSONDecoder()
    found = {}
    for m in re.finditer(r"\[timings\]\s*(?=\{)", output):
        try:
            block, _ = decoder.raw_decode(output[m.end():].lstrip())
            if isinstance(block, dict):
                found = block
        except ValueError:
            continue
    if found:
        return found
    # Compatibility with older logging using just [time] stage: X s.
    for stage in STAGES:
        matches = re.findall(rf"\[time\]\s*{stage}:\s*([0-9.]+)\s*s", output)
        if matches:
            found[stage] = float(matches[-1])
    return found


def parse_answer(output: str) -> str:
    for pattern in (r"^\[assistant\]\s*(.*)$", r"^Assistant:\s*(.*)$"):
        matches = re.findall(pattern, output, re.MULTILINE)
        if matches:
            return matches[-1].strip()
    return ""


def parse_pi_trace(output: str) -> tuple[str, str, str, str]:
    lines = [line.strip() for line in output.splitlines() if line.startswith("[pi_rpc]")]
    if not lines:
        return "", "", "", ""
    line = lines[-1]
    tools = re.search(r"\btools=(\[[^\]]*\])", line)
    seconds = re.search(r"\btool_s=(\{[^}]*\})", line)
    tokens_in = re.search(r"\binput_tokens=(\d+)", line)
    tokens_out = re.search(r"\boutput_tokens=(\d+)", line)
    return (tools.group(1) if tools else "", seconds.group(1) if seconds else "",
            tokens_in.group(1) if tokens_in else "", tokens_out.group(1) if tokens_out else "")


def ground_truth(db: Path) -> dict:
    if not db.is_file():
        return {"note": f"SQLite database missing: {db}"}
    try:
        with sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True) as con:
            count = con.execute("SELECT count(*) FROM observations WHERE person_detected = 1").fetchone()[0]
            last = con.execute(
                "SELECT MAX(observed_at_utc) FROM observations WHERE person_detected = 1"
            ).fetchone()[0]
        if last:
            last = datetime.fromisoformat(last.replace("Z", "+00:00")).astimezone(
                ZoneInfo("Australia/Melbourne")
            ).isoformat()
        return {"positive_count": count, "last_positive_local": last}
    except (sqlite3.Error, ValueError, KeyError) as exc:
        return {"note": f"Could not derive ground truth: {exc}"}


def correctness(task: str, answer: str, truth: dict) -> str:
    if not answer:
        return "ungraded"
    if task == "arithmetic":
        # Avoid marking '2 + 17 = 20' correct because it contains the input 17.
        return "pass" if bool(re.search(r"\b19\b", answer)) and not re.search(
            r"(?:\b20\b|\b21\b)", answer
        ) else "fail"
    if task == "positive_count" and "positive_count" in truth:
        expected = str(truth["positive_count"])
        return "pass" if re.search(rf"(?<!\d){re.escape(expected)}(?!\d)", answer) else "review"
    if task == "last_positive" and truth.get("last_positive_local"):
        # Time/date output varies; leave automatic grading to manual review.
        return "review"
    return "review"


def command_for(backend: str, case: str, args: argparse.Namespace, audio_dir: Path) -> list[str]:
    base = [str(args.python)]
    if backend == "week10":
        return [*base, "src/week_10/voice_assistant.py", "--model", args.ollama_model,
                "--text", case, "--no-play", "--output-dir", str(audio_dir / "week10")]
    return [*base, "src/week_11/voice_assistant.py", "--pi-model", args.pi_model,
            "--pi-system-prompt", str(args.prompt), "--pi-no-context-files",
            "--text", case, "--no-play", "--output-dir", str(audio_dir / "week11")]


def declared_cli_flags(source: Path) -> set[str]:
    """Inspect argparse add_argument literals without executing the wrapper.

    Week 11 uses parse_known_args() for --pi-* options and forwards --help to
    Week 09, so executing its --help both hides these options AND starts Pi RPC.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("--"):
                    flags.add(arg.value)
    return flags


def preflight(args: argparse.Namespace) -> tuple[bool, str]:
    """Validate the env and wrapper CLI *without starting a Pi session*."""
    try:
        probe = subprocess.run(
            [str(args.python), "-c",
             "import sys; print('prefix=' + sys.prefix); print('base_prefix=' + sys.base_prefix); print('exe=' + sys.executable)"],
            cwd=args.repo, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Cannot execute configured Python: {exc}"
    if probe.returncode:
        return False, f"Python probe failed:\n{probe.stdout}\n{probe.stderr}"
    print("Python executable:", args.python, flush=True)
    print("Python environment:\n" + probe.stdout.strip(), flush=True)
    lines = dict(line.split("=", 1) for line in probe.stdout.splitlines() if "=" in line)
    if lines.get("prefix") == lines.get("base_prefix"):
        return False, ("Benchmark interpreter is NOT running inside a virtual environment. "
                       "Use --python src/week_09/.venv/bin/python and keep the symlink path.")

    # The actual shared audio CLI is Week 09's.  This is the help the Week 11
    # wrapper delegates to; checking Week 11 --help directly is misleading.
    audio_file = args.repo / "src/week_09/voice_assistant.py"
    if not audio_file.is_file():
        return False, f"Missing shared audio implementation: {audio_file}"
    try:
        help_run = subprocess.run([str(args.python), str(audio_file), "--help"],
                                  cwd=args.repo, capture_output=True, text=True, timeout=40)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Week 09 audio --help failed: {exc}"
    help_text = help_run.stdout + "\n" + help_run.stderr
    if help_run.returncode:
        return False, f"Week 09 audio --help exited {help_run.returncode}:\n{help_text[-2500:]}"
    required_audio_flags = ("--model", "--text", "--no-play", "--output-dir")
    absent = [f for f in required_audio_flags if f not in help_text]
    if absent:
        return False, f"Shared Week 09 audio CLI missing required flags: {', '.join(absent)}"

    # Week 10 --help is expected to resolve to the Week 09 CLI; it is fine
    # to inspect this CLI provided it exits normally, without running an agent.
    week10 = args.repo / "src/week_10/voice_assistant.py"
    try:
        w10_help = subprocess.run([str(args.python), str(week10), "--help"],
                                  cwd=args.repo, capture_output=True, text=True, timeout=40)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Week 10 --help failed: {exc}"
    if w10_help.returncode:
        return False, (f"Week 10 --help exited {w10_help.returncode}:\n"
                       + (w10_help.stdout + w10_help.stderr)[-2500:])
    week10_help_text = w10_help.stdout + "\n" + w10_help.stderr
    absent_w10 = [f for f in required_audio_flags if f not in week10_help_text]
    if absent_w10:
        return False, f"Week 10 --help missing required flags: {', '.join(absent_w10)}"

    # Week 11 has a two-level parser: --pi-* flags are consumed by the outer
    # wrapper before delegating to Week 09.  NEVER run Week 11 --help in preflight;
    # that starts Pi RPC before showing only Week 09's help screen.
    week11 = args.repo / "src/week_11/voice_assistant.py"
    try:
        week11_flags = declared_cli_flags(week11)
        contents = week11.read_text(encoding="utf-8")
    except (OSError, SyntaxError) as exc:
        return False, f"Cannot inspect Week 11 wrapper: {exc}"
    required_pi_flags = ("--pi-model", "--pi-system-prompt", "--pi-no-context-files")
    absent_w11 = [f for f in required_pi_flags if f not in week11_flags]
    if absent_w11:
        return False, ("Week 11 wrapper source does not declare: " + ", ".join(absent_w11)
                       + ". Inspect its argument parser before benchmarking.")
    if "parse_known_args" not in contents:
        return False, ("Week 11 no longer appears to use the expected two-level parser; "
                       "please inspect the updated wrapper before benchmarking.")
    print("CLI preflight: Week 09 shared audio flags verified; Week 10 --help verified; "
          "Week 11 --pi-* flags verified in source (without starting RPC).", flush=True)
    return True, ""


def run_one(task: str, backend: str, repeat: int, args: argparse.Namespace,
            out_dir: Path, truth: dict, *, warmup: bool = False) -> dict:
    question = CASES[task]
    command = command_for(backend, question, args, out_dir / "audio")
    suffix = f"warmup{repeat:02d}" if warmup else f"r{repeat:02d}"
    log = out_dir / "logs" / f"{task}_{backend}_{suffix}.log"
    start = time.perf_counter()
    code = -1
    status = "ok"
    try:
        result = subprocess.run(command, cwd=args.repo, text=True, errors="replace",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=args.timeout_s, check=False)
        output, code = result.stdout, result.returncode
        if code:
            status = f"exit={code}"
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout or b""
        output = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        output += f"\n[TIMEOUT] exceeded {args.timeout_s} s\n"
        status = "timeout"
    elapsed = round(time.perf_counter() - start, 3)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("COMMAND: " + " ".join(command) + "\n" + output, encoding="utf-8")
    if status != "ok":
        print(f"ERROR in {backend} / {task} (exit code {code}):", flush=True)
        print("\n".join(output.strip().splitlines()[-22:]) or "(no process output)", flush=True)
        print(f"Full failure log: {log}", flush=True)
    timings = parse_timings(output)
    answer = parse_answer(output)
    tools, tool_s, tokens_in, tokens_out = parse_pi_trace(output)
    row = dict(task=task, backend=backend, repeat=repeat, question=question, answer=answer,
               status=status if timings else ("missing_timings" if status == "ok" else status),
               correct=correctness(task, answer, truth) if status == "ok" else "ungraded",
               wall_s=elapsed, **{key: timings.get(key, "") for key in STAGES},
               pi_tools=tools, pi_tool_s=tool_s, pi_tokens_in=tokens_in,
               pi_tokens_out=tokens_out, exit_code=code, log_file=str(log))
    print(f"{'warmup' if warmup else 'measure'} {task:16} {backend:6} "
          f"llm={str(row['llm_s']):>8} total={str(row['total_s']):>8} "
          f"wall={elapsed:>8}s check={row['correct']:>8} status={row['status']}", flush=True)
    return row


def median(rows: list[dict], key: str) -> str:
    values = [float(row[key]) for row in rows if row["status"] == "ok" and
              isinstance(row.get(key), (int, float))]
    return f"{statistics.median(values):.3f}" if values else "—"


def summary_markdown(rows: list[dict], truth: dict, tasks: list[str]) -> str:
    lines = ["# Week 10 vs Week 11 — controlled text-input benchmark", "",
             "Both agents used the Week 09 audio environment with `--text --no-play`.",
             "Microphone capture and speaker playback are excluded; Piper synthesis remains included.",
             "Each run starts a new wrapper process; `llm_s` excludes wrapper startup, while `wall_s` includes it.",
             "For each task and backend, warm-up requests precede consecutive measured requests (blocked order).",
             "Pi and Week 10 use different system prompts/orchestration policies; this is an application comparison,",
             "not a pure runtime A/B benchmark. No Telegram process should run during measurement.", "",
             "## Expected SQLite facts (at benchmark start)", "",
             "```json", json.dumps(truth, indent=2), "```", "",
             "## Medians (seconds; measured runs only)", "",
             "| Task | Agent | Runs | LLM+tools | TTS | Total (no playback) | Wall | Auto check |",
             "|---|---|---:|---:|---:|---:|---:|---|" ]
    for task in tasks:
        for backend in ("week10", "week11"):
            selection = [r for r in rows if r["task"] == task and r["backend"] == backend]
            passed = sum(r["correct"] == "pass" for r in selection)
            flagged = sum(r["correct"] == "fail" for r in selection)
            lines.append(f"| {task} | {backend} | {len(selection)} | {median(selection, 'llm_s')} | "
                         f"{median(selection, 'tts_s')} | {median(selection, 'total_s')} | "
                         f"{median(selection, 'wall_s')} | {passed} pass, {flagged} fail |")
    lines += ["", "## Interpretation", "", "- Review each log's tool calls and factual grounding; automatic scoring is intentionally limited.",
              "- Presence observations change over time. Do not compare their truth labels as if a fixed scene were used.",
              "- To compare conversation follow-ups and full voice, use a **persistent** Week 10 / Week 11 session",
              "  separately; this one-shot benchmark measures independent turns.",
              "- Warm/cold behavior is not fully isolated here: the first model load, tool-worker initialization,",
              "  thermal load and prompt caching can change performance.",
              "- Use the raw logs for any failures or suspicious latency differences.", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--python", type=Path, default=Path("src/week_09/.venv/bin/python"))
    ap.add_argument("--ollama-model", default="gemma4:e2b-8k")
    ap.add_argument("--pi-model", default="ollama/gemma4:e2b-8k")
    ap.add_argument("--prompt", type=Path, default=Path("src/week_11/voice_system.md"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--tasks", default="arithmetic,positive_count,last_positive")
    ap.add_argument("--timeout-s", type=float, default=240)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.repo = args.repo.resolve()
    # CRITICAL: Do not resolve the final Python symlink.  A venv usually points
    # .venv/bin/python -> /usr/bin/python3.13, and executing the resolved binary
    # loses the venv's sys.prefix/site-packages.
    args.python = (args.repo / args.python).absolute() if not args.python.is_absolute() else args.python
    args.prompt = (args.repo / args.prompt).resolve() if not args.prompt.is_absolute() else args.prompt
    tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    if not tasks or any(t not in CASES for t in tasks):
        ap.error(f"--tasks must contain names from {', '.join(CASES)}")
    if args.runs < 1 or args.warmups < 0 or args.timeout_s <= 0:
        ap.error("--runs and --timeout-s must be positive, --warmups must be nonnegative")
    required = [args.python, args.prompt, args.repo / "src/week_10/voice_assistant.py",
                args.repo / "src/week_11/voice_assistant.py"]
    if not args.dry_run:
        for path in required:
            if not path.is_file():
                ap.error(f"Required file not found: {path}")
    out_dir = args.output_dir or (args.repo / "outputs/week_11" /
                      ("benchmark_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    out_dir = out_dir.resolve()
    print("Repository:", args.repo)
    print("Questions:", ", ".join(tasks))
    print("Week 10:", " ".join(command_for("week10", CASES[tasks[0]], args, out_dir / "audio")))
    print("Week 11:", " ".join(command_for("week11", CASES[tasks[0]], args, out_dir / "audio")))
    if args.dry_run:
        return 0
    ok, reason = preflight(args)
    if not ok:
        print("\n[PREFLIGHT FAILED] " + reason, file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = ground_truth(args.repo / "outputs/week_10/events.sqlite3")
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print("SQLite ground truth:", json.dumps(truth))
    print("NOTE: stop Pi TUI, Telegram bot and background camera monitor before running.")
    print("BENCHMARK ORDER: block each backend per task; warm up immediately before measurements.")
    # IMPORTANT: interleaving Week 10 and Week 11 caused repeated cold-state
    # transitions on a memory-constrained Pi. Warm each backend immediately
    # before measuring consecutive requests with the same task and backend.
    rows: list[dict] = []
    warmup_rows: list[dict] = []
    for task_index, task in enumerate(tasks):
        # Alternate which backend gets the first block between TASKS, not RUNS.
        # Keep each backend's warm-ups adjacent to its measurements.
        backend_order = ("week10", "week11") if task_index % 2 == 0 else ("week11", "week10")
        for backend in backend_order:
            print(f"\n[BLOCK] task={task} backend={backend} warmups={args.warmups} "
                  f"measurements={args.runs}", flush=True)
            for i in range(1, args.warmups + 1):
                warm = run_one(task, backend, i, args, out_dir, truth, warmup=True)
                warmup_rows.append(warm)
                if warm["status"] != "ok":
                    print("[ABORTED] Warm-up failed. Subsequent measurements are invalid.",
                          file=sys.stderr)
                    return 1
            for i in range(1, args.runs + 1):
                row = run_one(task, backend, i, args, out_dir, truth)
                rows.append(row)
                with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                if row["status"] != "ok":
                    print("[ABORTED] Latest measurement failed; not continuing with invalid data.",
                          file=sys.stderr)
                    return 1
    with (out_dir / "warmups.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(warmup_rows)
    (out_dir / "summary.md").write_text(summary_markdown(rows, truth, tasks), encoding="utf-8")
    print("Results:", out_dir / "results.csv")
    print("Warmup details:", out_dir / "warmups.csv")
    print("Summary:", out_dir / "summary.md")
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
