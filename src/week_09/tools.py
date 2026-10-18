"""Strict tool dispatcher for Week 09. Executes earlier weeks in their own venvs."""
from __future__ import annotations

import atexit
import json
import math
import os
import select
import subprocess
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # repository root (src/week_09 -> repository)

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "detect_motion",
            "description": (
                "Read the PIR motion sensor's CURRENT output. It can indicate motion "
                "during its hardware hold time, but not room occupancy."
            ),
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "observe_scene",
            "description": (
                "Capture a fresh photo and run open-vocabulary YOLOE detection on selected "
                "target labels. Use for questions about visible people or objects. "
                "For broad 'what do you see' questions choose likely targets such as "
                "person, cup, bottle, chair, laptop, glasses, plant. "
                "This is NOT unconstrained image understanding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
                    "conf": {"type": "number", "description": "Confidence threshold, 0.1 to 0.8; default 0.2"},
                },
                "required": ["targets"],
                "additionalProperties": False,
            },
        },
    },
]


def validate_call(name: str, raw: Any) -> dict:
    """Return normalized validated kwargs; raise ValueError for untrusted input."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Arguments are not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("Tool arguments must be an object")
    if name == "detect_motion":
        if raw:
            raise ValueError("detect_motion takes no arguments")
        return {}
    if name != "observe_scene":
        raise ValueError(f"Unknown tool: {name}")
    if set(raw) - {"targets", "conf"}:
        raise ValueError("Unknown observe_scene arguments")
    targets = raw.get("targets")
    if not isinstance(targets, list) or not 1 <= len(targets) <= 8:
        raise ValueError("targets must contain 1–8 labels")
    if any(not isinstance(v, str) or not v.strip() or len(v) > 64 or
           any(ord(ch) < 32 for ch in v) for v in targets):
        raise ValueError("Each target must be nonempty text, max 64 characters")
    targets = [v.strip() for v in targets]
    if len({v.casefold() for v in targets}) != len(targets):
        raise ValueError("Duplicate target labels")
    conf = raw.get("conf", 0.2)
    if type(conf) not in (float, int) or not math.isfinite(conf) or not 0.1 <= conf <= 0.8:
        raise ValueError("conf must be a finite number between 0.1 and 0.8")
    return {"targets": targets, "conf": float(conf)}


def _run_bridge(python: Path, bridge: Path, payload: dict, timeout: int, extra_env: dict | None = None) -> dict:
    if not python.is_file():
        return {"ok": False, "error": f"Python environment is missing: {python}"}
    environment = os.environ.copy()
    environment.update(extra_env or {})
    try:
        process = subprocess.run(
            [str(python), str(bridge)],
            input=json.dumps(payload), text=True, capture_output=True,
            cwd=str(REPO), env=environment, timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Tool timed out after {timeout}s"}
    except OSError as exc:
        return {"ok": False, "error": f"Could not launch tool: {exc}"}
    if process.returncode != 0:
        return {"ok": False, "error": (process.stderr or process.stdout or "Tool failed")[-700:]}
    try:
        result = json.loads(process.stdout)
        if not isinstance(result, dict) or type(result.get("ok")) is not bool:
            raise ValueError("Missing ok status")
        return result
    except (json.JSONDecodeError, ValueError):
        return {"ok": False, "error": "Tool returned invalid JSON"}




class PersistentVisionWorker:
    """Single long-running Week 07–08 process; one request at a time.

    Binary, newline-delimited I/O avoids TextIOWrapper buffering interfering
    with select() timeouts. A broken worker is restarted on the next request.
    """

    MAX_REPLY_BYTES = 1_048_576

    def __init__(self, python: Path, bridge: Path, timeout: int = 180):
        self.python = python
        self.bridge = bridge
        self.timeout = timeout
        self.process: subprocess.Popen | None = None
        self._buffer = b""

    def _start(self) -> None:
        if not self.python.is_file():
            raise RuntimeError(f"Python environment is missing: {self.python}")
        self.process = subprocess.Popen(
            [str(self.python), str(self.bridge), "--persistent"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Do not pipe stderr without draining it; warnings could deadlock.
            stderr=None,
            cwd=str(REPO),
        )
        self._buffer = b""

    def close(self) -> None:
        proc = self.process
        self.process = None
        self._buffer = b""
        if proc is None:
            return
        if proc.stdin:
            try:
                proc.stdin.close()
            except OSError:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        if proc.stdout:
            proc.stdout.close()

    def request(self, payload: dict) -> dict:
        if self.process is None or self.process.poll() is not None:
            self.close()
            try:
                self._start()
            except (OSError, RuntimeError) as exc:
                return {"ok": False, "error": f"Could not launch vision worker: {exc}"}

        assert self.process is not None
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        try:
            self.process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            self.process.stdin.flush()
            deadline = time.monotonic() + self.timeout
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Vision worker timed out after {self.timeout}s")
                ready, _, _ = select.select([self.process.stdout], [], [], remaining)
                if not ready:
                    raise TimeoutError(f"Vision worker timed out after {self.timeout}s")
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("Vision worker exited without a response")
                self._buffer += chunk
                if len(self._buffer) > self.MAX_REPLY_BYTES:
                    raise RuntimeError("Vision worker response exceeded size limit")
            line, self._buffer = self._buffer.split(b"\n", 1)
            result = json.loads(line)
            if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                raise ValueError("Vision worker response missing ok status")
            return result
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            self.close()
            return {"ok": False, "error": f"Vision worker failure: {exc}"}


class PhysicalTools:
    def __init__(self):
        self._vision_worker: PersistentVisionWorker | None = None
        atexit.register(self.close)

    def close(self) -> None:
        if self._vision_worker is not None:
            self._vision_worker.close()
            self._vision_worker = None

    def execute(self, name: str, args: dict) -> dict:
        """Never run LLM-produced shell commands. Only two allowed adapters."""
        # Validate again at execution boundary (defense-in-depth).
        args = validate_call(name, args)
        start = time.perf_counter()
        if name == "detect_motion":
            result = _run_bridge(
                REPO / "src/week_06/.venv/bin/python",
                HERE / "pir_bridge.py", args, timeout=20,
                extra_env={"GPIOZERO_PIN_FACTORY": "lgpio"},
            )
        elif name == "observe_scene":
            if self._vision_worker is None:
                self._vision_worker = PersistentVisionWorker(
                    REPO / "src/week_07_08/.venv-vision/bin/python",
                    HERE / "vision_bridge.py", timeout=180,
                )
            result = self._vision_worker.request(args)
        else:
            raise ValueError(f"Unknown tool: {name}")
        result["tool_elapsed_s"] = round(time.perf_counter() - start, 3)
        return result
