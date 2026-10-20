"""Week 11 persistent, JSON-lines adapter to validated Week 09 hardware tools.

Run with the Pi host's ordinary Python 3.10+ interpreter. PhysicalTools launches
Week 06 and Week 07–08 in their existing specialized environments.
Protocol: {"tool": "health"|"detect_motion"|"observe_scene", "args": {...}}
          -> {"ok": true|false, ...} (one line per request)
"""
from __future__ import annotations

import contextlib
import json
import signal
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.week_09.tools import PhysicalTools  # noqa: E402


def main() -> None:
    # Graceful shutdown closes the nested persistent vision worker.
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))
    tools = PhysicalTools()
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object")
                tool = request.get("tool")
                args = request.get("args", {})
                if tool == "health":
                    result = {"ok": True, "status": "ready"}
                elif tool in ("detect_motion", "observe_scene"):
                    # Keep stdout exclusively for the JSON-lines protocol.
                    with contextlib.redirect_stdout(sys.stderr):
                        result = tools.execute(tool, args)
                else:
                    result = {"ok": False, "error": "Unknown tool"}
                if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                    raise ValueError("Tool returned an invalid result")
            except Exception as exc:
                result = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        tools.close()


if __name__ == "__main__":
    main()
