"""Run using Week 06's Python environment; emit one JSON object to stdout."""
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "src/week_06"))

try:
    from motion_tools import MotionTools
    tools = MotionTools()
    try:
        observation = tools.detect_motion()
    finally:
        tools.close()
    print(json.dumps(observation, allow_nan=False))
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
