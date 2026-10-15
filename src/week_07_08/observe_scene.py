"""Capture a fresh scene and return a timestamped YOLOE observation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time
from uuid import uuid4

from vision_observation import (
    DEFAULT_CONF, DEFAULT_TARGETS, confidence, number,
    targets_list, validate_observation, write_json,
)
from yoloe_detector import YOLOEDetector


def camera_diagnostics(error):
    """TimeoutExpired may hold bytes even when subprocess text=True was used."""
    chunks = []
    for name in ("stdout", "stderr"):
        value = getattr(error, name, None) or ""
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        if value:
            chunks.append(f"{name}:\n{value[-3000:]}")
    return "\n".join(chunks) or "No camera output was captured."


class SceneObserver:
    """Reusable perception tool. Keep one instance alive for multiple requests.

    Vision runs in this process; the camera script runs in its own environment.
    Calls must be serial. No LLM request or GPIO action occurs here.
    """

    def __init__(self, model="yoloe-26n-seg.pt", imgsz=640,
                 camera_python=None, camera_script=None,
                 outdir="outputs/observations", capture_timeout=60):
        directory = Path(__file__).resolve().parent
        python_candidates = [directory / ".venv-camera/bin/python",
                             directory.parents[1] / ".venv-camera/bin/python"]
        script_candidates = [directory / "camera_lab.py",
                             directory.parent / "week_07/camera_lab.py"]
        self.camera_python = Path(camera_python or next(
            (path for path in python_candidates if path.is_file()), python_candidates[0])).expanduser().absolute()
        # Preserve bin/python symlinks: resolving to /usr/bin/python bypasses the venv.
        self.camera_script = Path(camera_script or next(
            (path for path in script_candidates if path.is_file()), script_candidates[0])).resolve()
        if not self.camera_python.is_file() or not self.camera_script.is_file():
            raise FileNotFoundError("Camera Python/script missing; set --camera-python and --camera-script")
        number(capture_timeout, "capture_timeout", 0.01)
        self.capture_timeout = capture_timeout
        self.outdir = Path(outdir)
        self.detector = YOLOEDetector(model, imgsz)

    def observe_scene(self, targets=DEFAULT_TARGETS, conf=DEFAULT_CONF, max_age=60):
        targets = targets_list(targets)
        confidence(conf)
        number(max_age, "max_age", 0.01)
        start = time.perf_counter()
        setup = self.detector.prepare(targets)  # Downloads/embedding setup happen BEFORE capture.
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
        run_dir = self.outdir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        source = run_dir / "scene.jpg"
        capture_start = time.perf_counter()
        command = [str(self.camera_python), "-u", str(self.camera_script),
                   "capture", "--output", str(source.resolve())]
        try:
            subprocess.run(command, check=True, capture_output=True,
                           text=True, timeout=self.capture_timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Camera capture timed out after {self.capture_timeout}s.\n"
                f"Command: {shlex.join(command)}\n{camera_diagnostics(exc)}\n"
                "Run the camera command directly to isolate the capture failure."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Camera capture failed:\n{camera_diagnostics(exc)}") from exc
        capture_ms = (time.perf_counter() - capture_start) * 1000
        if not source.is_file() or not source.with_suffix(".json").is_file():
            raise RuntimeError("Camera command did not write both scene.jpg and its metadata")
        observation = self.detector.detect(source, targets=targets, conf=conf, outdir=run_dir)
        validate_observation(observation, max_age=max_age)
        observation["metrics"].update({
            "pre_capture_prompt_setup_ms": setup["prompt_setup_ms"],
            "pre_capture_targets_changed": setup["targets_changed"],
            "capture_process_ms": round(capture_ms, 3),
            "observe_scene_total_ms": round((time.perf_counter() - start) * 1000, 3),
            "tool_timing_note": "Total excludes constructing SceneObserver; includes prompt setup, camera process startup/settling, capture, detection and initial output saves. It is not pure sensor latency.",
        })
        write_json(run_dir / "metrics.json", observation["metrics"])
        write_json(run_dir / "observation.json", observation)
        return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--model", default="yoloe-26n-seg.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--camera-python")
    parser.add_argument("--camera-script")
    parser.add_argument("--outdir", default="outputs/observations")
    parser.add_argument("--max-age", type=float, default=60)
    args = parser.parse_args()
    try:
        targets_list(args.targets)
        confidence(args.conf)
        number(args.max_age, "max_age", 0.01)
        observer = SceneObserver(args.model, args.imgsz, args.camera_python,
                                 args.camera_script, args.outdir)
        observation = observer.observe_scene(args.targets, args.conf, args.max_age)
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Observation failed: {exc}\n")
    print(json.dumps(observation, indent=2))
    print("Saved:", Path(observation["source"]).parent / "observation.json")


if __name__ == "__main__":
    main()
