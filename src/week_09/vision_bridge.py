"""Run in Week 07–08 vision Python; return compact grounded JSON.

Default: one JSON request on stdin, one JSON response on stdout.
--persistent: newline-delimited JSON requests/responses; reuse SceneObserver.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "src/week_07_08"))

PERSON_COUNT_CONF = 0.30


def compact_observation(observation: dict) -> dict:
    """Retain raw counts for auditing; filter person counts for the agent."""
    filtered_detections = [
        d for d in observation["detections"]
        if d["label"] != "person" or d["confidence"] >= PERSON_COUNT_CONF
    ]
    filtered_counts = {target: 0 for target in observation["targets"]}
    for detection in filtered_detections:
        label = detection["label"]
        filtered_counts[label] = filtered_counts.get(label, 0) + 1

    return {
        "ok": True,
        "observed_at": observation.get("received_at_utc"),
        "targets": observation["targets"],
        "counts": filtered_counts,
        "raw_counts": observation["counts"],
        "confidence_threshold": observation["confidence_threshold"],
        "person_count_threshold": PERSON_COUNT_CONF,
        "detections": [
            {"label": d["label"], "confidence": round(d["confidence"], 3)}
            for d in filtered_detections[:16]
        ],
        "source": str(observation["source"]),
        "limitations": (
            "One-frame tentative detections. "
            "Counts do not establish physical room occupancy. "
            "Zero is not proof of absence. "
            "Scores are not calibrated probabilities."
        ),
    }


def serve(persistent: bool) -> None:
    observer = None
    # In persistent mode, each request occupies one line. Standard one-shot mode
    # remains compatible with the pre-existing subprocess.run bridge.
    requests = sys.stdin if persistent else [sys.stdin.read()]
    for line in requests:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object")
            if observer is None:
                # Reserve stdout strictly for protocol replies, even if ML packages print.
                with contextlib.redirect_stdout(sys.stderr):
                    from observe_scene import SceneObserver
                    observer = SceneObserver()
            with contextlib.redirect_stdout(sys.stderr):
                observation = observer.observe_scene(
                    targets=request["targets"], conf=request["conf"],
                )
            result = compact_observation(observation)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:600]}"}
        print(json.dumps(result, allow_nan=False), flush=True)
        if not persistent:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent", action="store_true")
    args = parser.parse_args()
    serve(args.persistent)


if __name__ == "__main__":
    main()
