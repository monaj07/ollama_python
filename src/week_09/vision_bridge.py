"""Run using Week 07–08 vision Python; emit compact, grounded JSON."""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "src/week_07_08"))

PERSON_COUNT_CONF = 0.30

try:
    request = json.load(sys.stdin)
    # Some ML libraries print banners; reserve stdout for the result JSON.
    with contextlib.redirect_stdout(sys.stderr):
        from observe_scene import SceneObserver
        observer = SceneObserver()
        observation = observer.observe_scene(
            targets=request["targets"], conf=request["conf"],
        )

    # Apply person-specific filtering only for agent observations.
    filtered_detections = [
        d for d in observation["detections"]
        if d["label"] != "person"
        or d["confidence"] >= PERSON_COUNT_CONF
    ]

    # Recalculate counts from the filtered detections.
    filtered_counts = {
        target: 0 for target in observation["targets"]
    }

    for detection in filtered_detections:
        label = detection["label"]
        filtered_counts[label] = filtered_counts.get(label, 0) + 1
    
    # Only compact evidence enters the LLM context; full observation remains on disk.
    result = {
        "ok": True,
        "observed_at": observation.get("received_at_utc"),
        "targets": observation["targets"],

        # Filtered counts for the agent
        "counts": filtered_counts,

        # Keep original counts for debugging and evaluation
        "raw_counts": observation["counts"],

        "confidence_threshold": observation["confidence_threshold"],
        "person_count_threshold": PERSON_COUNT_CONF,

        "detections": [
            {
                "label": d["label"],
                "confidence": round(d["confidence"], 3),
            }
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
    print(json.dumps(result, allow_nan=False))
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:600]}"}))
