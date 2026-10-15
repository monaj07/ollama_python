"""Shared observation contract; no camera, ML or Ollama dependencies."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

DEFAULT_TARGETS = ("cup", "teapot", "glasses", "shoe")
DEFAULT_CONF = 0.2
SCHEMA = "pi-vision-v1"
LIMITATIONS = (
    "These are unverified detections from one frame. Counts may include false "
    "positives and duplicates; objects may be missed. Zero means not detected, "
    "not absent. Confidence scores are not calibrated probabilities. No identity, "
    "distance, intent or activity is measured."
)


def targets_list(values):
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 12:
        raise ValueError("Supply 1 to 12 target names as a list")
    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip() or len(value) > 64:
            raise ValueError("Each target must contain 1 to 64 characters")
        if any(ord(char) < 32 for char in value):
            raise ValueError("Target names cannot contain control characters")
        value = value.strip()
        if value.casefold() in {item.casefold() for item in result}:
            raise ValueError(f"Duplicate target: {value}")
        result.append(value)
    return result


def number(value, name, minimum, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def confidence(value):
    number(value, "confidence", 0, 1)
    if value == 0:
        raise ValueError("confidence must be greater than zero")
    return float(value)


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Missing camera receipt timestamp; capture with camera_lab.py first")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid camera receipt timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Camera receipt timestamp must include a timezone")
    return parsed


def read_capture_metadata(source):
    """Keep the camera timestamp; never substitute detection time or file mtime."""
    sidecar = Path(source).with_suffix(".json")
    if not sidecar.exists():
        return None
    record = json.loads(sidecar.read_text())
    if not isinstance(record, dict):
        raise ValueError(f"Capture metadata must be an object: {sidecar}")
    stamp = record.get("received_at_utc")
    if stamp is not None:
        timestamp(stamp)
    return stamp


def validate_observation(observation, max_age=60, now=None):
    """Reject stale, future-dated, inconsistent or malformed observations."""
    number(max_age, "max_age", 0)
    if max_age == 0:
        raise ValueError("max_age must be positive")
    if not isinstance(observation, dict) or observation.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported observation schema")
    targets = targets_list(observation.get("targets"))
    counts = observation.get("counts")
    if not isinstance(counts, dict) or set(counts) != set(targets):
        raise ValueError("Counts must contain exactly the requested targets")
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("Counts must be nonnegative integers")
    threshold = confidence(observation.get("confidence_threshold"))
    detections = observation.get("detections")
    if not isinstance(detections, list):
        raise ValueError("Missing detections list")
    actual = Counter()
    for detection in detections:
        if not isinstance(detection, dict) or detection.get("label") not in targets:
            raise ValueError("Detection label is outside the requested targets")
        score = number(detection.get("confidence"), "detection confidence", 0, 1)
        if score + 1e-7 < threshold:
            raise ValueError("Detection score is below the configured threshold")
        box = detection.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Each bounding box must contain four coordinates")
        for coordinate in box:
            number(coordinate, "box coordinate", -1e9, 1e9)
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Bounding boxes must have positive width and height")
        actual[detection["label"]] += 1
    if any(counts[label] != actual[label] for label in targets):
        raise ValueError("Counts do not match the supplied detections")
    received = timestamp(observation.get("received_at_utc"))
    age = ((now or datetime.now(timezone.utc)) - received).total_seconds()
    if age < -2 or age > max_age:
        raise ValueError(f"Observation age {age:.1f}s is outside the allowed range; capture again")
    return {
        "received_at_utc": observation["received_at_utc"],
        "age_seconds": round(age, 2),
        "targets": targets,
        "counts": dict(counts),
        "confidence_threshold": threshold,
        "scores_by_label": {
            label: [round(d["confidence"], 4) for d in detections if d["label"] == label]
            for label in targets
        },
        "limitations": LIMITATIONS,
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
