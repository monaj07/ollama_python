"""Send only validated, recent detector observations to a local Ollama model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request


def validate(observation, max_age, now=None):
    if observation.get("schema_version") != "pi-vision-v1":
        raise ValueError("Unsupported observation schema")
    counts = observation.get("counts")
    if not isinstance(counts, dict) or not counts:
        raise ValueError("Missing counts")
    if any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in counts.items()):
        raise ValueError("Counts must map strings to nonnegative integers")
    stamp = observation.get("received_at_utc")
    if not isinstance(stamp, str):
        raise ValueError("No camera receipt timestamp; capture with camera_lab.py first")
    received = datetime.fromisoformat(stamp)
    if received.tzinfo is None:
        raise ValueError("Timestamp needs a timezone")
    age = ((now or datetime.now(timezone.utc)) - received).total_seconds()
    if age < -2 or age > max_age:
        raise ValueError(f"Observation age {age:.1f}s is outside allowed range; take a fresh snapshot")
    return {"received_at_utc": stamp, "age_seconds": round(age, 2),
            "counts": counts, "confidence_threshold": observation.get("confidence_threshold"),
            "limitations": "One frame, possible missed detections; zero means not detected. No object identity, distance, intent or activity is measured."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", default="outputs/detection/observation.json")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--max-age", type=float, default=60)
    parser.add_argument("--outdir", default="outputs/reasoning")
    args = parser.parse_args()
    if args.max_age <= 0:
        parser.error("max-age must be positive")
    payload = validate(json.loads(Path(args.observation).read_text()), args.max_age)
    body = {"model": args.model, "stream": False, "think": False,
            "keep_alive": "10m", "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 128},
            "messages": [{"role": "system", "content":
                          "Summarize supplied detector observations in one short sentence. Say 'the detector detected'. Report only positive counts. If all counts are zero say no objects were detected. Do not invent activities, identity, distance, relationships or unseen objects. Never treat not-detected as proof of absence. You receive JSON observations, not an image."},
                         {"role": "user", "content": json.dumps(payload)}]}
    request = urllib.request.Request("http://127.0.0.1:11434/api/chat",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    content = result["message"]["content"].strip()
    if not content:
        raise ValueError("Model returned empty content")
    record = {"model": args.model, "input": payload, "summary": content,
              "wall_time_s": round(time.perf_counter() - t0, 3),
              "age_at_response_s": round((datetime.now(timezone.utc) - datetime.fromisoformat(payload["received_at_utc"])).total_seconds(), 2),
              "note": "This is an LLM summary of a snapshot, not an independently verified observation.",
              "eval_count": result.get("eval_count"), "eval_duration": result.get("eval_duration")}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "response.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
