"""Detect objects in a local JPEG; save boxes and an auditable observation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--outdir", default="outputs/detection")
    args = parser.parse_args()
    if args.runs < 1 or args.imgsz <= 0 or not 0 < args.conf <= 1:
        parser.error("Invalid runs, image size or confidence")
    import cv2
    import numpy as np
    from ultralytics import YOLO

    source = Path(args.source)
    image = cv2.imread(str(source))
    if image is None:
        raise OSError(f"Could not load {source}")
    meta_path = source.with_suffix(".json")
    capture = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model = YOLO(args.model, task="detect")
    load_ms = (time.perf_counter() - t0) * 1000

    def predict():
        return model.predict(
            image,
            imgsz=args.imgsz,
            conf=args.conf,
            device="cpu",
            rect=False,
            verbose=False,
        )[0]

    t0 = time.perf_counter()
    predict()  # Exclude cold first prediction from warmed measurements.
    first_ms = (time.perf_counter() - t0) * 1000
    durations = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        result = predict()
        durations.append((time.perf_counter() - t0) * 1000)
    detections = []
    for box in result.boxes:
        detections.append({"label": result.names[int(box.cls.item())],
                           "confidence": round(float(box.conf.item()), 4),
                           "bbox_xyxy": [round(float(x), 1) for x in box.xyxy[0].tolist()]})
    counts = Counter(d["label"] for d in detections)
    # Watched classes have explicit zeros; other detected classes remain visible.
    watched = {label: counts.get(label, 0) for label in ("person", "cup", "laptop")}
    watched.update(counts)
    observation = {"schema_version": "pi-vision-v1",
                   "received_at_utc": capture.get("received_at_utc"),
                   "processed_at_utc": datetime.now(timezone.utc).isoformat(),
                   "source": str(source), "model": args.model,
                   "image_size": [image.shape[1], image.shape[0]],
                   "input_size_requested": args.imgsz,
                   "predictor_image_size": [int(value) for value in model.predictor.imgsz],
                   "confidence_threshold": args.conf,
                   "counts": watched, "detections": detections,
                   "limitations": "Counts describe detections in one frame. Zero means not detected; objects can be missed. Confidence is not a calibrated probability."}
    (outdir / "observation.json").write_text(json.dumps(observation, indent=2) + "\n")
    if not cv2.imwrite(str(outdir / "annotated.jpg"), result.plot()):
        raise OSError("Could not save annotated image")
    median = float(np.median(durations))
    metrics = {"model": args.model,
               "input_size_requested": args.imgsz,
               "predictor_image_size": [int(value) for value in model.predictor.imgsz],
               "warm_runs": args.runs,
               "model_construct_ms": round(load_ms, 2),
               "first_predict_ms": round(first_ms, 2),
               "warm_predict_median_ms": round(median, 2),
               "warm_predict_p95_ms": round(float(np.percentile(durations, 95)), 2),
               "predict_only_fps_equivalent": round(1000 / median, 2),
               "note": "Repeated predictions of one loaded image; excludes camera, disk I/O, JSON, plot, and LLM. Some backends load lazily in first_predict."}
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"counts": watched, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
