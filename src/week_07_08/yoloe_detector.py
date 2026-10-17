"""Reusable text-prompted YOLOE detector and headless command-line interface."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import time

from vision_observation import (
    DEFAULT_CONF, DEFAULT_TARGETS, LIMITATIONS, SCHEMA,
    confidence, read_capture_metadata, targets_list, write_json,
)


class YOLOEDetector:
    """Keep weights loaded; recompute text embeddings only when targets change.

    Use one instance serially. The underlying Ultralytics predictor is mutable;
    this wrapper is not intended for concurrent calls from multiple threads.
    """

    def __init__(self, model="yoloe-26n-seg.pt", imgsz=640):
        if type(imgsz) is not int or imgsz <= 0 or imgsz % 32:
            raise ValueError("imgsz must be a positive multiple of 32")
        if not str(model).endswith(".pt") or str(model).endswith("-pf.pt"):
            raise ValueError("Use a text-promptable YOLOE .pt checkpoint, not an export or -pf.pt")
        from ultralytics import YOLOE

        self.model_path = str(model)
        self.imgsz = imgsz
        start = time.perf_counter()
        self.model = YOLOE(self.model_path)
        self.model_construct_ms = (time.perf_counter() - start) * 1000
        self.targets = None
        self.predictions = 0

    def prepare(self, targets=DEFAULT_TARGETS):
        """Configure prompts before capturing, so first-use downloads do not age a frame."""
        targets = tuple(targets_list(targets))
        changed = targets != self.targets
        start = time.perf_counter()
        if changed:
            self.model.set_classes(list(targets))
            self.targets = targets
        return {"targets_changed": changed,
                "prompt_setup_ms": round((time.perf_counter() - start) * 1000, 3)}

    def detect(self, source, targets=DEFAULT_TARGETS, conf=DEFAULT_CONF,
               outdir=None, runs=0):
        """Return an observation dict; optionally save boxes, JSON and benchmark metrics.

        runs=0 performs one prediction. runs=N adds N timed predictions of the
        same loaded image after the initial prediction. Missing camera metadata
        is allowed for image experiments, but the reasoning step rejects it.
        """
        conf = confidence(conf)
        if type(runs) is not int or not 0 <= runs <= 1000:
            raise ValueError("runs must be an integer from 0 to 1000")
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        import cv2

        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Could not decode image: {source}")
        received = read_capture_metadata(source)
        setup = self.prepare(targets)

        def predict():
            start = time.perf_counter()
            result = self.model.predict(
                image, imgsz=self.imgsz, conf=conf, rect=False,
                iou=0.55, device="cpu", verbose=False,
            )[0]
            self.predictions += 1
            return result, (time.perf_counter() - start) * 1000

        prediction_index = self.predictions + 1
        result, prediction_ms = predict()
        warm_times = []
        for _ in range(runs):
            result, elapsed = predict()
            warm_times.append(elapsed)
        detections = [
            {"label": result.names[int(box.cls.item())],
             "confidence": float(box.conf.item()),
             "bbox_xyxy": [float(value) for value in box.xyxy[0].tolist()]}
            for box in result.boxes
        ]
        found = Counter(d["label"] for d in detections)
        metrics = {
            "model_construct_ms": round(self.model_construct_ms, 3),
            **setup,
            "prediction_index_in_process": prediction_index,
            "prediction_ms": round(prediction_ms, 3),
            "warm_runs": runs,
            "input_size_requested": self.imgsz,
            "predictor_image_size": [int(v) for v in self.model.predictor.imgsz],
            "note": "Prediction timings exclude capture, image loading, text setup, output files and LLM. The first prediction in a process may include initialization. Benchmark repeats one loaded image.",
        }
        if warm_times:
            ordered = sorted(warm_times)
            position = (len(ordered) - 1) * 0.95
            lower, upper = math.floor(position), math.ceil(position)
            p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
            median = statistics.median(warm_times)
            metrics.update({
                "warm_predict_median_ms": round(median, 3),
                "warm_predict_p95_ms": round(p95, 3),
                "predict_only_fps_equivalent": round(1000 / median, 2) if median > 0 else None,
            })
        observation = {
            "schema_version": SCHEMA,
            "received_at_utc": received,
            "processed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source.resolve()),
            "model": self.model_path,
            "targets": list(self.targets),
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "input_size_requested": self.imgsz,
            "predictor_image_size": metrics["predictor_image_size"],
            "confidence_threshold": conf,
            "counts": {label: found[label] for label in self.targets},
            "detections": detections,
            "metrics": metrics,
            "limitations": LIMITATIONS,
        }
        if outdir is not None:
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            # Boxes only keep the detector interface clear; the checkpoint also computes masks.
            if not cv2.imwrite(str(outdir / "annotated.jpg"), result.plot(masks=False)):
                raise OSError("Could not save annotated image")
            write_json(outdir / "metrics.json", metrics)
            write_json(outdir / "observation.json", observation)
        return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS),
                        help="Target names; quote multiword names, e.g. 'potted plant'")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--model", default="yoloe-26n-seg.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=0, help="Extra warmed benchmark runs; default 0")
    parser.add_argument("--outdir", default="outputs/yoloe")
    args = parser.parse_args()
    try:
        targets = targets_list(args.targets)
        confidence(args.conf)
        if not 0 <= args.runs <= 1000:
            raise ValueError("runs must be from 0 to 1000")
        detector = YOLOEDetector(args.model, args.imgsz)
        observation = detector.detect(args.source, targets, args.conf, args.outdir, args.runs)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Detection failed: {exc}\n")
    import json
    print(json.dumps({"counts": observation["counts"], "metrics": observation["metrics"],
                      "observation": str(Path(args.outdir) / "observation.json")}, indent=2))


if __name__ == "__main__":
    main()
