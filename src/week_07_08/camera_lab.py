"""Headless Picamera2 capture and frame-difference motion experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time


def write_image(cv2, path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["capture", "motion"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--pixel-threshold", type=int, default=25)
    parser.add_argument("--motion-ratio", type=float, default=0.02)
    parser.add_argument("--output", default="outputs/frame.jpg")
    parser.add_argument("--outdir", default="outputs/motion")
    args = parser.parse_args()
    if min(args.width, args.height, args.fps, args.seconds) <= 0:
        parser.error("Dimensions, FPS and duration must be positive")
    if not 0 <= args.pixel_threshold <= 255 or not 0 < args.motion_ratio <= 1:
        parser.error("Invalid threshold")

    import cv2
    import numpy as np
    from libcamera import controls
    from picamera2 import Picamera2

    cam = Picamera2()
    try:
        # Picamera2 RGB888 produces B,G,R bytes, as OpenCV expects.
        cam.configure(cam.create_video_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"},
            controls={"FrameRate": args.fps}, buffer_count=4, queue=False,
        ))
        cam.set_controls({
            "AfMode": controls.AfModeEnum.Continuous,
            "AeEnable": True,
            "AwbEnable": True,
            "ExposureValue": 1.0,
        })
        cam.start()
        time.sleep(1)  # Allow exposure, white balance and focus to settle.

        def capture():
            t0 = time.perf_counter()
            request = cam.capture_request()
            try:
                frame = request.make_array("main").copy()
                metadata = request.get_metadata()
            finally:
                request.release()
            return frame, metadata, datetime.now(timezone.utc).isoformat(), (time.perf_counter() - t0) * 1000

        if args.mode == "capture":
            frame, meta, received_at, wait_ms = capture()
            output = Path(args.output)
            write_image(cv2, output, frame)
            record = {"received_at_utc": received_at,
                      "sensor_timestamp_ns": meta.get("SensorTimestamp"),
                      "frame_shape": list(frame.shape),
                      "capture_wait_ms": round(wait_ms, 2),
                      "note": "Receipt time is not the exact sensor exposure time."}
            output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record, indent=2))
            return

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        previous = None
        waits, processing = [], []
        frames = 0
        active = 0
        last_print = last_save = -float("inf")
        start = time.perf_counter()
        try:
            with (outdir / "events.jsonl").open("w") as log:
                while time.perf_counter() - start < args.seconds:
                    frame, meta, received_at, wait_ms = capture()
                    t0 = time.perf_counter()
                    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
                    mask = np.zeros_like(gray) if previous is None else cv2.threshold(
                        cv2.absdiff(gray, previous), args.pixel_threshold, 255, cv2.THRESH_BINARY
                    )[1]
                    ratio = float(cv2.countNonZero(mask) / mask.size)
                    changed = previous is not None and ratio >= args.motion_ratio
                    previous = gray
                    process_ms = (time.perf_counter() - t0) * 1000
                    waits.append(wait_ms)
                    processing.append(process_ms)
                    frames += 1
                    active += int(changed)
                    record = {"received_at_utc": received_at,
                              "motion": changed, "changed_pixel_ratio": round(ratio, 4),
                              "capture_wait_ms": round(wait_ms, 2),
                              "processing_ms": round(process_ms, 2)}
                    log.write(json.dumps(record) + "\n")
                    now = time.perf_counter()
                    if now - last_print >= 1:
                        print(json.dumps(record), flush=True)
                        last_print = now
                    if changed and now - last_save >= 2:
                        tagged = frame.copy()
                        cv2.putText(tagged, f"motion {ratio:.1%}", (12, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        write_image(cv2, outdir / "latest_motion.jpg", tagged)
                        write_image(cv2, outdir / "latest_mask.png", mask)
                        last_save = now
        except KeyboardInterrupt:
            pass
        elapsed = time.perf_counter() - start
        if not frames:
            raise RuntimeError("No frames captured")
        summary = {"frames_processed": frames, "requested_fps": args.fps,
                   "effective_processed_fps": round(frames / elapsed, 2),
                   "capture_wait_median_ms": round(statistics.median(waits), 2),
                   "processing_median_ms": round(statistics.median(processing), 2),
                   "processing_p95_ms": round(float(np.percentile(processing, 95)), 2),
                   "motion_frames": active,
                   "note": "Capture wait is blocking wait/copy time, not sensor-to-display latency. Loop FPS includes logging and saved images."}
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        cam.close()


if __name__ == "__main__":
    main()
