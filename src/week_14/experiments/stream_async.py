
import queue
import threading
import time

import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

model = YOLO("models/ncnn320/yolo26n_ncnn_model")

camera = Picamera2()
camera.configure(camera.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    queue=False,
    buffer_count=4,
))
camera.start()
time.sleep(2)

latest = queue.Queue(maxsize=1)
stop = threading.Event()
stats = {"replaced": 0}


def capture_worker():
    while not stop.is_set():
        try:
            with camera.captured_request() as req:
                frame = req.make_array("main")
                sensor_ns = req.get_metadata()["SensorTimestamp"]
        except RuntimeError:
            if stop.is_set():
                break
            raise

        if latest.full():
            try:
                latest.get_nowait()
                stats["replaced"] += 1
            except queue.Empty:
                pass

        latest.put_nowait((frame, sensor_ns))


thread = threading.Thread(target=capture_worker, daemon=True)
thread.start()

ages, processing, finished_times, timestamps = [], [], [], []

try:
    for i in range(85):
        frame, sensor_ns = latest.get(timeout=3)

        start = time.perf_counter()

        model.predict(
            frame, imgsz=320, conf=0.3,
            rect=False, verbose=False,
        )

        time.sleep(0.060)  # Same simulated workload

        end = time.perf_counter()
        end_ns = time.monotonic_ns()

        if i >= 5:
            processing.append((end - start) * 1000)
            ages.append((end_ns - sensor_ns) / 1e6)
            finished_times.append(end)
            timestamps.append(sensor_ns)

finally:
    stop.set()
    camera.stop()
    thread.join(timeout=3)


def report(name, values):
    print(f"{name}: median={np.median(values):.2f} ms, "
          f"p95={np.percentile(values, 95):.2f} ms")


report("Processing", processing)
report("Frame age", ages)

fps = (len(finished_times) - 1) / (
    finished_times[-1] - finished_times[0]
)

print(f"Processed throughput: {fps:.2f} FPS")
print("Replaced queued frames:", stats["replaced"])
print("First 10 ages:", np.round(ages[:10], 1))
print("Last 10 ages:", np.round(ages[-10:], 1))
print("Processed-frame sensor intervals:",
      np.round(np.diff(timestamps[:11]) / 1e6, 1))
