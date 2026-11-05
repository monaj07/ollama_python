
import time
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

model = YOLO("models/ncnn320/yolo26n_ncnn_model")

camera = Picamera2()
config = camera.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    queue=False,
    buffer_count=4,
)
camera.configure(config)
camera.start()
time.sleep(2)  # Camera warm-up

records = []

try:
    for i in range(85):
        start = time.perf_counter()

        with camera.captured_request() as request:
            frame = request.make_array("main")
            sensor_ns = request.get_metadata()["SensorTimestamp"]
        captured = time.perf_counter()

        result = model.predict(
            frame,
            imgsz=320,
            conf=0.3,
            rect=False,
            verbose=False,
        )[0]
        time.sleep(0.060)  # Simulate 60 ms of extra processing
        finished = time.perf_counter()
        finished_ns = time.monotonic_ns()

        if i < 5:  # Discard inference warm-up samples
            continue

        records.append({
            "capture": (captured - start) * 1000,
            "predict": (finished - captured) * 1000,
            "total": (finished - start) * 1000,
            "sensor_ns": sensor_ns,
            "frame_age": (finished_ns - sensor_ns) / 1e6,
            **result.speed,
        })

finally:
    camera.stop()

print("\nStage                 Median     p95")
for key in ["capture", "preprocess", "inference",
            "postprocess", "predict", "total", "frame_age"]:
    values = [r[key] for r in records]
    print(f"{key:20} {np.median(values):7.2f} "
          f"{np.percentile(values, 95):7.2f} ms")

# Throughput estimated over the sequential measured loop
print("Loop throughput estimate:",
      round(1000 / np.mean([r["total"] for r in records]), 2),
      "frames/s")

ages = np.array([r["frame_age"] for r in records])

print("First 10 frame ages:", np.round(ages[:10], 1))
print("Last 10 frame ages: ", np.round(ages[-10:], 1))

timestamps = [r["sensor_ns"] for r in records]
intervals = np.diff(timestamps) / 1e6

print("Sensor frame interval median:",
      round(np.median(intervals), 2), "ms")
