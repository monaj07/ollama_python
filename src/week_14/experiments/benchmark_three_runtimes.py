
import sys
import time
import statistics

from ultralytics import YOLO

image = sys.argv[1]

models = {
    "PyTorch": "models/yolo26n.pt",
    "ONNX Runtime": "models/yolo26n.onnx",
    "NCNN": "models/ncnn320/yolo26n_ncnn_model",
}

for name, path in models.items():
    model = YOLO(path)

    def predict():
        return model.predict(
            image,
            imgsz=320,
            conf=0.3,
            device="cpu",
            rect=False,
            verbose=False,
        )[0]

    for _ in range(5):
        predict()

    wall_times = []
    inference_times = []

    for _ in range(20):
        start = time.perf_counter()
        result = predict()
        wall_times.append((time.perf_counter() - start) * 1000)
        inference_times.append(result.speed["inference"])

    print(f"\n{name}")
    print("Median inference:", round(statistics.median(inference_times), 2), "ms")
    print("Median full predict:", round(statistics.median(wall_times), 2), "ms")
    print("Detected objects:", len(result.boxes))

    print("Detections:")
    for box in result.boxes:
        label = result.names[int(box.cls.item())]
        confidence = float(box.conf.item())
        coords = [round(v, 1) for v in box.xyxy[0].tolist()]
        print(f"  {label}: {confidence:.3f}, bbox={coords}")
