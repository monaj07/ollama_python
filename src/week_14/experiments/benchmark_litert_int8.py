
import time
import numpy as np
from ai_edge_litert.interpreter import Interpreter

data = np.load("models/mnn_reference.npz")
x = data["x"]
onnx_ref = data["y"]

def run_model(path):
    model = Interpreter(model_path=path, num_threads=2)
    model.allocate_tensors()

    inp = model.get_input_details()[0]
    out = model.get_output_details()[0]

    model.set_tensor(inp["index"], x)

    # Warm-up
    for _ in range(5):
        model.invoke()

    times = []
    for _ in range(30):
        start = time.perf_counter()
        model.invoke()
        times.append((time.perf_counter() - start) * 1000)

    return model.get_tensor(out["index"]), np.array(times)

fp32, t32 = run_model("models/yolo26n.tflite")
int8, t8 = run_model("models/yolo26n_int8.tflite")

for name, times in [("FP32", t32), ("INT8", t8)]:
    print(f"{name}: median={np.median(times):.2f} ms, "
          f"p95={np.percentile(times, 95):.2f} ms")

def compare(name, a, b):
    e = np.abs(a - b)
    print(f"{name}: max={e.max():.6g}, "
          f"mean={e.mean():.6g}, "
          f"p99={np.percentile(e, 99):.6g}")


def to_pixel_coordinates(y):
    y = y.copy()
    y[:, [0, 2], :] *= 320  # x, width
    y[:, [1, 3], :] *= 320  # y, height
    return y

fp32_pixels = to_pixel_coordinates(fp32)
int8_pixels = to_pixel_coordinates(int8)

compare("LiteRT FP32 vs ONNX (corrected)", fp32_pixels, onnx_ref)
compare("LiteRT INT8 vs ONNX (corrected)", int8_pixels, onnx_ref)
compare("LiteRT INT8 vs FP32 (pixels)", int8_pixels, fp32_pixels)


compare("LiteRT FP32 vs ONNX", fp32, onnx_ref)
compare("LiteRT INT8 vs FP32", int8, fp32)

# Inspect class scores separately from bounding-box coordinates
compare("Class scores", int8[:, 4:, :], fp32[:, 4:, :])
compare("Box coordinates", int8[:, :4, :], fp32[:, :4, :])

print("Speedup:", round(np.median(t32) / np.median(t8), 2), "x")
