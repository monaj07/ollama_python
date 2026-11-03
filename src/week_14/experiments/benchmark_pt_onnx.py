
import time
import statistics as stats

import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO

torch.set_num_threads(2)

# Reproducible input
x = np.random.default_rng(42).random(
    (1, 3, 320, 320), dtype=np.float32
)

# PyTorch
pt = YOLO("models/yolo26n.pt").model.eval()
tensor = torch.from_numpy(x)

# ONNX Runtime
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

ort_session = ort.InferenceSession(
    "models/yolo26n.onnx",
    sess_options=opts,
    providers=["CPUExecutionProvider"],
)
input_name = ort_session.get_inputs()[0].name


def benchmark(fn, warmup=5, runs=30):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            fn()
            times.append((time.perf_counter() - start) * 1000)

    return {
        "median_ms": stats.median(times),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": min(times),
    }


for name, fn in [
    ("PyTorch", lambda: pt(tensor)),
    ("ONNX Runtime", lambda: ort_session.run(None, {input_name: x})),
]:
    print(name, benchmark(fn))
