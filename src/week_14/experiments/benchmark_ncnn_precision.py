"""Compare NCNN's default FP16-enabled path with an FP32-oriented path.

Run from the ollama_python repository root, in the existing Ultralytics venv:
    python src/week_14/benchmark_ncnn_precision.py path/to/before.jpg
Or run this downloaded script directly from the repository root:
    python ~/Downloads/benchmark_ncnn_precision.py path/to/before.jpg
"""
import argparse
import statistics
import time
from pathlib import Path

import cv2
import ncnn
import numpy as np
import onnxruntime as ort
from ultralytics.data.augment import LetterBox

parser = argparse.ArgumentParser()
parser.add_argument("image", help="Path to the same before.jpg used in prior tests")
parser.add_argument("--model", default="models/ncnn320/yolo26n_ncnn_model")
parser.add_argument("--onnx", default="models/yolo26n.onnx")
args = parser.parse_args()

# Preprocess ONCE, so all runs see exactly the same RGB/FP32/CHW input.
image = cv2.imread(args.image)
if image is None:
    raise FileNotFoundError(args.image)
image = LetterBox(new_shape=(320, 320), auto=False, stride=32)(image=image)
x = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
mat = ncnn.Mat(x[0])

model_dir = Path(args.model)
param = model_dir / "model.ncnn.param"
weights = model_dir / "model.ncnn.bin"
if not param.is_file() or not weights.is_file():
    raise FileNotFoundError(f"Expected model.ncnn.param and model.ncnn.bin in {model_dir}")


def load_net(fp16):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    # All flags MUST be set before loading the model.
    net.opt.use_fp16_storage = fp16
    net.opt.use_fp16_arithmetic = fp16
    net.opt.use_fp16_packed = fp16
    if net.load_param(str(param)) != 0 or net.load_model(str(weights)) != 0:
        raise RuntimeError("NCNN model loading failed")
    return net


def forward(net):
    with net.create_extractor() as ex:
        if ex.input(net.input_names()[0], mat) != 0:
            raise RuntimeError("NCNN input failed")
        outputs = []
        for name in sorted(net.output_names()):
            status, out = ex.extract(name)
            if status != 0:
                raise RuntimeError(f"NCNN extraction failed: {name}")
            outputs.append(np.asarray(out).copy())
    return outputs


def compare_raw(label, left, right):
    """Compare matching YOLO raw outputs, separating boxes from scores."""
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    # ONNX has an explicit batch dimension; NCNN's exported output omits it.
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if b.ndim == 3 and b.shape[0] == 1:
        b = b[0]
    if a.shape != b.shape:
        print(f"{label}: incompatible output shapes: {a.shape} vs {b.shape}")
        return
    if a.ndim != 2 or a.shape[0] != 84:
        print(f"{label}: unexpected YOLO output shape: {a.shape}")
        return
    print(f"\n{label}:")
    for title, x, y in (
        ("Bounding-box channels", a[:4], b[:4]),
        ("Class-score channels", a[4:], b[4:]),
    ):
        d = np.abs(x - y)
        print(f"  {title}: max={d.max():.6g}, mean={d.mean():.6g}, "
              f"p99={np.percentile(d, 99):.6g}")

    # Compare only candidate locations where at least one raw class score is
    # at or above 0.30, to avoid interpreting low-confidence boxes as detections.
    scores_a = a[4:].max(axis=0)
    scores_b = b[4:].max(axis=0)
    selected = np.maximum(scores_a, scores_b) >= 0.30
    print(f"  Candidate locations with max class score >= 0.30: {selected.sum()}")
    if selected.any():
        score_delta = np.abs(scores_a[selected] - scores_b[selected])
        box_delta = np.abs(a[:4, selected] - b[:4, selected])
        print(f"  Selected candidate top-class score diff: "
              f"max={score_delta.max():.6g}, mean={score_delta.mean():.6g}")
        print(f"  Selected candidate coordinate diff: "
              f"max={box_delta.max():.6g}, mean={box_delta.mean():.6g}")


nets = {"FP16-enabled": load_net(True), "FP16-disabled": load_net(False)}
outputs = {}
times = {name: [] for name in nets}
for name, net in nets.items():
    for _ in range(5):
        forward(net)
    outputs[name] = forward(net)
    print(f"{name} output shapes: {[a.shape for a in outputs[name]]}")

# Alternate measurement order to reduce time/temperature-order bias.
for name in ("FP16-enabled", "FP16-disabled", "FP16-disabled", "FP16-enabled"):
    for _ in range(20):
        start = time.perf_counter()
        forward(nets[name])
        times[name].append((time.perf_counter() - start) * 1000)

for name, measurements in times.items():
    print(f"{name}: median={statistics.median(measurements):.2f} ms, "
          f"p95={np.percentile(measurements,95):.2f} ms")

compare_raw("FP16-enabled vs disabled", outputs["FP16-enabled"][0], outputs["FP16-disabled"][0])

# Optional reference check: only compare raw outputs when shapes correspond.
onnx_path = Path(args.onnx)
if onnx_path.is_file():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {session.get_inputs()[0].name: x})
    for name in nets:
        compare_raw(f"{name} vs ONNX FP32", outputs[name][0], ort_outputs[0])
else:
    print("ONNX reference not found; skipping cross-runtime raw-output check")
