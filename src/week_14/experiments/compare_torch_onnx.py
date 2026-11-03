
# import numpy as np
# import torch
# import onnxruntime as ort
# from ultralytics import YOLO

# # Identical FP32 input for both runtimes
# x = np.random.default_rng(42).random(
#     (1, 3, 320, 320), dtype=np.float32
# )

# # PyTorch inference
# torch.set_num_threads(2)
# model = YOLO("models/yolo26n.pt").model.eval()

# with torch.inference_mode():
#     pt_output = model(torch.from_numpy(x))

# if isinstance(pt_output, tuple):
#     pt_output = pt_output[0]

# pt_output = pt_output.numpy()

# # ONNX Runtime inference
# options = ort.SessionOptions()
# options.intra_op_num_threads = 2

# session = ort.InferenceSession(
#     "models/yolo26n.onnx",
#     sess_options=options,
#     providers=["CPUExecutionProvider"],
# )

# input_name = session.get_inputs()[0].name
# onnx_output = session.run(None, {input_name: x})[0]

# # Numerical comparison
# error = np.abs(pt_output - onnx_output)

# print("PyTorch shape:", pt_output.shape)
# print("ONNX shape:   ", onnx_output.shape)
# print("Max abs error:", error.max())
# print("Mean abs error:", error.mean())
# print("99th percentile:", np.percentile(error, 99))
# print(
#     "All close (1e-3):",
#     np.allclose(pt_output, onnx_output, atol=1e-3, rtol=1e-3),
# )


import cv2
from ultralytics.data.augment import LetterBox
import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO

image = cv2.imread("/home/monaj/images/before_320.jpg")
assert image is not None, "Image not found"

# Resize with aspect ratio preserved and padding
image = LetterBox(
    new_shape=(320, 320),
    auto=False,
    stride=32,
)(image=image)

# BGR -> RGB, HWC -> CHW, uint8 -> FP32
x = image[:, :, ::-1].transpose(2, 0, 1)
x = np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0

print("Input:", x.shape, x.dtype, x.min(), x.max())

# PyTorch
model = YOLO("models/yolo26n.pt").model.eval()

with torch.inference_mode():
    pt = model(torch.from_numpy(x))

if isinstance(pt, tuple):
    pt = pt[0]

pt = pt.numpy()

# ONNX Runtime
session = ort.InferenceSession(
    "models/yolo26n.onnx",
    providers=["CPUExecutionProvider"],
)
onnx_out = session.run(None, {"images": x})[0]

# Compare
error = np.abs(pt - onnx_out)

print("Max error:", error.max())
print("Mean error:", error.mean())
print("P99 error:", np.percentile(error, 99))
print("Equivalent:", np.allclose(pt, onnx_out, atol=1e-3, rtol=1e-3))
