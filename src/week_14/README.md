# Week 14 — Edge Vision, Multimodal Inference & Change Detection

**Status:** **Module 6 completed (5 November 2026).** The first experimental phase (through 2 November) compared VLM reasoning, YOLO and classical image differencing; subsequent experiments tested inference runtimes, precision, INT8 quantisation, streaming and application-level visual grounding. Advanced tracking/segmentation and additional runtime ports are deferred to the later capstone projects.

> **Goal.** Measure what classical CV, a specialised detector, and a small local vision-language model can and cannot do on *the same actual scene*. Optimise based on evidence rather than assuming an LLM is the correct tool for every visual subtask.

The chapter follows three stages: compare VLM, detector and classical-CV results on the saved scene; validate exports and measure runtime/precision trade-offs; then profile live capture and evaluate visual claims. The final sections collect lessons, artifacts, rerun instructions and conclusions.

## Contents

1. [Hardware, scene and hypotheses](#1-hardware-scene-and-hypotheses)
2. [What we tested](#2-what-we-tested)
3. [VLM inference: multi-image vs single-image](#3-vlm-inference-multi-image-vs-single-image)
4. [YOLO26n NCNN: detections and latency](#4-yolo26n-ncnn-detections-and-latency)
5. [Classical CV: pixel differences and automatic Otsu](#5-classical-cv-pixel-differences-and-automatic-otsu)
6. [Portable model export and numerical equivalence](#6-portable-model-export-and-numerical-equivalence)
7. [Same-model runtime benchmarks and NCNN FP16 ablation](#7-same-model-runtime-benchmarks-and-ncnn-fp16-ablation)
8. [MNN and LiteRT: conversion, INT8 and gotchas](#8-mnn-and-litert-conversion-int8-and-gotchas)
9. [Live camera: end-to-end latency, backpressure and asynchronous capture](#9-live-camera-end-to-end-latency-backpressure-and-asynchronous-capture)
10. [Visual grounding, efficiency and claim-level evaluation](#10-visual-grounding-efficiency-and-claim-level-evaluation)
11. [Engineering lessons and gotchas](#11-engineering-lessons-and-gotchas)
12. [Dataset and artifacts](#12-dataset-and-artifacts)
13. [Rerun and verify](#13-rerun-and-verify)
14. [Consolidated conclusions and repeatability checklist](#14-consolidated-conclusions-and-repeatability-checklist)

## 1. Hardware, scene and hypotheses

**Device:** Raspberry Pi 5 Model B Rev 1.1, 8 GB RAM, Debian 13 aarch64; Pi Camera Module 3. Ollama hosts local VLMs; YOLO26n is run via exported NCNN models. This follows the earlier Week 07/08 YOLOv8n experiments but the model in these Week 14 detector outputs is **YOLO26n**, not YOLOv8n.

**Task:** From two fixed-view photographs of the same indoor scene, report what changed. The patterned **mug** moved from the **left** of the glass lidded bowl to the **right**. The bowl is stationary. Both input photos are **640 × 480**. Outside vegetation can move and ambient illumination can vary.

**Ground truth for this pilot (human inspection):** one mug changed location; the glass ornament did not. No detailed object masks, multiple trials, or calibrated ground truth annotations were collected. This is an exploratory engineering study, **not** a statistical comparison of model accuracy.

| Hypothesis / question | Outcome |
|---|---|
| Can a small local VLM correctly infer a move from two RGB images? | Sometimes, but not reliably on this pair; LFM3-VL at 320×240 mostly succeeded once, other configurations hallucinated changes. |
| Is one-image spatial reasoning easier? | Both tested VLMs answered the location question correctly in the reported single-image trials. |
| Can fast YOLO-NCNN detections be matched across frames? | Not reliably: the mug was missed in the BEFORE 320 detection; some 640 labels were wrong. |
| Can classical CV find changed regions without knowing the location? | Yes: Otsu + connected components identified the mug's old and new positions, but cannot say *what* moved. |

![Same scene before and after, with an automatically generated change mask and candidate regions](data/change_detection/otsu_full_640/summary.png)

## 2. What we tested

- **Scene comparison — VLM:** `lfm3-vl:2b` with one and two photos at 640×480; two photos at 480×360 and 320×240. `gemma4:e2b` with one and two 320×240 photos. Ollama Python SDK (`Client.chat`), `think=False`, `stream=False`, `num_ctx=4096`, `num_predict=128`, `num_thread=2`, `temperature=0`. Ordered image list: BEFORE then AFTER.
- **Scene comparison — specialised detector:** `yolo26n_ncnn_model` at 320 and 640 model input, warm repeated predictions (10 runs) and detection JSONs. Main detection threshold `conf=0.4`; a separate `conf=0.2` retry did not recover a `cup` in BEFORE (reported result; no raw JSON saved).
- **Scene comparison — classical CV:** raw RGB absolute image difference + grayscale threshold of 30; then Gaussian 3×3 blur, absolute difference, Otsu's automatic threshold, and 8-connected components ranked by region area. No manually selected ROI.
- **Runtime and precision:** same-checkpoint PyTorch, ONNX Runtime and NCNN comparisons; NCNN FP16 ablation; MNN conversion validation; LiteRT FP32 and static INT8 size, latency and output comparisons.
- **Live pipeline:** Pi Camera Module 3 capture, sequential inference, a simulated 60 ms downstream workload, and a one-slot asynchronous latest-frame queue.
- **Grounding and evaluation:** distinguish recognition, spatial relations and temporal correspondence; illustrate claim-level precision/recall without presenting it as a measured accuracy benchmark.

**Important separation:** the YOLO 320 JSONs were produced from 320×240 photographs and use model input 320×320. The 640 JSON used a 640×480 photo and model input 640×640. Do not confuse photographed aspect ratio with the detector's square inference canvas.

## 3. VLM inference: multi-image vs single-image

Prompt used for the two-image runs:

> These are two photos of the same scene, ordered BEFORE then AFTER. Describe only the visible changes between them. Be specific and concise. If uncertain, say so. Do not invent changes.

The single-image question asked for the patterned mug's position *relative to the glass bowl*. Its exact original phrasing was not retained. Our `compare_vlm.py` contains a **reconstructed** equivalent, clearly labelled.

### 3.1 Complete reported server-side timings

All durations are seconds; token counts are as reported by Ollama. See [machine-readable CSV](data/vlm_benchmarks.csv).

| Model | Task | Size **per image** | Input tokens | Output tokens | Load | Prompt eval | Decode | Total | Observed answer |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| LFM3-VL:2b | Pair | 640×480 | 656 | 72 | 17.959 | 191.218 | 6.647 | 215.863 | Incorrect: mug supposedly replaced by a bowl; invented extra bowl |
| LFM3-VL:2b | Single AFTER | 640×480 | 341 | 87 | 18.192 | 95.835 | 7.863 | 121.917 | Correct mug-to-bowl spatial relation |
| LFM3-VL:2b | Pair | 480×360 | 386 | 102 | 3.375 | 105.517 | 9.228 | 118.139 | Partial: mug on right; wrongly claimed bowl moved/swapped |
| LFM3-VL:2b | Pair | 320×240 | 216 | 69 | 3.623 | 51.714 | 6.163 | 61.510 | Mostly correct mug movement; invented orange-colour change |
| Gemma4:e2b | Pair | 320×240 | 211 | 36 | 54.408 | 104.537 | 4.648 | 163.614 | Incorrect: claimed ornament disappeared |
| Gemma4:e2b | Single AFTER | 320×240 | 115 | 23 | 55.012 | 52.501 | 1.867 | 109.394 | Correct mug-to-bowl spatial relation |

For example here is the best response among all, which was yielded by `LFM3-VL:2b | Pair | 320×240` (fourth row):
```
In the first image, there is a mug on the table next to the glass jar, 
while in the second image, the mug has been moved to the right side of the table, closer to the edge. 
Additionally, the mug in the second image appears to have more orange coloring compared to the one in the first image.
```
**Interpretation:** Decreasing LFM3-VL's *per-image* size from 640×480 to 320×240 reduced prompt-eval time **191.218 → 51.714 s (3.70×)** and reported total time **215.863 → 61.510 s (3.51×)**, while reported input tokens dropped **656 → 216 (3.04×)**. The smaller pair was correctly interpreted in this one trial; **this is not evidence that downsampling generally improves visual reasoning**. The different model load times (3–18 s for LFM, 54–55 s for Gemma) must not be misattributed to image processing; residency/load state differs. Server `prompt_eval_duration` can include multimodal image processing, not just text-token prefill.

In the 640×480 single-image LFM run, the model emitted candidate 0–1000-scale bounding boxes for mug `[822,571,925,738]` and bowl `[519,512,663,701]`. Those were **model outputs**, not measured bounding-box accuracy.

**Failure mode:** single-image *spatial* relation is not the same as two-image *temporal object correspondence*. Feeding two photos often yielded plausible but wrong narratives. One representative 320×240 LFM response was subsequently supplied and quoted above; **the other verbatim responses were not archived**. Do not mistake [these observation notes](data/vlm_response_notes.md) for verbatim transcripts.

### 3.2 Why image resolution matters

Vision input is typically converted into patches/visual tokens. Fewer pixels generally reduce visual processing, but exact image tokenisation varies by model and implementation, and lower resolution can erase necessary evidence. Here the two-image prompt counts were 656 / 386 / 216 at 640 / 480 / 320 respectively. Token counts include text and other overhead, so do not equate them directly with patch counts.

## 4. YOLO26n NCNN: detections and latency

These results came from the **already existing Week 07/08-style detector setup** on the Pi. We preserve the returned `metrics.json`, `observation.json` and annotated photographs rather than inventing a CLI for the earlier implementation. The actual Week 14 model path is `models/ncnn320/yolo26n_ncnn_model` or `models/ncnn640/yolo26n_ncnn_model` in the experiment's working directory.

### 4.1 Warm prediction latency, not end-to-end latency

| NCNN input | Photo | First prediction | Warm median | Warm p95 | Warm predictions/s equivalent | Warm runs |
|---|---|---:|---:|---:|---:|---:|
| 320×320 | BEFORE | 1336.39 ms | 17.02 ms | 25.74 ms | 58.76 | 10 |
| 320×320 | AFTER | 1319.47 ms | 17.01 ms | 26.99 ms | 58.80 | 10 |
| 640×640 | BEFORE | 1379.31 ms | 71.08 ms | 85.97 ms | 14.07 | 10 |

At 640 pixels the warm inference step took **71.08 / 17.02 ≈ 4.18×** as long as the 320 BEFORE run. Only the **BEFORE** 640 outputs were uploaded. These metrics exclude camera capture, disk I/O, annotation, JSON export and any VLM inference. The first prediction also includes potential lazy backend initialisation; it is **not** a cold camera-to-answer measurement.

### 4.2 Detections and semantic mistakes

| Experiment | Raw class counts | Key observation |
|---|---|---|
| 320 BEFORE | chair=3, cup=0 | **Mug missed** |
| 320 AFTER | chair=2, cup=1, potted plant=1 | Mug recognised as `cup`, confidence **0.4551**, bbox **[263.4, 137.8, 295.6, 176.2]** |
| 640 BEFORE | chair=2, laptop=2, vase=2, cup=0 | Mug labelled `vase` at **0.4275**, bbox **[210.0, 271.5, 284.2, 345.0]**; glass bowl also `vase` at **0.4387**; two chairs labelled `laptop` |
| 640 AFTER | **No uploaded run** | Do not infer counts, accuracy or latency |

At 320, comparing detection counts would falsely report that a new mug appeared rather than the same mug moving. Lowering `conf=0.4` to **0.2** did **not** recover a `cup` from the BEFORE image in the user's retry. At 640, detail increased but classification confusion remained. A confidence filter cannot correct an incorrect class label. A zero count means **not detected**, not **absent from the scene**.

**Annotation references:**

| Experiment | Image | Observation | Benchmark |
|---|---|---|---|
| 320 BEFORE | ![YOLO 320 before](data/yolo_ncnn/320/before/annotated.jpg) | [`observation.json`](data/yolo_ncnn/320/before/observation.json) | [`metrics.json`](data/yolo_ncnn/320/before/metrics.json) |
| 320 AFTER | ![YOLO 320 after](data/yolo_ncnn/320/after/annotated.jpg) | [`observation.json`](data/yolo_ncnn/320/after/observation.json) | [`metrics.json`](data/yolo_ncnn/320/after/metrics.json) |
| 640 BEFORE | ![YOLO 640 before](data/yolo_ncnn/640/before/annotated.jpg) | [`observation.json`](data/yolo_ncnn/640/before/observation.json) | [`metrics.json`](data/yolo_ncnn/640/before/metrics.json) |

## 5. Classical CV: pixel differences and automatic Otsu

### 5.1 Absolute differences: simplest baseline

1. Load the **aligned** BEFORE/AFTER photographs.
2. Compute absolute RGB-channel differences.
3. Convert to grayscale and threshold at **30** to create a binary change mask.

The mug appears in **both old and new positions**; moving leaves, changes in lighting, table edges and other changes also create a signal. See the actual [320 pixel diff](data/change_detection/pixel_diff_320/diff.png) and [threshold-30 mask](data/change_detection/pixel_diff_320/mask.png). This initial threshold was manually chosen: a deliberate baseline, not an optimal or adaptive parameter.

### 5.2 Gaussian smoothing + Otsu + connected components

We then used an **automatic whole-image threshold**, not a manually selected table ROI:

```text
BEFORE RGB ─┐
            ├─ GaussianBlur(3×3) ─ absdiff ─ grayscale ─ Otsu ─ binary mask
AFTER RGB ──┘                                             └─ connected components
                                                             └─ top 2 by area
```

On the original **640×480** images, reproduced from the source photographs:

| Measurement | Result |
|---|---:|
| Otsu threshold | **41 / 255** |
| Changed pixels | **3.68%** |
| Largest connected region (xywh) | **(182, 272, 140, 84)**; area **3856 px** |
| Second largest (xywh) | **(528, 276, 63, 76)**; area **3557 px** |

![Full-resolution Otsu result — old and new mug positions](data/change_detection/otsu_full_640/regions.png)

Two major regions correspond to the old/new mug locations, though the first region also covers some adjacent changes. The algorithm does **not** know which region is source versus destination, that they represent the same physical mug, or whether the event matters. The method is still parameterised: 3×3 blur, 8-connectivity and a **top-2** output policy. Otsu adapts the pixel threshold but does not solve scene interpretation.

**Resolution caution:** The user also ran the Otsu script on 320×240 versions; those [mask](data/change_detection/otsu_user_320/change_mask_auto.png) and [overlay](data/change_detection/otsu_user_320/change_regions_auto.png) are archived separately. They **must not** be confused with the 640×480 region coordinates or threshold above. The original source images and the exact output-resolution distinctions are recorded in [`data/README.md`](data/README.md).

**Why not implement optical flow now?** With only two widely separated still photographs, the first practical goal is change localisation, not dense per-pixel velocity estimates. Optical flow is useful for closer consecutive frames and motion estimation, but camera movement, large displacements and dynamic backgrounds still create challenges. We deliberately **deferred** optical flow, background modelling, homography, full tracking and semantic verification to larger capstone projects.

## 6. Portable model export and numerical equivalence

The saved-image tests above exposed semantic limitations. The next experiments hold the detector checkpoint fixed to investigate deployment speed and numerical agreement.

### 6.1 Export once, then compare compatible artifacts

The runtime experiments deliberately reuse the **same YOLO26n checkpoint** (`models/yolo26n.pt`) rather than conflating different model architectures with runtime performance. We checked its SHA-256 against copies kept under the local NCNN export directories: the source checkpoint copies matched. This establishes provenance for the *source* weights, not identical numerical execution in all exported runtimes.

The FP32 ONNX export had these reported properties:

| Property | Observed value |
|---|---|
| Model | `yolo26n.pt` / YOLO26n |
| Input | `float32` NCHW `[1, 3, 320, 320]` |
| Output | `[1, 84, 2100]` raw detector proposals |
| ONNX opset | 18 |
| ONNX graph nodes | 405 |
| Input preprocessing | 320×320 tensor, with the **same** preprocessing in both inference paths |

Before comparing speed, we checked that PyTorch and ONNX Runtime agreed on a real image: the reported mean absolute raw-output difference was **1.16×10⁻⁶**, with the equivalence test passing `allclose` at the tested `1e-3` tolerance. This is the correct order of work: **check the numbers first, benchmark second**.

**Why this matters:** exported outputs are often *raw predictions*, not postprocessed final detections. A `[1, 84, 2100]` tensor is not a list of 2,100 validated objects. Score filtering, box conversion and NMS/postprocessing policy can change apparent results even with identical raw tensors. ONNX has no automatic semantic guarantee: the actual data layout/units must be established experimentally.

### 6.2 Standardise what is being timed

For the Ultralytics-level comparisons, the reported setup used the same real photo, `imgsz=320`, `rect=False` (fixed 320×320 inference canvas) and `conf=0.3`. Compare **inference**, **preprocess/postprocess** and **complete `predict()` wall time** separately. Warm multiple runs, keep CPU thread settings explicit, and avoid comparing numbers taken with different precision modes as though they were solely runtime effects.

## 7. Same-model runtime benchmarks and NCNN FP16 ablation

### 7.1 Raspberry Pi 5 warm measurements, same YOLO26n model

| Backend (Pi) | Reported inference time | Reported full `predict()` time | Important qualification |
|---|---:|---:|---|
| PyTorch eager, FP32 | ~96 ms | ~99.6 ms | Baseline eager execution |
| ONNX Runtime, FP32 | ~28 ms | ~38 ms | Graph/runtime optimisation; raw two-thread forward was ~38.29 ms in a separate test |
| NCNN, default fast settings | ~15.25 ms | ~18.49 ms | Default NCNN used FP16 storage/arithmetic/packing; **not precision-matched** to FP32 |

Ultralytics' detector interface reported the same set of five detected classes at `conf=0.3` in this comparison, though individual scores differed slightly. The table is **a practical default-configuration comparison**, not a proof that NCNN alone is 6× faster than precision-matched PyTorch. In a separate raw-forward experiment, ONNX Runtime took **38.29 ms with graph optimisation** and **41.16 ms without**; those are not the same timing boundary as Ultralytics' stage-level `inference` field.

### 7.2 Quantify the NCNN precision advantage

| NCNN setting | Median model inference time |
|---|---:|
| Default `use_fp16_storage`, `use_fp16_arithmetic`, `use_fp16_packed` enabled | **14.05 ms** |
| All three FP16 options disabled | **29.91 ms** |
| Ratio | **2.13× faster** with FP16 enabled in this experiment |

When FP16 was disabled, NCNN raw predictions closely agreed with ONNX FP32: mean absolute bounding-box difference about **4.3×10⁻⁵**, and class-score difference about **2×10⁻⁸** in the reported comparison. With FP16 enabled, some high-confidence candidate scores differed by as much as **~0.0257**. One potted-plant score moved from approximately **0.407 (FP32)** to **0.388 (NCNN FP16)**; therefore, with `conf=0.4` the *reported detection set can change*, even without a large geometric error.

This is the central lesson: **quantisation/precision can change decisions at a threshold**. “Same model weights” does not imply “same detections”. We should inspect box coordinates, raw logits/scores, thresholds and postprocessing before blaming an export or inferring a true accuracy difference.

### 7.3 Interpretation and fair benchmarking

* Pi CPU thread count, backend thread scheduling, memory pressure and the input tensor layout all influence speed. The broader experiments used two threads for some raw PyTorch/ONNX forwards, while the NCNN default used four CPU threads; they are **not a thread-matched comparison**.
* Warm prediction latency is **not** startup/load latency or camera-to-action latency. The earlier 320/640 NCNN tests in Section 4 used another benchmark protocol and must not be silently combined with this table.
* The real engineering decision is often **latency × numerical drift × task-level accuracy × RAM/power**, not milliseconds alone.

## 8. MNN and LiteRT: conversion, INT8 and gotchas

### 8.1 MNN: fast isn't enough if the output is wrong

We converted the ONNX detector to an MNN artifact of approximately **9.3 MB** and observed **~19.94 ms** inference on the Pi. However, the *raw* output disagreed strongly with the ONNX reference (reported maximum absolute difference ~**383** and mean absolute difference ~**1.53**), even though the input tensor round-trip was checked and matched exactly.

**Status: unresolved.** This does *not* establish a broken MNN runtime or a bad conversion: tensor ordering, output convention, layout, preprocessing or conversion semantics still need investigation. Do **not** quote this as an accuracy-comparable MNN result. It is a useful conversion-validation failure case.

### 8.2 LiteRT FP32 vs static INT8

The LiteRT export was performed on a Mac in a compatible **Python 3.11** environment after Python 3.14 caused a TorchAO duplicate-registration problem. FP32 and static INT8 exports were then tested on the Pi.

| Measurement | LiteRT FP32 | LiteRT static INT8 |
|---|---:|---:|
| Artifact size | **9.89 MB** | **2.91 MB** |
| Pi inference time | **28.86 ms** | **10.91 ms** |
| Reported input and output interfaces | FP32 input/output | FP32 input/output (despite internal quantisation) |

Thus the INT8 artifact was about **3.40× smaller**, with **2.65× faster** reported model inference. The INT8 calibration set contained **only four COCO8 images**, an inadequate basis for asserting preservation of detection accuracy on this indoor scene or broader deployment data. This is an educational size/latency experiment, **not a completed INT8 accuracy benchmark**.

**Critical output gotcha:** the LiteRT raw output used **normalised xywh coordinates**, while our ONNX reference used image-pixel coordinates. With a **320×320** input, multiply the four LiteRT coordinates by 320 *before* comparing them to ONNX. Once the coordinate convention was corrected, LiteRT FP32 vs ONNX FP32 showed mean absolute raw-output difference about **2.57×10⁻⁶** and maximum about **9.46×10⁻⁴**. Earlier huge “errors” were a units mismatch, not evidence of a numerical runtime defect.

After unit correction, LiteRT INT8 vs FP32 had approximately **3.49 px mean absolute difference in proposal box coordinates** across raw proposals (including low-confidence candidates; maximum ~57 px). Class-score mean absolute difference was ~**6.57×10⁻⁶**, with maximum ~**0.039**. Small *average* errors do not prevent a few predictions from crossing class/confidence thresholds.

**Actual `conf=0.3` detection examples on the two scene photos:**

| Image | FP32 detections (selected scores) | INT8 detections (selected scores) |
|---|---|---|
| BEFORE | chairs .820/.553/.481; dining table .345; potted plant .314 | chairs .536/.536/.356; plant .356 at a different location; table absent |
| AFTER | chairs .817/.700; plant .516; cup .497; table .365 | chairs .536/.536; cup .462; plant .462; table absent |

The score and detection drift is real for the observed pair, but no labelled evaluation set was collected, so **precision/recall/mAP change is unknown**. In particular, don't interpret an INT8 `0.536` confidence as a calibrated 53.6% probability.

### 8.3 Other runtimes: deliberate scope choice

* **ExecuTorch**: modern PyTorch-native path roughly `torch.export` → ExecuTorch lowering → `.pte` execution; not universally a drop-in or zero-friction deployment. No Pi benchmark in this module.
* **Core ML**: relevant for Apple platforms; conversion and device-specific compute-unit settings matter. No controlled Mac benchmark here.
* **OpenVINO**: relevant for supported CPUs/accelerators; benefits depend on supported target hardware and model operators. No controlled Pi benchmark here.
* **MNN** was attempted but remains numerically unvalidated as noted above. No speed ranking is justified from that run.

We deliberately skipped additional conversion work to prioritise capture latency, grounding and evaluation, reserving deployment expansion for the later capstone programme.

## 9. Live camera: end-to-end latency, backpressure and asynchronous capture

After measuring model execution, we moved to the full capture-and-processing loop to test whether fast inference also delivers fresh observations.

### 9.1 Know which clock you are measuring

The Pi Camera Module 3 was configured through Picamera2 as a **640×480 `RGB888` stream** with `queue=False`, `buffer_count=4`, then fed through YOLO26n NCNN at `imgsz=320`, `conf=0.3`, `rect=False`. In this Picamera2/Ultralytics combination, the resulting NumPy channel ordering was compatible with the detector's expected input path; **do not assume a pixel format name is the NumPy colour-channel order** in other libraries.

The profiling script used a Picamera2 `captured_request()` context manager and `request.get_metadata()['SensorTimestamp']`. It recorded the capture-call wait, Ultralytics preprocessing/inference/postprocessing, `predict()` wall time, total Python loop time, sensor-to-finished **frame age**, and successive *selected* sensor timestamps. The camera sensor continues operating while Python inference executes: capture-call duration is **not** camera exposure time or sensor frame period.

For stable timestamp comparisons, use `time.monotonic_ns()` with the sensor's timestamp domain and make sure that clock compatibility is correct on the deployed system. `time.perf_counter()` is suitable for elapsed operation durations. Do not casually subtract unrelated clock domains.

### 9.2 Baseline: live streaming with normal inference

| Stage / metric | Median | p95 |
|---|---:|---:|
| Capture-call wait | **15.60 ms** | **16.28 ms** |
| Preprocess | 1.22 ms | 1.31 ms |
| NCNN inference | 14.77 ms | 17.09 ms |
| Postprocess | 1.21 ms | 1.24 ms |
| Full `predict()` | **17.63 ms** | **19.94 ms** |
| Sequential loop total | **33.24 ms** | **36.59 ms** |
| Sensor timestamp → detection finish (frame age) | **27.99 ms** | **30.17 ms** |

Processed throughput was **29.65 FPS**; consecutive **processed** sensor timestamps were **33.32 ms** apart (approximately 30 Hz). The frame-age interval need not equal the Python loop interval: the camera sensor and the application loop overlap to some extent already. A “~15 ms capture wait” cannot be added to “~28 ms frame age” without double-counting pipeline stages.

### 9.3 Deliberate overload: 60 ms extra work

We inserted `time.sleep(0.060)` **after** `model.predict()` but **before** taking the final completion timestamp. This models an additional 60 ms of downstream work; it does **not** mean the NCNN neural network became 60 ms slower. The reported `predict`-region wall duration **includes the sleep**, whereas Ultralytics `result.speed` separates neural-network stages.

| Metric | Sequential slow-processing run |
|---|---:|
| Capture wait median / p95 | **21.77 / 23.06 ms** |
| Preprocess median | 1.16 ms |
| Inference median | **15.32 ms** |
| Postprocess median | 1.17 ms |
| Prediction + artificial work median / p95 | **78.18 / 82.00 ms** |
| Entire loop median / p95 | **99.96 / 102.20 ms** |
| Frame age median / p95 | **88.58 / 92.78 ms** |
| Throughput | **9.97 FPS** |
| Interval between **processed-frame** sensor timestamps | **99.96 ms** median |

The first ten frame ages were approximately **87–90 ms**, while the last ten were approximately **91–95 ms**, not a steadily rising multi-hundred-millisecond delay. This is consistent with bounded age and skipped observations rather than an unbounded FIFO application backlog. The **99.96 ms** sensor interval refers to the frames **our code selected**, not proof that the camera sensor itself slowed to 10 Hz. To establish the produced-frame rate during overload independently, measure the camera stream separately.

A fresh-frame sequential policy may wait for the next frame after processing, losing throughput but often starting from a newer image.

<img width="759" height="244" alt="image" src="https://github.com/user-attachments/assets/f673d8f9-e3a8-4043-a8a1-3508589e49a2" />


### 9.4 Asynchronous producer–consumer test

We moved capture to a separate Python `threading.Thread` and passed `(frame, SensorTimestamp)` tuples through a shared, thread-safe `queue.Queue(maxsize=1)`. The producer replaces an unconsumed queued frame with the newest one; the main thread consumes frames and runs YOLO plus the same 60 ms simulated work. The queue is **for communication and bounded buffering**; the separate thread makes the pipeline concurrent. The GIL does not eliminate useful overlap in camera I/O and native inference.

| Metric | Async latest-frame test |
|---|---:|
| Processing median / p95 (YOLO + simulated work) | **79.52 / 90.92 ms** |
| Sensor-to-completion frame age median / p95 | **106.81 / 128.05 ms** |
| Processed throughput | **12.23 FPS** |
| Old frames replaced in the one-slot application queue | **158** |
| Selected-frame sensor intervals | **66.6 or 100 ms** in the sampled trace |

**Measured trade-off:** throughput improved from **9.97 to 12.23 FPS (+22.7%)**, but median frame age worsened from **88.58 to 106.81 ms (+18.23 ms)**, and p95 age increased more. The first/last samples did **not** show a monotonically growing backlog. These are *observations for this workload*, not a universal claim that asynchronous capture makes frames older.

Why can age worsen even when throughput rises? With `queue=False` sequential capture, the consumer waits for a newly captured frame. The background producer instead supplies the **most recent frame that has already arrived**; it can have waited part of the ~33 ms camera cycle. Approximate age when processing began, subtracting separately measured medians, was **~10.4 ms sequential** vs **~27.3 ms async** (illustrative estimates, not paired frame-by-frame measurements). In short, **a ready frame is not necessarily a fresh frame**.

```text
Sequential:    capture fresh frame -> YOLO -> artificial work -> repeat
Latest-frame:  camera thread -> replace single buffered frame
                                  |
               main thread -------+-> YOLO -> artificial work -> repeat
```
<img width="759" height="391" alt="image" src="https://github.com/user-attachments/assets/589b07f3-485a-4c3d-a445-69fa6e254614" />

**Concurrency details and gotchas:**

* `Queue(maxsize=1)` supplies locking and blocking `get()` automatically; a raw `list` can implement this only with careful locking/waiting logic. A single shared latest-frame reference protected by `Lock`/`Condition` is also valid; a queue is a design convenience, not the source of concurrency.
* With **one producer and one consumer**, it is the producer's `if full(): get_nowait(); put_nowait(...)` replacement policy that keeps the application queue bounded. Handle `queue.Empty` if the consumer drains it between the check and removal. With multiple producers, replacement would need stronger atomic coordination and/or `queue.Full` handling.
* Count **replaced buffered frames**, but do not interpret that count as every frame dropped within the camera/driver stack.
* `sensor_ns` is captured alongside its frame. Measuring only loop FPS can hide stale observations; use **median and p95 frame age**, as well as dropped-frame behaviour.
* A larger FIFO backlog without replacement could increase latency without bound while production exceeds consumption. For reactive tasks, explicitly choose **freshness versus completeness**; for forensic recording, a separate durable recording path may be appropriate.
* `threading.Event` signals shutdown; release camera requests via their context manager and terminate/join the capture worker cleanly. For safety-critical closed-loop control, neither of these rough prototypes is a complete real-time system.

**Reproduction note:** the committed scripts are [`experiments/stream_profile.py`](experiments/stream_profile.py) and [`experiments/stream_async.py`](experiments/stream_async.py). Both currently include the 60 ms deliberate slowdown **after** `model.predict()`; remove that line to reproduce the normal-inference baseline. Keep the initial warm-up frames excluded from reported medians. See [Rerun and verify](#13-rerun-and-verify) for prerequisites and commands.

## 10. Visual grounding, efficiency and claim-level evaluation

The scene-comparison and deployment results address different parts of the system. This section brings them together: an observation must arrive in time and support the claim made about it.

### 10.1 Three different visual tasks

| Task | Example question | Required evidence |
|---|---|---|
| Recognition | “Is there a mug?” | A defensible instance/class observation |
| Spatial grounding | “Is the mug right of the bowl?” | Object localisation and relational geometry |
| Temporal grounding | “Which object moved?” | Reliable correspondence across BEFORE/AFTER and change evidence |

Our LFM3-VL/Gemma4 tests exposed a common error: a fluent two-image description can incorrectly claim that the **stationary bowl moved/disappeared**. Single-image spatial correctness does not guarantee temporal correctness. Likewise, a YOLO *miss* does **not** establish the mug's absence, and a classical change mask localises pixels but does not establish object identity.

A practical (future) hierarchy is:

```text
Camera / saved frames
  -> inexpensive quality/change/detection checks
  -> retain regions, timestamps and original evidence
  -> invoke VLM selectively for ambiguity, semantics and explanation
  -> verify claims against evidence; abstain when evidence is insufficient
```

This is an **architectural proposal**, not a deployed, validated Week 14 fusion agent. Do not let a fallible detector be the *only* gate for VLM calls: it can miss precisely the objects that need verification. Input downsampling reduced this pair's VLM latency, but can also erase small or safety-relevant objects; crops/full-resolution fallback should be task-specific.

### 10.2 Score atomic claims, not just plausible prose

For a two-image ground truth containing exactly one relevant change—**mug moved right**, while **bowl stayed still**—suppose a VLM predicts both “mug moved right” and “bowl moved left”. At *claim-level*:

* **TP = 1:** correct mug movement.
* **FP = 1:** invented bowl movement.
* **FN = 0:** no relevant true change missed.
* **Precision = TP/(TP+FP) = 1/2 = 50%.**
* **Recall = TP/(TP+FN) = 1/1 = 100%.**
* **F1 = 2PR/(P+R) = 2/3 ≈ 66.7%.**

The result demonstrates that *perfect recall is compatible with unsupported claims*. This scoring example is **illustrative**, not a measured aggregate metric for the tested VLMs. A meaningful evaluation dataset would need labelled scenes, object correspondences, hard negatives, lighting/occlusion conditions, and independent annotation of atomic claims. Track precision/recall, false-change rate, missing-object claims, localisation correctness, inference time and sensor-to-answer age separately.

## 11. Engineering lessons and gotchas

1. **Detection is not temporal reasoning.** Recognising a mug in each image is not the same as linking it across images. An LLM can hallucinate that stationary furniture moved; a class detector can miss one instance or relabel a mug as a vase.
2. **Geometry isn't identity.** Nearest-centre matching is attractive, but two identical cups can swap places or cross; one-to-one assignment, detection gating and appearance cues may be required. These were **not implemented** in Week 14.
3. **Threshold is not calibration.** `conf=0.4551` is not a 45.51% calibrated probability. Lowering threshold to 0.2 did not fix the 320 BEFORE `cup` detection.
4. **More pixels aren't a guaranteed fix.** 640 NCNN was ~4.2× slower than 320 and misclassified objects; LFM3-VL reasoning was less reliable in the tested large-image run. Single anecdotes do not establish universal accuracy trends.
5. **Otsu is a region proposer, not an agent.** It found two changed regions with no manual ROI; it cannot identify a mug or establish an event narrative, and background fluctuations remain a real confounder.
6. **Do not mix latency metrics.** YOLO's ~17 ms number is a warm prediction call. Ollama reported seconds of multimodal prompt evaluation and decode; neither measurement includes the same pipeline stages. For fair deployed comparisons, instrument capture → preprocess → model → postprocess → answer.
7. **Warm-start vs cold-start.** YOLO first predictions were ~1.3 s despite 17 ms warm latency. VLM load_duration differed substantially by model/run; model residency or caches are not controlled.
8. **One scene is not a benchmark dataset.** We performed an exploratory ablation across sizes/models, without repeated random seeds, structured annotations, environmental perturbations, or statistically meaningful accuracy estimates.
9. **Model architecture must match the artifact.** Week 14's exported detector reports `yolo26n_ncnn_model`, not `yolov8n` used in earlier sessions. Do not conflate those measurements.
10. **JSON isn't ground truth.** YOLO observations encode the model's predictions and labels. Actual changed objects were established by visual inspection of the supplied photographs.
11. **Check file size/resolution.** Pixel-diff artifacts and the user's Otsu rerun are **320×240**; full-resolution Otsu summary/metrics are **640×480**. Bounding boxes do not transfer without scaling.
12. **Privacy:** the photos show a domestic interior. Review image artifacts before pushing to a public GitHub repository.

## 12. Dataset and artifacts

- [`data/README.md`](data/README.md): provenance, sources and resolution distinctions.
- [`data/vlm_benchmarks.csv`](data/vlm_benchmarks.csv): every reported VLM model/size/task metric.
- [`data/vlm_response_notes.md`](data/vlm_response_notes.md): answer correctness notes and unavailability of verbatim logs.
- [`data/photos/`](data/photos/): two original scene photos and reconstructed scaled copies.
- [`data/yolo_ncnn/`](data/yolo_ncnn/): original annotations + JSON observations + prediction-only speed benchmarks (320 BEFORE/AFTER, 640 BEFORE).
- [`data/change_detection/`](data/change_detection/): raw pixel difference, manual threshold mask, user's 320 Otsu rerun, and reproducible full-size Otsu mask/overlay/metrics/composite.
- [`experiments/`](experiments/): scene-comparison helpers, runtime benchmarks and live-camera prototypes listed below.

| Experiment group | Committed scripts |
|---|---|
| Scene preparation and comparison | [`resize_images.py`](experiments/resize_images.py), [`compare_vlm.py`](experiments/compare_vlm.py), [`image_diff_baseline.py`](experiments/image_diff_baseline.py), [`change_detection_otsu.py`](experiments/change_detection_otsu.py) |
| Export validation and runtime comparison | [`compare_torch_onnx.py`](experiments/compare_torch_onnx.py), [`benchmark_pt_onnx.py`](experiments/benchmark_pt_onnx.py), [`benchmark_three_runtimes.py`](experiments/benchmark_three_runtimes.py) |
| Precision and quantisation | [`benchmark_ncnn_precision.py`](experiments/benchmark_ncnn_precision.py), [`benchmark_litert_int8.py`](experiments/benchmark_litert_int8.py) |
| Live capture | [`stream_profile.py`](experiments/stream_profile.py), [`stream_async.py`](experiments/stream_async.py) |

**Data completeness disclaimer:** the original scene/VLM/detector artifacts supplied in the first phase are archived as described above. Only one historical VLM response was later quoted verbatim; other raw VLM answers, the 640 AFTER detector run, a saved `conf=0.2` prediction JSON, an independently annotated detection dataset, and raw machine-readable logs for the later runtime and live-camera tests were **not supplied as part of these archived experiment artifacts**. Later measured results are recorded from our terminal reports in Sections 6–10. Do not treat summary tables as raw benchmark logs.

## 13. Rerun and verify

### 13.1 Environment

Run these commands **from repository root on the Pi**. This initial environment covers the saved-image CV and VLM helpers; runtime and camera dependencies are listed separately below.

```bash
# Reuse your existing Python environment if it already has these packages.
# Otherwise use a project-local venv (compatible with Debian's PEP 668 policy).
python3 -m venv --system-site-packages src/week_14/.venv
source src/week_14/.venv/bin/activate
python -m pip install pillow numpy opencv-python-headless ollama
# On the Pi you can reuse the installed system OpenCV via --system-site-packages.
```

### 13.2 Saved-image scene comparison

**Create derived image sizes** (this creates new JPEGs; not guaranteed byte-for-byte identical to the originally used resized files):

```bash
python3 src/week_14/experiments/resize_images.py \
  src/week_14/data/photos/before.jpg src/week_14/data/photos/after.jpg \
  --output-dir /tmp/week14_resized
```

**Initial difference baseline:**

```bash
python3 src/week_14/experiments/image_diff_baseline.py \
  src/week_14/data/photos/before_320.jpg \
  src/week_14/data/photos/after_320.jpg \
  --threshold 30 --output-dir /tmp/week14_diff
```

**Automatic Otsu baseline at full resolution:**

```bash
python3 src/week_14/experiments/change_detection_otsu.py \
  src/week_14/data/photos/before.jpg \
  src/week_14/data/photos/after.jpg \
  --output-dir /tmp/week14_otsu
# Expected: threshold 41; changed ~3.68%; top boxes at
# (182,272,140,84) and (528,276,63,76) on the supplied originals.
```

**One-image and two-image VLM reruns (expensive on the Pi):**

```bash
ollama list
ollama pull lfm3-vl:2b  # only if not installed
# Ollama should be serving on http://127.0.0.1:11434
python3 src/week_14/experiments/compare_vlm.py \
  --model lfm3-vl:2b --mode compare \
  --before src/week_14/data/photos/before_320.jpg \
  --after src/week_14/data/photos/after_320.jpg \
  --output /tmp/week14_vlm_pair.json

python3 src/week_14/experiments/compare_vlm.py \
  --model lfm3-vl:2b --mode single \
  --after src/week_14/data/photos/after_320.jpg \
  --output /tmp/week14_vlm_single.json
```

> `compare_vlm.py` is an independent **reproducibility helper**, not a byte-identical replacement for any pre-existing `src/week_14/compare_images.py`. It saves future *verbatim answers* and raw timing counters. Only one historical answer was subsequently supplied verbatim; the CSV preserves the reported metric values and summaries without fabricating other missing response logs.

**Model detection outputs:** The YOLO metrics and screenshots already exist in `data/yolo_ncnn/`. The NCNN models are **not bundled**; they remain in the Pi's `models/ncnn320/` and `models/ncnn640/`. Reuse the previously implemented detector workflow from Weeks 07/08 to generate new observations. Do not treat the `warm_predict_*` numbers as VLM or end-to-end timings.

### 13.3 Runtime and live-camera experiments

Reuse the Pi environment with Ultralytics, PyTorch, ONNX Runtime and NCNN installed for the runtime benchmarks. LiteRT additionally needs `ai_edge_litert`; live capture needs the system Picamera2 installation and an attached camera. These dependencies are not installed by the minimal setup above.

The scripts resolve `models/` relative to the **current working directory**. Run from repository root with the exported artifacts under `models/`, or adapt the paths to your Pi setup. Model binaries and the LiteRT reference tensor archive are not bundled with the README's saved-data artifacts.

```bash
# Requires models/yolo26n.pt, models/yolo26n.onnx and
# models/ncnn320/yolo26n_ncnn_model/.
python3 src/week_14/experiments/benchmark_three_runtimes.py \
  src/week_14/data/photos/before_320.jpg
python3 src/week_14/experiments/benchmark_ncnn_precision.py \
  src/week_14/data/photos/before_320.jpg

# Requires the NCNN model above and the Pi camera.
# Both scripts currently include 60 ms of simulated downstream work.
python3 src/week_14/experiments/stream_profile.py
python3 src/week_14/experiments/stream_async.py
```

For LiteRT, [`benchmark_litert_int8.py`](experiments/benchmark_litert_int8.py) expects `models/yolo26n.tflite`, `models/yolo26n_int8.tflite` and `models/mnn_reference.npz` (arrays `x` and `y` for the shared input and ONNX reference). [`compare_torch_onnx.py`](experiments/compare_torch_onnx.py) retains the original absolute image path `/home/monaj/images/before_320.jpg`; adapt it before rerunning on another checkout. Review each script's input and precision settings before comparing its output with the reported tables.

### 13.4 Saved-data sanity check

These checks validate Python syntax and selected archived metrics; they do not execute model inference or camera capture.

```bash
python3 -m py_compile src/week_14/experiments/*.py
python3 - <<'PY'
import csv, json
from pathlib import Path
p = Path('src/week_14/data')
with (p / 'vlm_benchmarks.csv').open() as f:
    assert len(list(csv.DictReader(f))) == 6
j = json.loads((p / 'change_detection/otsu_full_640/metrics.json').read_text())
assert j['otsu_threshold'] == 41
assert [r['bbox_xywh'] for r in j['top_regions']] == [[182,272,140,84],[528,276,63,76]]
print('Week 14 saved-data checks passed')
```

## 14. Consolidated conclusions and repeatability checklist

| Decision area | What the actual experiments support |
|---|---|
| VLM for scene changes | Expressive but slow and prone to unsupported temporal claims on this pair; smaller inputs reduced prompt cost sharply, **without establishing a general accuracy improvement**. |
| Classical CV | Fast and interpretable change localisation; no semantics, weak under lighting/camera motion; use as a candidate generator, not an event oracle. |
| YOLO/NCNN | Excellent warm speed on Pi, but class misses and threshold effects matter; source checkpoints, precision mode, preprocessing and confidence must be documented. |
| Runtime export | PyTorch↔ONNX and LiteRT FP32 became numerically close after comparing equivalent tensors/coordinate units; MNN raw outputs still need investigation. |
| LiteRT INT8 | Smaller/faster artifact but detection scores and sets changed with four-image calibration; **accuracy not quantified**. |
| Streaming | Normal live capture approached 30 processed FPS; under artificial overload, sequential capture favoured fresh observations while one-slot async favoured throughput. |
| Grounding / evaluation | Measure claim correctness and evidence, not fluent prose alone. Benchmark *both* throughput and observation age. |

**Checklist before treating a number as a deployment benchmark:**

1. Pin device, OS, model checksum, backend/version, input image and preprocessing; state precision, runtime options and threads.
2. Clarify whether the number is cold load, raw model forward, Ultralytics `predict()`, Ollama server timing, or full camera → output latency.
3. Validate output layout, coordinate units, normalisation and postprocessing **before** comparing numerical error or detections.
4. Warm runs, specify sample counts, report distribution (median/p95), and use a meaningful labelled dataset for any accuracy claim.
5. Under streaming load, measure selected sensor timestamps, processed FPS, dropped/replaced frames and **frame age**, not only per-inference ms.
6. Check memory/thermal/clock state and ensure input images/artifacts are appropriate for public release.

**Where next:** Module 6 is complete for the learning roadmap. Additional MNN debugging, cross-device runtime work, segmentation, optical flow, object tracking, efficient detector/VLM fusion and production-quality evaluation belong in the six larger capstone projects. The next module proceeds to **offline speech/audio inference and latency-aware voice pipelines**.
