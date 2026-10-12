# Raspberry Pi vision: weeks 07 and 08

This starter uses your Pi 5, Camera Module 3 and SSH. It captures and processes images locally. Setup downloads packages/model weights; later capture, detection and Ollama inference can run offline. It does not change your existing repository or GPIO circuit.

## Objectives

| Week | Build | Learn | Completion evidence |
| --- | --- | --- | --- |
| 07 | Headless image capture and frame-difference motion detection | Camera configuration, BGR arrays, grayscale, blur, thresholds, resolution/FPS, blocking wait versus processing latency | Correct-colour snapshot; motion logs/masks; metrics at two resolutions |
| 08 | Object detector producing JSON; local LLM summarizing it | Boxes/classes/confidence, model runtime and input size, errors, observation freshness, separate perception/reasoning | Annotated real images; counts checked against ground truth; warm detector timings; grounded short summary |

Week 07 answers 'did the image change?' Week 08 answers 'which trained object categories were detected?' Neither alone identifies a person, determines intent, or proves that an undetected object is absent.

## 1. Connect the camera

On the Pi, `sudo shutdown -h now`, wait for shutdown, then unplug power. Connect the 15-pin end of the Standard-Mini cable to Camera Module 3 and the 22-pin end to one Pi 5 CAM/DISP connector. Use CAM/DISP1 for consistency. Open the small connector latch gently, seat the cable squarely with its exposed contacts facing the connector's electrical contacts, then close the latch. The official hardware guide shows orientation; do not infer it just from the blue backing. Mount the camera securely and remove any lens film. Reconnect power.

The camera uses CSI rather than GPIO. No new breadboard wiring is needed. Stop any old GPIO program during these camera experiments.

## 2. Copy this starter to the Pi

Unzip the download on your Mac. If the resulting folder is in Downloads, run on the Mac:

```bash
scp -r ~/Downloads/pi_vision_starter monaj@edgepi.local:~/
ssh monaj@edgepi.local
```

Run the following commands on the Pi unless explicitly labelled Mac:

```bash
cd ~/pi_vision_starter
cat /etc/os-release
uname -m
sudo apt update
sudo apt install -y rpicam-apps python3-picamera2 python3-opencv python3-numpy python3-venv
rpicam-hello --list-cameras
mkdir -p outputs
rpicam-still --nopreview --timeout 3000 --autofocus-mode auto --width 1280 --height 720 --output outputs/camera_test.jpg
```

Expected: a listed IMX708 camera and a nonempty camera_test.jpg. YOLO instructions assume 64-bit Raspberry Pi OS (`aarch64`). Current software uses rpicam-* rather than legacy raspistill/Picamera. If no camera is detected, shut down/unplug and check the cable before changing boot configuration. Do not enable legacy camera mode.

On the Mac, inspect the first image:

```bash
scp monaj@edgepi.local:~/pi_vision_starter/outputs/camera_test.jpg ~/Downloads/
open ~/Downloads/camera_test.jpg
```

## 3. Week 07 environment and Python capture

Picamera2/libcamera and OpenCV come from apt. Inherit them in a camera-only environment:

```bash
cd ~/pi_vision_starter
python3 -m venv --system-site-packages .venv-camera
.venv-camera/bin/python -c 'import cv2, numpy; from picamera2 import Picamera2; print("Camera imports OK")'
.venv-camera/bin/python src/week_07/camera_lab.py capture --output outputs/frame.jpg
```

This saves frame.jpg and frame.json (camera receipt time, sensor timestamp, shape and blocking capture wait). Copy frame.jpg to your Mac as above to check focus and colour. Array shape should be [480,640,3]. Picamera2's RGB888 gives B,G,R bytes, matching OpenCV. Do not swap red and blue again.

## 4. Week 07 motion experiment

Keep the camera fixed in stable lighting. Start the experiment, leave the scene still for several seconds, move a hand/cup, then become still again:

```bash
.venv-camera/bin/python src/week_07/camera_lab.py motion --seconds 30 --outdir outputs/motion_640
```

Inspect events.jsonl and summary.json. While movement occurs the script saves latest_motion.jpg and latest_mask.png, at most once every two seconds. White mask pixels exceed the per-pixel change threshold. A motion event means at least 2% of pixels changed by more than 25 grayscale intensity units between consecutive blurred frames. These are initial tuning settings, not calibrated universal thresholds.

Experiment in this order:

1. Static scene: observe noise/false triggers.
2. Move a hand or cup: observe true triggers.
3. Leave a cup stationary: observe no ongoing motion signal.
4. Change room lighting or move the camera: observe false triggers from global changes.
5. Move an object very slowly: note possible missed changes with consecutive-frame differencing.

Run the same 30-second scene sequence at higher resolution:

```bash
.venv-camera/bin/python src/week_07/camera_lab.py motion --width 1280 --height 720 --fps 15 --seconds 30 --outdir outputs/motion_1280
cat outputs/motion_640/summary.json
cat outputs/motion_1280/summary.json
```

Use `--pixel-threshold 35 --motion-ratio 0.04` to investigate fewer false positives; reduce thresholds to investigate sensitivity. This demo does not compensate for camera motion, illumination changes, or autofocus/exposure adjustment. Let focus settle; use a stable, well-lit scene for your primary comparison.

Compare effective processed FPS, capture-wait median, processing median/p95 and false triggers. Capture wait includes waiting for a frame and copying it. It is not complete sensor-to-output latency. FPS includes log/image-writing overhead. A requested 15 FPS is a configuration request, not a promised measured processing rate. Sensor timestamps are retained for later timing work; the script does not subtract them from an unrelated clock.

## 5. Week 08 detector environment

Use a separate ordinary venv for ML packages. The JPEG boundary means pip's NumPy/OpenCV dependencies cannot interfere with the apt-managed camera environment. You do not need to activate either environment; explicitly calling its Python avoids ambiguity.

```bash
cd ~/pi_vision_starter
python3 -m venv .venv-vision
.venv-vision/bin/python -m pip install --upgrade pip
.venv-vision/bin/python -m pip install ultralytics ncnn
.venv-vision/bin/python -c 'import cv2, ultralytics, ncnn; print("Detector imports OK")'
mkdir -p models/320
cd models/320
../../.venv-vision/bin/python -c 'from ultralytics import YOLO; YOLO("yolo26n.pt").export(format="ncnn", imgsz=320)'
cd ../..
.venv-vision/bin/python -m pip freeze > outputs/vision_packages.txt
```

Export downloads weights and may download/install additional conversion dependencies. Let it finish while online. If it asks you to rerun after a dependency installation, rerun the export. The NCNN directory contains multiple files including metadata: preserve the entire directory. Export happens on the Pi for this baseline. It is setup work, not inference timing.

YOLO26n is a small pretrained detector from the current Ultralytics Pi guide. Start at 320 input size, then compare 640; higher input size may improve small-object detection at additional compute cost. This is a fixed-vocabulary object detector, not text-prompt detection, segmentation, tracking, or a VLM.

## 6. Detect and benchmark one real image

Arrange a person, a clear cup/mug, and an open laptop in good lighting. Not every arrangement will yield all three labels.

```bash
.venv-camera/bin/python src/week_07/camera_lab.py capture --output outputs/scene.jpg
.venv-vision/bin/python src/week_08/detect.py --source outputs/scene.jpg --model models/320/yolo26n_ncnn_model --imgsz 320 --outdir outputs/ncnn320
cat outputs/ncnn320/observation.json
```

On the Mac:

```bash
scp monaj@edgepi.local:~/pi_vision_starter/outputs/ncnn320/annotated.jpg ~/Downloads/
open ~/Downloads/annotated.jpg
```

The detector writes annotated.jpg, observation.json and metrics.json. All detected classes are retained; person/cup/laptop also receive explicit zero counts when not detected. The JSON includes bounding boxes, confidence threshold, model/input size and capture receipt time. Missing timestamps on unrelated image files are allowed for detector testing, but rejected by the reasoning script.

Use the same source to compare warmed PyTorch and NCNN:

```bash
.venv-vision/bin/python src/week_08/detect.py --source outputs/scene.jpg --model models/320/yolo26n.pt --imgsz 320 --outdir outputs/pytorch320
.venv-vision/bin/python src/week_08/detect.py --source outputs/scene.jpg --model models/320/yolo26n.pt --imgsz 640 --outdir outputs/pytorch640
cat outputs/ncnn320/metrics.json
cat outputs/pytorch320/metrics.json
cat outputs/pytorch640/metrics.json
```

For NCNN 640, create models/640, copy models/320/yolo26n.pt into it, export there with imgsz=640, then use that directory with --imgsz 640. Match prediction size to the exported input size. Exporting twice to the same weight directory would replace the earlier export.

Each benchmark excludes the first prediction and performs 10 warmed predictions of the same in-memory image. The script reports median/p95 prediction wall time and prediction-only FPS equivalent, including preprocessing/model/postprocessing within model.predict. This is not live camera FPS. Model construction and first prediction are reported separately because backend initialization can be lazy. Timings exclude camera startup/capture, image read, annotation/JSON writing and LLM. Use the same scene, confidence, environment and thermal conditions for comparison; a single frame measures speed, not accuracy.

In a second SSH/tmux pane, if desired:

```bash
watch -n 2 'vcgencmd measure_temp; vcgencmd get_throttled'
```

## 7. Check detector errors

Capture about 10 scenes: no target objects; person only; cup only; laptop only; all three; two cups; partially hidden cup; distant cup; dim light; stationary person. Use distinct filenames/outdirs to retain evidence. For each, manually record actual visible counts and compare detected counts/boxes. List misses, false detections and duplicate boxes. Try --conf 0.25 and --conf 0.60 on the same saved image to observe the tradeoff. Confidence scores are not calibrated correctness probabilities. A label 'cup' is not guaranteed for every glass/container.

The 10-scene exercise is a diagnostic sanity check, not a general accuracy benchmark. Counts are per image; repeatedly seeing a person is not a count of unique people over time.

## 8. Connect detector JSON to Ollama

Verify Ollama and choose an already-installed model from week 04:

```bash
ollama list
curl http://127.0.0.1:11434/api/tags
```

If your Ollama service is stopped, start `ollama serve` in another terminal. Do not start a duplicate service if it is already running. The default below is the familiar qwen3.5:4b; replace it with your benchmark winner if desired. The script requests think=false, supported by your previous Qwen tests; a model that does not support that option needs the request adjusted.

Capture a fresh scene immediately before this sequence. Only proceed to the next line if the prior command succeeded:

```bash
.venv-camera/bin/python src/week_07/camera_lab.py capture --output outputs/scene.jpg
.venv-vision/bin/python src/week_08/detect.py --source outputs/scene.jpg --model models/320/yolo26n_ncnn_model --imgsz 320 --outdir outputs/ncnn320
.venv-camera/bin/python src/week_08/reason.py --observation outputs/ncnn320/observation.json --model qwen3.5:4b
```

Reasoning rejects malformed/noninteger counts, missing camera timestamps and input observations older than 60 seconds. It requests one short sentence, temperature 0, 128 output tokens and no thinking. It records request wall time and observation age again when the response arrives. The output describes the captured frame, not necessarily the scene at response time. Prompt constraints are helpful but are not deterministic output-grounding guarantees: inspect the response for invented activities or objects.

The LLM receives only JSON, never the JPEG. A valid response might say: 'The detector detected one person, one cup and one laptop.' It cannot infer that a person is working, drinking, happy, known to you, or close to the camera. For count reporting alone, a Python template is faster and more reliable. The LLM step teaches a reusable observation interface for later questions, memory and tools; it does not make this simple counting task more accurate.

One observation -> one LLM request. Do not invoke the LLM for every camera frame. Later a button, motion event or debounced stable count change can request a fresh observation. Take care with capture failure: never reason from leftover JSON after a failed capture/detection command.

## Next integration

Expose the capture/detect sequence as a read-only `observe_scene()` tool in the earlier Python agent. First call it explicitly and inspect its output; only then allow the LLM to choose the tool. Keep camera, detector and LLM durations separate. Live detection with a persistent loaded model and newest-frame handling is the next optimization; this starter intentionally uses an easy-to-debug snapshot/file interface.

When the experiments are complete, scripts belong under src/week_07 and src/week_08 in ollama_python. Preserve measured results in documentation; leave generated images, virtual environments and downloaded models out of git. This download has not been committed to that repository.

## Validation and references

The scripts have been syntax-checked and observation validation has been exercised with recent/stale/malformed inputs. No physical Pi camera, ARM detector inference or your Ollama service is available in the authoring environment, so the commands still require the hardware checkpoints above.

- Hardware installation: https://www.raspberrypi.com/documentation/accessories/camera.html
- Camera commands: https://www.raspberrypi.com/documentation/computers/camera_software.html
- Picamera2 manual (installation, pixel formats, virtual environments): https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
- Ultralytics Pi deployment/NCNN: https://docs.ultralytics.com/guides/raspberry-pi/
