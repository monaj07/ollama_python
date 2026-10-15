# Week 08: prompted perception and an Ollama summary

This add-on continues the working Pi camera and YOLO experiments. It adds four
files under your current `src/week_07_08` directory; it does not replace your
camera script, existing YOLO26 detector or previous reasoning script.

## Defaults and interface

| Option | Default | Meaning |
|---|---|---|
| Model | `yoloe-26n-seg.pt` | Original text-promptable pretrained checkpoint |
| Targets | `cup teapot glasses shoe` | Categories to look for in this request |
| Confidence | `0.2` | Provisional cutoff from the current experiment |
| Input | 640, `rect=False` | Square detector input; capture size is independent |
| Device | CPU | Raspberry Pi inference |
| Benchmark runs | 0 | One prediction per normal detection request |
| Observation maximum age | 60 seconds | Age from camera receipt to submission to Ollama |
| Ollama model | `qwen3.5:4b` | Override with an installed model |

`0.2` is a starting setting, not a validated operating point across scenes.
Changing it does not remove the need to check boxes and false detections.
Use one name per concept; for example, use `glasses` rather than supplying both
`glasses` and `eyeglasses`. Different names such as `cup` and `teapot` remain
separate categories. No deduplication or custom suppression is added here.

Files:

- `yoloe_detector.py`: CLI plus reusable `YOLOEDetector` class.
- `vision_observation.py`: shared observation schema and validation.
- `reason_yoloe.py`: validated JSON to one Ollama request.
- `observe_scene.py`: fresh capture plus detection behind a reusable tool method.

The `-seg.pt` checkpoint also computes segmentation masks. The annotated output
shows boxes only. This does not remove mask computation from inference.

## 1. Check out the repository for YOLOE and Ollama

```bash
cd ~/ollama_python
```

Reuse the existing `src/week_07_08/.venv-vision` environment that successfully ran
YOLOE. No upgrade or reinstall is needed. The model and first text-prompt setup
may download files if they are not cached; perform initial setup while online.
Run from the same repository root as the earlier experiment so its cached text
encoder remains available.

## 2. Use the detector on an existing image

```bash
src/week_07_08/.venv-vision/bin/python src/week_07_08/yoloe_detector.py \
  --source outputs/scene.jpg \
  --targets cup teapot glasses shoe \
  --conf 0.2 \
  --outdir outputs/yoloe
```

Prompts and the threshold come from the user. For multiword targets, quote each
name individually:

```bash
src/week_07_08/.venv-vision/bin/python src/week_07_08/yoloe_detector.py \
  --source outputs/scene.jpg \
  --targets glasses "potted plant" cup \
  --conf 0.3 \
  --outdir outputs/yoloe_alternative
```

`--model` and `--imgsz` are optional; defaults preserve the current setup.
Use the original text-promptable YOLOE `.pt` checkpoint. Exported backends and
prompt-free `-pf.pt` checkpoints are not accepted by this wrapper.

Each output directory contains:

- `annotated.jpg`: boxes and labels for manual inspection.
- `observation.json`: requested targets, counts (including zeros), scores, boxes,
  model/input settings and timestamps.
- `metrics.json`: initialization, prompt setup and prediction timings.

The detector reads the sibling `scene.json` written by `camera_lab.py`.
It preserves `received_at_utc` rather than resetting freshness when detection
runs. Images without this metadata can still be detected, but their observations
have a null receipt timestamp and are rejected by the Ollama step. An old image
does not become fresh by rerunning the detector.

Example shape (illustrative, not an actual new camera result):

```json
{
  "schema_version": "pi-vision-v1",
  "targets": ["cup", "glasses"],
  "received_at_utc": "2026-10-14T10:00:00+00:00",
  "confidence_threshold": 0.2,
  "counts": {"cup": 0, "glasses": 1},
  "detections": [
    {"label": "glasses", "confidence": 0.21, "bbox_xyxy": [10, 10, 40, 40]}
  ]
}
```

Zero means not detected. Counts may include false positives or overlapping
duplicates. Scores are not calibrated probabilities. The wrapper preserves
those limitations and does not silently convert counts into verified facts.

## 3. Optional YOLOE timing measurement

This is separate from the normal one-prediction path:

```bash
src/week_07_08/.venv-vision/bin/python src/week_07_08/yoloe_detector.py \
  --source outputs/scene.jpg \
  --targets cup teapot glasses shoe \
  --conf 0.2 \
  --runs 10 \
  --outdir outputs/yoloe_benchmark
```

This makes an initial prediction followed by ten timed predictions of the same
loaded image. It saves median, interpolated p95 and prediction-only FPS equivalent.
It also records requested and predictor input sizes separately. Prediction
timings exclude image loading, camera work, text prompting, saved images and
Ollama. First process initialization is separate; ten repeated frames are not
an accuracy benchmark or a long-running thermal test.

Do not assume YOLOE has the same speed as the measured YOLO26n NCNN model.
This wrapper uses the original YOLOE PyTorch checkpoint to retain variable prompts.

## 4. Give a fresh observation to Ollama

Check the local service and installed models:

```bash
ollama list
curl --fail http://127.0.0.1:11434/api/tags
```

Choose the camera Python path that matches your setup. This example selects the
environment inside `src/week_07_08` when present and otherwise the repository-root
camera environment:

```bash
CAMERA_PY=src/week_07_08/.venv-camera/bin/python
if [ ! -x "$CAMERA_PY" ]; then
  CAMERA_PY=.venv-camera/bin/python
fi
"$CAMERA_PY" -c 'from picamera2 import Picamera2; print("Camera imports OK")'
```

Keep the exposure/focus adjustments already made in your `camera_lab.py`.
The add-on does not change them. Capture a fresh image and run each command only
after the preceding command succeeds:

```bash
"$CAMERA_PY" src/week_07_08/camera_lab.py capture \
  --output outputs/scene.jpg

src/week_07_08/.venv-vision/bin/python src/week_07_08/yoloe_detector.py \
  --source outputs/scene.jpg \
  --targets cup teapot glasses shoe \
  --conf 0.2 \
  --outdir outputs/yoloe

"$CAMERA_PY" src/week_07_08/reason_yoloe.py \
  --observation outputs/yoloe/observation.json \
  --model qwen3.5:4b
```

The reasoner uses only Python's standard library, so either existing environment
can run it. If your capture script still lives at `src/week_07/camera_lab.py`,
use that path in the first command.

Replace `qwen3.5:4b` with an installed model such as `gemma4:e2b` when preferred.
Thinking is disabled, temperature is zero, context is 2048 and output is limited
to 128 tokens. There is exactly one HTTP request and no automatic retry.

The reasoner validates:

- Recognized schema, requested names and nonnegative integer counts.
- Counts matching detection records, valid scores and bounding boxes.
- A timezone-aware camera receipt timestamp.
- Snapshot age no greater than 60 seconds (and no more than 2 seconds in the future).

Only JSON counts, per-label scores, requested targets, age and limitations are
sent to Ollama. No image, file content or segmentation mask is sent. Labels are
treated as data in the prompt.

The result is saved to `outputs/yoloe_reasoning/response.json`, including the
submitted evidence, model summary and request time. The capture age is checked
again at response time and recorded as `stale_at_response`; a slow LLM response
still describes its original snapshot, not a later scene.

A suitable output for one tentative glasses detection and no cups is:

> The detector reported one possible pair of glasses; this detection is unverified.

The summary must not claim an activity, assert absence, or silently turn weak
detections into verified facts. If the detector falsely reports a cup, the LLM
cannot correct it from this JSON alone. Prompting is not a grounding guarantee;
inspect both the annotated image and summary.

If the timestamp is missing or stale, capture again. Never edit timestamps to
make an old image pass. If initial model downloads consumed the freshness window,
complete them before a new capture, or use the tool interface below, which
prepares prompts first. `--max-age` is configurable for an explicitly different
snapshot-age policy, not to conceal slow processing. Capture failures and Ollama
errors exit with a nonzero code.

## 5. Establish the agent interface

`SceneObserver.observe_scene(targets, conf=0.2)` is the perception tool method.
It captures and detects; it does not call an LLM or GPIO tool.

Test it independently:

```bash
src/week_07_08/.venv-vision/bin/python src/week_07_08/observe_scene.py \
  --targets cup glasses \
  --conf 0.2
```

The tool discovers a camera environment beside the script or at the repository
root. It looks for `camera_lab.py` beside it or in `src/week_07`. Override these
explicitly if needed:

```bash
src/week_07_08/.venv-vision/bin/python src/week_07_08/observe_scene.py \
  --targets cup glasses \
  --camera-python .venv-camera/bin/python \
  --camera-script src/week_07_08/camera_lab.py
```

Each call creates a unique directory under `outputs/observations`, containing the
raw image, camera metadata, annotated image, observation and metrics. Its path
is printed. No latest-file lookup or old-image fallback is used when capture fails.
The total tool timing includes camera process startup and camera settling; it
does not imply continuous-camera performance.

For future agent integration, keep one instance alive so weights and unchanged
prompts are reused. Run the following from the repository root:

```bash
PYTHONPATH=src/week_07_08 src/week_07_08/.venv-vision/bin/python - <<'PY'
from observe_scene import SceneObserver
from reason_yoloe import summarize_observation

observer = SceneObserver()

# Later, register this bound method with the agent's tool dispatcher.
observe_scene = observer.observe_scene

observation = observe_scene(targets=["cup", "glasses"], conf=0.2)
response = summarize_observation(observation, model="qwen3.5:4b")
print(response["summary"])
PY
```

Prompts are configured before capture. A call using the same target list reuses
that configuration; a changed list recomputes text embeddings on the same model.
Calls must be serial because the predictor and target configuration are mutable.
The CLI starts a fresh process each time, so persistent reuse applies to the
Python class, not separate CLI invocations.

This example is an explicit perception-to-summary pipeline. Agent-selected tool
calls, bounded re-observation, LED actions and memory remain the next integration
stage. Expose only the intended perception method through your existing
dispatcher when that stage begins.

## Completion checkpoint

Week 08's explicit pipeline is complete when you have:

1. Checked real detector misses and false positives against annotated images.
2. Measured YOLOE prediction time separately using `--runs 10`.
3. Captured a fresh frame, produced an observation and obtained an evidence-limited summary.
4. Confirmed that an observation older than 60 seconds is rejected before Ollama is called.
5. Called the reusable observation interface successfully on the Pi.

## Validation of this bundle

The included 13 standard-library tests cover observation validation, stale/missing
timestamps, count inconsistencies, invalid scores/boxes, model/prompt reuse,
optional benchmarks, fresh capture directories, capture failure handling,
JSON-only Ollama requests and incomplete responses:

```bash
python3 -m unittest discover -s tests -v
```

These tests use substitutes for the camera, model and HTTP endpoint. They do not
verify ARM inference, physical capture quality or actual LLM grounding. Those
checks must run on your Pi with the commands above.

## References

- [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/): prompting, checkpoint selection and export behavior.
- [Ollama Chat API](https://docs.ollama.com/api/chat): chat messages, thinking control, streaming and response fields.
- The working `camera_lab.py` receipt metadata and your current Pi experiment provide the capture interface and provisional threshold used here.
