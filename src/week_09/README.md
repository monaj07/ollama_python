# Week 09 — Offline voice assistant with physical tools

Week 09 integrates earlier Raspberry Pi projects into one **push-to-talk,
fully local voice assistant**: record speech, transcribe with whisper.cpp,
call `gemma4:e2b` through Ollama, optionally inspect the room using the
PIR sensor and YOLOE camera detector, and speak a reply using Piper.

The implementation was tested on a Raspberry Pi 5 (8 GB) with Camera Module 3,
a PIR on GPIO23, a USB microphone, and a Bluetooth speaker.

## What this week demonstrates

- End-to-end **speech → STT → local LLM/tool loop → TTS → speaker**.
- Tool selection with strict schemas and argument validation; separate Week 06
  and Weeks 07–08 Python environments for hardware adapters.
- A bounded, per-turn Ollama agent with no retained conversation or sensor
  memory between turns.
- An evidence-completeness policy for physical room-occupancy questions:
  attempt both PIR and person detection rather than trusting PIR alone.
- Guarded wording for failed, negative, or conflicting sensor observations;
  **motion does not establish occupancy, and no detection does not prove absence**.
- A persistent vision worker that reuses YOLOE and its cached text prompts
  within an interactive assistant session.
- JSONL logs and per-stage latency measurements.

This is **not** always-on listening: press Enter to record one fixed-duration
utterance. Each microphone turn is recorded afresh; `--audio` and `--text`
are one-shot test modes.

## Architecture

```text
USB microphone --arecord--> 16 kHz mono WAV
                             |
                             v
                   whisper.cpp tiny.en
                             |
                             v
                      recognized text
                             |
                             v
                   Week 09 VoiceAgent
                     /           \
            detect_motion     observe_scene
               (PIR)        (Camera + YOLOE)
                |                  |
      Week 06 GPIO venv     Persistent Week 07–08
                           vision worker / venv
                     \           /
                       tool evidence
                             |
                             v
                     Ollama gemma4:e2b
                             |
                             v
                        Piper TTS
                             |
                             v
                    paplay / PipeWire
                             |
                             v
                     Bluetooth speaker
```

`voice_assistant.py` handles recording, STT, TTS, playback, and stage timings.
`agent.py` handles Ollama, tool calls, occupancy evidence policy, and traces.
`tools.py` validates and dispatches physical tools; `pir_bridge.py` and
`vision_bridge.py` communicate with the respective earlier-week environments.
The camera tool returns compact detector evidence, **not an image to Ollama**.

## Requirements and local setup

Run from the repository root on the Pi. The following must already be installed
and working (see Weeks 06 and 07–08 for the physical sensor and camera setup):

- Local Ollama server with `gemma4:e2b` (`ollama list`).
- Working Week 06 PIR environment: `src/week_06/.venv/bin/python`.
- Working Weeks 07–08 vision environment: `src/week_07_08/.venv-vision/bin/python`,
  with Picamera2, YOLOE model assets and `SceneObserver` dependencies.
- `whisper-cli` under `~/opt/whisper.cpp/build/bin/`, plus
  `~/opt/whisper.cpp/models/ggml-tiny.en.bin`.
- Week 09 virtual environment with the Ollama Python SDK and Piper installed;
  Piper voice files in `src/week_09/voices/` (default
  `en_US-lessac-medium.onnx` and its configuration).
- ALSA recording (`arecord`) and audio output (`paplay`), with a configured
  PipeWire/PulseAudio default sink (the Bluetooth speaker for this setup).

```bash
cd ~/ollama_python
source src/week_09/.venv/bin/activate
ollama list
arecord -l
pactl info | grep 'Default Sink'
```

The microphone used in this experiment was `plughw:0,0`; substitute the
actual device printed on your Pi. The Ollama endpoint is
`http://127.0.0.1:11434`. The assistant preloads its model at launch.

## Run the assistant

Interactive push-to-talk; Enter to speak, `q` to exit:

```bash
cd ~/ollama_python
source src/week_09/.venv/bin/activate
python src/week_09/voice_assistant.py --mic-device plughw:0,0 --duration 5
```

Text-only input skips recording and Whisper, **but still synthesizes speech**:

```bash
python src/week_09/voice_assistant.py --text 'Is anyone in the room right now?' --no-play
```

Process an existing **16 kHz, mono, 16-bit PCM WAV** recording:

```bash
python src/week_09/voice_assistant.py --audio /path/to/question.wav
```

Use `--no-play` to keep the synthesized WAV without sending it to the speaker.
The default model is `gemma4:e2b`, Whisper is `tiny.en`, and the Piper voice is
`en_US-lessac-medium`; override their paths/model names with the CLI flags
shown by `python src/week_09/voice_assistant.py --help`.

Try a general question (e.g. `What is 2 plus 17?`), a motion-only question
(e.g. `Is movement detected right now?`), a camera-only question
(e.g. `Can you see a cup?`), and an occupancy question
(e.g. `Is anybody in the room right now?`). For the occupancy query, both
PIR and camera should be attempted even if the model selects only one.

## Annotated experiments: detector evidence is not room occupancy

These **real camera experiments** show why Week 09 treats a `person` box as
uncertain evidence, not proof of a person physically in the room. The scores
and boxes in the images are raw YOLOE output at `conf=0.2`, **before** the
Week 09 person-only `0.30` filtering policy. These captures predate the
later 1-second camera-settling timing experiment.

### Experiment A — overlapping person boxes on a partial arm

![Two overlapping tentative person boxes (0.58 and 0.20) around a foreground arm, with cups and plants in the background](assets/person-overlap-arm.jpg)

**Camera target:** `person`, observed 2026-10-17. The original output
contains **two raw `person` boxes**, scores `0.5812` and `0.2046`,
overlapping on the foreground arm. The Week 09 `person>=0.30` threshold
would discard the weaker box and leave **one filtered candidate**, but the
remaining box is still not an independently verified person. Increasing the
threshold alone does not eliminate all false positives.

### Experiment B — person in the room, person on a screen, and a hand

![YOLOE boxes for a person in front of the camera (0.86), a person on a screen (0.85), and a hand mistaken for a person (0.22)](assets/person-on-screen-and-hand.jpg)

**Camera target:** `person`, observed 2026-10-17. Raw detections were
`0.8623` (person physically in front of the camera), `0.8526` (person
shown on a screen), and `0.2217` (hand/gesture mistaken for a person).
The Week 09 0.30 filter changes the candidate count **from 3 to 2**, but
**one of the remaining two appears on a screen**. These figures do not
establish physical room occupancy.

### Earlier Week 08 detector experiments

| Open-vocabulary YOLOE: glasses on a face | Fixed-vocabulary YOLO26 NCNN: objects on a table |
|---|---|
| ![YOLOE glasses on a face, confidence approximately 0.90](../week_07_08/results/glasses-worn/annotated.jpg) | ![YOLO26 NCNN 640 cup and furniture detections](../week_07_08/assets/yolo26-ncnn640.png) |
| [Original glasses experiment and metadata](../week_07_08/README.md#recorded-glasses-experiment-and-metadata): glasses could be missed when held separately but detected strongly when worn. | [Original YOLO26 benchmark](../week_07_08/README.md#recorded-table-scene-comparison): detection varied with image resolution and backend. |

These final two are **earlier Week 08 reference images**, not captures made
by the Week 09 assistant. Week 09 calls YOLOE, not the fixed-vocabulary
YOLO26 model used for the table experiment.

## Tool and evidence constraints

| Constraint | Implementation |
|---|---|
| Allowed tools | `detect_motion` and `observe_scene` only |
| `observe_scene` arguments | 1–8 text targets; confidence 0.1–0.8 (default 0.2) |
| Max tool rounds | 3, plus a final model round without tools |
| Tool-call budgets | 4 calls per turn, 2 per model round |
| Person-specific filter | confidence >= 0.30 for `person` in Week 09 only |
| Other class detections | original detector threshold is retained |
| Occupancy evidence | Attempt PIR + person camera, with explicit uncertainty |
| Camera failure | Return structured failure; do not infer an empty room |

The vision result includes filtered counts, original `raw_counts`, a timestamp,
and a path to the saved scene. Detection confidence scores are not calibrated
probabilities. People depicted on screens, hands mistaken for people, and
occlusions remain possible sources of error.

## Logs and output

- `src/week_09/outputs/runs.jsonl`: run IDs, model, stage durations.
- `src/week_09/outputs/tool_events.jsonl`: per-request tool events and answers.
- `src/week_09/outputs/`: recording WAVs, transcripts and synthesized replies.
- `outputs/observations/`: captured scenes and full detector metadata from
  Weeks 07–08. The agent receives a compact evidence subset.

```bash
tail -n 5 src/week_09/outputs/runs.jsonl
tail -n 10 src/week_09/outputs/tool_events.jsonl
python -m pytest src/week_09/tests
```

These local artefacts are not intended to be committed to Git; check
`.gitignore` before adding new generated files.

## Raspberry Pi 5 measurements

All figures are from this experiment, not universal hardware benchmarks.
The camera settling delay was reduced from 3 seconds to 1 second after
comparing the captured images under similar conditions.

| Vision configuration | Measured latency |
|---|---:|
| Original one-shot bridge, 3 s camera settling | 9.670 s |
| One-shot bridge, 1 s camera settling | 7.808 s |
| Persistent vision worker, initial request | 6.785 s |
| Persistent vision worker, warm request 2 | 2.830 s |
| Persistent vision worker, warm request 3 | 2.841 s |

The warm worker averages **2.836 s**, approximately **63.7% less wall time**
than the 7.808 s one-shot bridge. Reusing `SceneObserver` eliminated repeated
YOLOE prompt setup (previously ~1.9 s per first observation); camera capture
remained about 2.2 s per observation.

Two consecutive warm spoken occupancy requests measured:

| Stage | Run 1 | Run 2 |
|---|---:|---:|
| Record | 5.037 s | 5.029 s |
| Whisper STT | 1.685 s | 1.116 s |
| Agent + tools | 22.123 s | 22.713 s |
| Piper TTS | 4.040 s | 4.150 s |
| Playback | 6.246 s | 6.160 s |
| **End-to-end** | **39.132 s** | **39.169 s** |

The camera tool was **2.910 s / 2.881 s**, but Ollama took about 8.5 s
for the first model round and 10.4–11.0 s for the second. The Ollama timings
showed prompt evaluation (~7.8 s and ~8.7 s) dominating generation, while
model load time was ~0.002 s. These measurements do not establish the root
cause of slow prompt evaluation.

While Ollama and the persistent vision worker were both running, the system
reported **1.9 GiB available RAM and 0 B swap in use**. No model eviction was
observed during that check.

## Reproducible memory and latency optimisation recipe

**Change one variable at a time**, keep cold and warm runs separate, and
check accuracy and failure behaviour after each performance change. These
figures are observed on one Pi 5 setup, not universal performance targets.

### 1. Establish a repeatable end-to-end baseline

```bash
cd ~/ollama_python
source src/week_09/.venv/bin/activate
python src/week_09/voice_assistant.py --mic-device plughw:0,0 --duration 5
# Ask the same question three times in this ONE interactive process; then quit.
tail -n 5 src/week_09/outputs/runs.jsonl
```

Compare `record_s`, `stt_s`, `llm_s`, `tts_s`, `play_s` and `total_s`.
`llm_s` **includes tools and multiple model calls**, not only Ollama.
Use `--text 'Is anyone in the room right now?' --no-play` to skip microphone,
STT and speaker playback, but remember `--text` mode exits after one request
and does not measure warmed worker reuse across questions.

### 2. Separate Ollama prompt evaluation from generation

The Ollama chat response has `load_duration`, `prompt_eval_duration` and
`eval_duration` (all **nanoseconds**), and the corresponding prompt/output
token counts. Temporarily wrap each `self.client.chat(...)` in
`src/week_09/agent.py` with `time.perf_counter()` and print per-round:

```text
[ollama round N] wall=...s, load=...s, prompt=...s,
                 generation=...s, prompt_tokens=...,
                 generated_tokens=..., tok/s=...
```

This optional per-round instrumentation was added locally for profiling and
**is not included in this documentation-only patch**. It is reasonable to
retain it (ideally behind a debug/verbosity switch), because it does not
change model decisions. The built-in `voice_assistant.py` stage timings and
`tool_events.jsonl` are independent of these extra printouts.

| Symptom | First investigation |
|---|---|
| Large `load` duration | `ollama ps`, `keep_alive`, model eviction and warm-up options |
| Large `prompt` duration, small `load` | Prompt/tool schema size, cache reuse, CPU and memory contention |
| Large `generation` duration | Number of generated tokens and tokens/sec |
| Large `llm_s`, short individual Ollama calls | Slow tools, repeats and excessive model rounds |

Our final two warm occupancy tests had `load≈0.002 s`, but prompt processing
was about **7.8 s** in round 1 and **8.7 s** in round 2. These timings do
**not** establish the root cause. Shortening tool-result JSON or reducing
model rounds remains unimplemented future work, requiring reliability tests.

### 3. Confirm the LLM and detector fit together in memory

While the interactive assistant is still running, use a second terminal:

```bash
ollama ps
free -h
ps -eo pid,rss,comm,args --sort=-rss | head -n 12
vmstat 1 5
```

On the Pi 5 (8 GB), `gemma4:e2b` and the persistent YOLOE worker were
resident together with **1.9 GiB available** and **0 B swap used**. Check
again under load. Prefer `Mem: available` over raw `Mem: free` for usable
headroom; `vmstat` helps reveal swapping. Process RSS values include shared
pages and should not simply be added to estimate exclusive memory. An
`ollama ps` model size and a `ps` process RSS number measure different things.

Our chosen Ollama settings are `num_ctx=2048`, `num_predict=100`, temperature
0 and `keep_alive=15m`. Use matching relevant options when pre-warming the
model and executing real calls, so warm-up does not create unexpected extra
allocation. Increasing context can raise memory use. Run **one persistent
vision worker per interactive assistant** rather than spawning a new YOLOE
model for each capture; its weights and same-target text embeddings can be
reused only while its process remains alive.

### 4. Isolate vision and camera timing

```bash
PYTHONPATH=src/week_09 python - <<'PY'
from tools import PhysicalTools

tools = PhysicalTools()
try:
    for i in range(3):
        result = tools.execute("observe_scene", {"targets": ["person"], "conf": 0.2})
        print(i + 1, result.get("ok"), result.get("tool_elapsed_s"), result.get("counts"))
        if not result.get("ok"):
            print(result.get("error"))
            break
finally:
    tools.close()
PY
```

Original one-shot bridge: **9.670 s**, with 3-second camera settling.
A **local experiment** changed `time.sleep(3)` to `time.sleep(1)` in
`src/week_07_08/camera_lab.py` and lowered the one-shot result to
**7.808 s**, with similar-looking images in that room. **The remote
`camera_lab.py` still uses 3 seconds.** If reproducing our 2.836-second
warm worker results, retain or deliberately reapply that local 1-second
change, after checking exposure, focus, white balance and detector accuracy.

With the 1-second delay, persistent-worker observations took **6.785 s**
initially and **2.830 / 2.841 s** warm. In-process warm YOLOE prediction
was about **0.61 s**, versus camera capture about **2.22 s**. Shortening
capture, caching target prompts and retaining the process provided more
benefit than changing detectors without evidence.

### 5. Optimise the measured bottleneck without losing safeguards

- **Prompt:** try compact model-facing evidence (keep original tool results
  in traces), simpler tool descriptions, or fewer LLM rounds only if the
  general/PIR/camera/occupancy tests still pass. A negative or failed sensor
  result must never become a confident claim of an empty room.
- **Perception:** warm the worker, reuse targets, and test camera settling
  under the actual lighting. A `person>=0.30` cutoff removes a weak hand
  detection but still admits a person displayed on a screen.
- **Audio:** recording is currently a fixed **5 s**; shorter `--duration`
  can save time but may cut off speech. TTS and playback depend on reply
  length; preserve necessary uncertainty language.
- **Regression:** `python -m pytest src/week_09/tests`, followed by a real
  general question, PIR-only, camera-only, occupancy and simulated camera
  failure test. Compare cold/warm latency and available RAM each time.

Stop when added implementation complexity exceeds the measured improvement.
The proposed deterministic occupancy fast path was **not implemented**
in Week 09; general tool decisions still go through Ollama.

## Validation and known limitations

The local test suite passed, and manual checks covered general questions,
PIR-only questions, camera-only questions, the two-sensor occupancy policy,
and a simulated camera failure. In the failure case the assistant explicitly
said the camera data was unavailable instead of fabricating a reading.

The agent **does not establish real-world occupancy with certainty**: the PIR
may miss stationary people; one camera frame may miss people or detect a
person on a screen. The system provides grounded, cautious summaries of its
limited observations, not continuous tracking or identity recognition.

Further optimisation of Ollama prompt processing, shorter speech synthesis,
and more sophisticated occupancy policies are deferred. They are **not
required to complete Week 09**.
