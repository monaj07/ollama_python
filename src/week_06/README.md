# Week 06 — Environment agent: motion sensing

A local language model chooses when to read a Raspberry Pi PIR motion sensor
through a validated `detect_motion()` tool, then explains the observation.
The current implementation uses a Raspberry Pi 5 (8 GB) and defaults to
`gemma4:e2b` through Ollama.

## Objective and learning points

Extend software tool calling into observations of the physical environment.
Week 05 used a button to trigger an LED action; this week a typed question
lets the model decide whether it needs a sensor reading before answering.

- Separate GPIO access from the model's reasoning through a small tool interface.
- Request observations when relevant instead of sending continuous sensor streams
  to the model.
- Validate tool names, arguments, and call counts before reading hardware.
- Preserve observation timestamps and distinguish sensor evidence from inference.
- Bound the tool loop and trace each request, reading, answer, and failure.

The central lesson is that a valid reading still needs careful interpretation.
**A PIR detects changes in infrared radiation associated with movement; it does
not reliably establish whether a room is occupied or empty.** A stationary
person can produce an inactive signal, and a signal can remain active after
movement stops because of the sensor's hardware hold time.

## Hardware and wiring

- Raspberry Pi 5 with Raspberry Pi OS, power supply, and local Ollama.
- HC-SR501-style PIR module with a 3.3 V output signal and a 5 V supply input.
- Breadboard and six male-to-female jumper wires, or three suitable direct wires.

Shut down with `sudo poweroff`, wait for shutdown, and disconnect power before
changing wiring. Identify **VCC, OUT, and GND for the actual module** from its
labels or documentation; pin order varies, and viewing the other side reverses
left and right. Do not assume an unlabeled module's pin order from this table.
The Pi's GPIO inputs use 3.3 V logic: **never connect the 5 V supply to GPIO23**.

| PIR connection | Pi connection | Physical header pin | BCM number |
|---|---|---:|---:|
| VCC | 5 V supply | 2 | — |
| OUT | GPIO input | 16 | 23 |
| GND | Ground | 14 | — |

`MotionSensor(23)` uses BCM numbering, so it refers to physical pin **16**.
Run `pinout` on Raspberry Pi OS to inspect the header reference.

### Breadboard connection used in this experiment

Each connection uses two male-to-female wires: one from the Pi to the
breadboard, and one from the breadboard to the PIR.

| Pi-side hole and wire | Sensor-side hole and wire |
|---|---|
| a28 ← Pi physical pin 2 (5 V) | c28 → PIR VCC |
| a29 ← Pi physical pin 16 (GPIO23) | c29 → PIR OUT |
| a30 ← Pi physical pin 14 (GND) | c30 → PIR GND |

For each number, a–e share one electrical connection. Thus a28 and c28 connect,
but rows 28, 29, and 30 remain separate. Use otherwise unused rows and keep
exposed pins from touching adjacent contacts.

The existing Week 05 LEDs and button can remain connected. These scripts only
use GPIO23; they do not read the button or control either LED.

## Files

| File | Purpose |
|---|---|
| `check_motion.py` | Hardware-only test; prints the motion state every 0.5 seconds |
| `motion_tools.py` | `MotionTools` wrapper and interactive sensor-tool check |
| `motion_agent.py` | Ollama tool loop, validation, CLI, timing, and JSONL tracing |

All three entry points allow 60 seconds for the PIR to settle before use.
Run only one sensor script at a time so GPIO23 is not claimed by multiple
processes. Stop the current script with Ctrl+C before starting the next one.

## Setup on the Pi

Run these commands on the Pi from the repository root:

```bash
cd ~/ollama_python
sudo apt update
sudo apt install -y python3-venv python3-gpiozero python3-lgpio
python3 -m venv --system-site-packages src/week_06/.venv
source src/week_06/.venv/bin/activate
python -m pip install ollama 'pydantic>=2,<3'
export GPIOZERO_PIN_FACTORY=lgpio
ollama pull gemma4:e2b
ollama list
```

`--system-site-packages` exposes the apt-installed GPIO packages inside the
virtual environment. The `lgpio` backend supports GPIO access on the Pi 5.
The user running the scripts needs GPIO access, normally through the `gpio`
group. Ollama must already be installed and serving at
`http://127.0.0.1:11434`.

For each new terminal, activate the environment and set the pin factory again:

```bash
cd ~/ollama_python
source src/week_06/.venv/bin/activate
export GPIOZERO_PIN_FACTORY=lgpio
```

The sensor checks require only GPIO Zero; the agent additionally needs Ollama's
Python SDK and Pydantic 2. No I2C configuration is needed for the PIR.

## Run the stages in order

### 1. Hardware-only check

```bash
python src/week_06/check_motion.py
```

After the startup wait, expect output such as:

```text
18:32:33 | motion_signal_active=False
18:32:35 | motion_signal_active=True
```

Walk sideways across the field of view, approximately 1–2 metres away, then
stand still and wait for the output to return to `False`. A whole hand moving
sideways at 30–50 cm is another useful check. Small finger movements a few
centimetres from the lens are not a reliable sensitivity test.

The delay control changes how long the output stays active; it is not Python
latency. Start with a short hold time, following the module's documentation.
Sensitivity and retrigger mode also affect readings. Allow for the module's
blocking interval after the output goes inactive; waiting about 10 seconds
between separate checks worked in this experiment.

### 2. Sensor-tool check

```bash
python src/week_06/motion_tools.py
```

Press Enter to request a reading. The result is a Python dictionary with this
structure (shown here as JSON):

```json
{
  "ok": true,
  "observed_at": "2026-10-09T01:00:00+00:00",
  "motion_signal_active": false
}
```

The timestamp is in UTC. Each invocation reads the current GPIO Zero motion
state; the wrapper does not maintain a movement history.

### 3. Local sensor agent

```bash
python src/week_06/motion_agent.py
```

Wait for `Ready`, then type a question. Relevant requests should print a
`[tool] detect_motion({})` line and a `[result]` before the final answer.
Type `quit` or `exit`, or press Ctrl+C, to stop and close the GPIO resource.

| Check | Expected behaviour |
|---|---|
| “Is movement detected right now?” while moving | Calls the tool and describes an active signal if movement is captured |
| Same question after the signal settles | Calls the tool and describes an inactive signal |
| “Is the room empty?” | Calls the tool and explains that PIR evidence cannot establish emptiness |
| “What is 12 + 7?” | Answers 19 without calling the sensor tool |

For the active check, keep moving until the `[result]` appears. **The observation
is taken when Python executes the tool, not when you submit the question.**
Inference can take time, so a brief initial movement may end before the reading.
The final answer may also arrive after the state has changed.

The hardware and these four agent checks were reported as working during the
experiment. They are manual checks, not an automated accuracy benchmark or a
guarantee that every future model response will follow the instructions.

## Agent behaviour and validation

Each question starts a fresh message list. The model receives the tool schema,
chooses whether to call it, and receives the result through a tool message before
generating an answer. Previous questions and readings are not carried forward.

`MotionArguments` is an empty Pydantic model with strict validation and extra
fields forbidden. Python accepts only `detect_motion`, with no arguments and
exactly one call in a tool round. The model cannot choose a GPIO pin or execute
arbitrary Python. Invalid names, arguments, or multiple calls fail the request.

The loop permits at most three tool rounds, followed by a final model request
with no tools offered: at most four model requests per user question. Sensor
read exceptions become `{"ok": false, "error": "..."}` results. Other request
errors are logged and printed, and the CLI returns to the input prompt.

The prompt instructs the model to obtain fresh evidence for motion or occupancy
questions, explain unavailable readings, and answer in at most three short
sentences. These are model instructions; Python does not independently enforce
tool use for every relevant question or the correctness of the final prose.

### Current configuration

| Setting | Value |
|---|---|
| Model | `gemma4:e2b` |
| Ollama host | `http://127.0.0.1:11434` |
| Client timeout | 180 seconds |
| Temperature | 0 |
| Context size | 2048 tokens |
| Output cap | 256 tokens per model response |
| Thinking | Disabled (`think=False`) |
| Keep-alive | 10 minutes |
| Maximum tool rounds | 3 |

These settings are defined in `motion_agent.py`; the script does not read a
`MODEL` environment variable. To try `qwen3.5:4b`, download it with
`ollama pull qwen3.5:4b` and edit the `MODEL` constant. Re-run the same checks
after changing models.

## Traces and timing

The agent appends one JSON object per event to
`src/week_06/logs/events.jsonl`, regardless of the current working directory.
Generated logs are ignored by Git.

```bash
tail -n 12 src/week_06/logs/events.jsonl
tail -f src/week_06/logs/events.jsonl
```

| Event | Contents |
|---|---|
| `user_request` | Question and request ID |
| `model_response` | Round number and raw assistant message, including any tool calls |
| `tool_call` | Accepted tool name and arguments, logged after validation |
| `tool_result` | Observation or sensor error |
| `final_answer` | Final text returned to the user |
| `request_error` | Exception type and message |
| `request_interrupted` | Interruption during a request |
| `request_finished` | Total elapsed time for that request |
| `user_exit` | Exit reason; request ID is null |

Match events by `request_id` to reconstruct a request. Event timestamps and
sensor observation timestamps use UTC. `[elapsed]` and `elapsed_s` cover the
request, including model rounds, tool execution, and logging; they exclude the
initial 60-second settling wait and time spent typing the question.
They are not token-generation speed measurements.

## Troubleshooting and references

- **Signal remains active:** check the hardware hold-time and retrigger settings.
  The script does not impose a long software hold period.
- **Intermittent detection:** test broad sideways movement after settling,
  allow time between triggers, and check sensitivity and the field of view.
- **GPIO busy:** stop the other sensor process before starting a new one.
- **Import or pin-factory error:** activate the Week 06 environment, confirm the
  apt GPIO packages are installed, and set `GPIOZERO_PIN_FACTORY=lgpio`.
- **Ollama connection or model error:** check that the server is running locally
  and the configured model appears in `ollama list`.

- [GPIO Zero MotionSensor API](https://gpiozero.readthedocs.io/en/stable/api_input.html#motionsensor-d-sun-pir)
- [GPIO Zero pin factories](https://gpiozero.readthedocs.io/en/stable/api_pins.html)
- [Raspberry Pi GPIO reference](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#gpio-and-the-40-pin-header)
- [Adafruit PIR testing and adjustment](https://learn.adafruit.com/pir-passive-infrared-proximity-motion-sensor/testing-a-pir)
- [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling)
- [Pydantic model configuration](https://docs.pydantic.dev/latest/concepts/config/)
