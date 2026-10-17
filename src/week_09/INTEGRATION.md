# Week 09, Step 5: agentic physical tools

These four Python files go into `~/ollama_python/src/week_09/`:

- `agent.py` — bounded Ollama loop, per-turn tracing and tool validation.
- `tools.py` — strict dispatcher; executes earlier weeks using their existing venvs.
- `pir_bridge.py` — imports Week 06 `MotionTools`, reads GPIO23 and closes it.
- `vision_bridge.py` — imports Week 07–08 `SceneObserver`, returns a compact result.

**Keep your existing, locally modified `voice_assistant.py`; do not overwrite it.**

## One change to your existing `voice_assistant.py`

Near the imports add:

```python
from agent import VoiceAgent
```

Replace its old `ask_agent()` implementation with:

```python
_agent: VoiceAgent | None = None


def ask_agent(question: str, model: str) -> str:
    global _agent
    if _agent is None or _agent.model != model:
        _agent = VoiceAgent(model=model)
    return _agent.ask(question)
```

The previous `with timed('llm_s', ...)` around `ask_agent(...)` in `process_turn()` stays unchanged. The existing warm-up MUST use `Client.chat(messages=[], think=False, options={'temperature': 0, 'num_ctx': 2048, 'num_predict': 100}, keep_alive='15m')`, matching the agent's settings. Do not change your text, audio, record, transcribe, Piper or Bluetooth functions.

## Dependencies and preparation

From the repository root, confirm both earlier virtual environments remain present:

```bash
ls -l src/week_06/.venv/bin/python src/week_07_08/.venv-vision/bin/python
```

Ensure the PIR is wired to GPIO23 per Week 06, that the sensor has settled for about 60 seconds after powering on, and the camera works with its earlier test command. Do not run an earlier PIR script at the same time as the voice agent.

No additional pip dependencies are needed in Week 09 for GPIO or vision because `tools.py` invokes the prior environments. Ollama is still required in Week 09.

## Test stages

```bash
cd ~/ollama_python
source src/week_09/.venv/bin/activate

# Baseline: no physical tools expected.
python src/week_09/voice_assistant.py --text 'What is 2 plus 17?' --no-play

# PIR tool; the model should call detect_motion.
python src/week_09/voice_assistant.py --text 'Is motion detected right now?' --no-play

# Camera tool; the model should call observe_scene.
python src/week_09/voice_assistant.py --text 'Can you see a cup?' --no-play

# Both PIR and camera are expected.
python src/week_09/voice_assistant.py --text 'Is anybody in the room?' --no-play

# Finally: spoken tests, multiple turns in one process.
python src/week_09/voice_assistant.py --mic-device plughw:0,0 --duration 5
```

Inspect tool decisions and results:

```bash
tail -n 20 src/week_09/outputs/tool_events.jsonl
```

**Important limitations:** tool use is selected by a small model and is not guaranteed merely by prompt instructions. If it answers environmental questions without tools, that is a failed routing test. A tool failure returns structured `ok:false` data, and the assistant should admit evidence was unavailable. The sensor is only a PIR signal; the camera sees one limited field of view and YOLOE searches only selected labels.

**Performance:** camera runs in a subprocess for interpreter isolation. This means YOLOE is recreated on each tool call; it is reliable for first integration, not optimal latency. Its memory usage and possible Ollama eviction must be benchmarked before building a persistent vision worker.

## Adapter smoke tests (before running the full voice assistant)

Run these from the repository root after copying the new files:

```bash
# Uses existing Week 06 venv and GPIO23; do not run other PIR scripts concurrently.
printf '{}' | GPIOZERO_PIN_FACTORY=lgpio src/week_06/.venv/bin/python src/week_09/pir_bridge.py

# Uses existing Week 07–08 vision venv; first run can be slow.
printf '{"targets":["person"],"conf":0.2}' | src/week_07_08/.venv-vision/bin/python src/week_09/vision_bridge.py

# Run hardware-independent tests from Week 09 venv.
python -m unittest discover -s src/week_09/tests -v
```

If the first returns `{ "ok": false, ... }`, inspect Week 06 GPIO wiring and `.venv` installation. If the second fails, run the existing Week 07–08 observation command directly; don't reinstall the whole vision stack into Week 09.
