# Week 11 — Raspberry Pi Bluetooth speech output (Telegram stays text-only)

The Pi extension `.pi/extensions/local_speaker.ts` listens for **completed** agent answers, synthesizes locally using Piper and plays them through the **default PipeWire Bluetooth sink**. It does not send any audio to Telegram, call an LLM, or register a model-callable tool. It works for Pi TUI answers regardless of whether the question came from the terminal or Telegram; Telegram still receives only the usual text/images.

It deliberately skips Pi RPC mode because `src/week_11/voice_assistant.py` already handles Piper and playback for the microphone loop. This avoids duplicate speech in that pipeline.

## Configure Piper once

Find your existing Piper `.onnx` model used in Week 09:

```bash
cd ~/ollama_python
rg -n 'piper|\.onnx' src/week_09/voice_assistant.py
```

By default, the helper automatically uses `src/week_09/voices/en_US-lessac-medium.onnx` from this repository. No environment variables are required. To override the voice with a different `.onnx` model, set:

```bash
export PI_SPEAKER_MODEL=/absolute/path/to/your/working-voice.onnx
# No standalone Piper binary is needed: this uses Week 09's
# Python virtualenv and its `python -m piper` installation.
# PI_SPEAKER_PIPER is optional ONLY if using a standalone Piper binary.
```

Check your speaker routing:

```bash
wpctl status
wpctl inspect @DEFAULT_AUDIO_SINK@
```

If Bluetooth is connected but not the default sink, choose its output ID from `wpctl status` and run `wpctl set-default <sink-id>`. PipeWire must route the default sink to the Bluetooth speaker. If the speaker is off/not the default sink, playback is skipped automatically, and Pi/Telegram still answer normally.

Test the audio directly before launching Pi:

```bash
src/week_09/.venv/bin/python src/week_11/local_speaker.py 'Testing the Raspberry Pi Bluetooth speaker.'
```

Expected: `spoken: Bluetooth speaker` and audible speech. If you get a missing-model error, check `src/week_09/voices/` or set `PI_SPEAKER_MODEL` to the exact file. If disconnected: `skipped: Bluetooth speaker is not the default audio sink`.

## Run Pi normally

```bash
cd ~/ollama_python/src/week_11
pi --model ollama/gemma4:e2b-8k \
  --system-prompt "$PWD/voice_system.md" --no-context-files \
  --tools detect_motion,observe_scene,check_presence,query_memory,tg_attach \
  --no-mcp --tui-mode regular
```

The extension is discovered automatically as `.pi/extensions/local_speaker.ts`; no extra Pi flag is required. For every completed reply with text, the Pi terminal shows its usual answer, Telegram receives its usual text/images (if the turn came from Telegram), and **the Pi itself speaks over its Bluetooth speaker** if connected and selected as default.

If you want to temporarily disable speaker output without editing code, launch Pi with explicit extensions (`--no-extensions --extension .pi/extensions/environment.ts` plus the Telegram package path, if needed). Alternatively, power off/disconnect the Bluetooth speaker. Note that `--no-context-files` does not disable extensions.

### Tests

```bash
python3 -m unittest discover -s src/week_11/tests -p test_local_speaker.py -v
```
