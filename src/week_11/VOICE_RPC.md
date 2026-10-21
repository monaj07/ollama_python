# Week 11 — Pi RPC voice experiment

Purpose: reuse the **working Week 09 audio path** (USB mic → whisper.cpp → Piper → Bluetooth speaker) while replacing the Week 10 Python LLM/router with **one long-running Pi RPC process**. Week 10 remains unchanged as the benchmark baseline.

The Week 11 bridge starts Pi from `src/week_11/` with the **compact `voice_system.md` system prompt** (matching our successful RPC ablation). Pi still loads project `AGENTS.md` and `.pi/extensions/environment.ts` unless `--pi-no-context-files` is specified. It allows only `detect_motion`, `observe_scene`, `check_presence`, `query_memory` and disables MCP for these controlled tests. Pi owns all LLM calls, conversation sessions and tool decisions. The Python process only handles audio I/O and JSONL transport.

## A. Text-first RPC smoke test (no microphone or TTS required)

From the repository root on Raspberry Pi:

```bash
cd ~/ollama_python
python3 -m unittest discover -s src/week_11/tests -p 'test_pi_rpc.py' -v
python3 src/week_11/pi_rpc.py --no-session --system-prompt src/week_11/voice_system.md --prompt 'What is 2 + 17?'
```

Expected: a completed answer of `19`; no JSONL protocol noise. The tool list should be empty. First use may be slower if Ollama is cold.

Next, **keep a single Pi process alive**:

```bash
python3 src/week_11/pi_rpc.py --no-session --system-prompt src/week_11/voice_system.md
```

Ask (one line per prompt):

- `What is 2 + 17?` — no tool
- `How many person-positive detection events are saved?` — `query_memory` operation=summary
- `What did you see most recently?` — `query_memory` operation=recent
- `Is there anyone in the room right now?` — `check_presence`, PIR and camera; a new observation is saved
- `What about the last observation?` — contextual follow-up without assuming a new capture

Enter `q` to exit. Each printed `[pi_rpc]` record reports **agent elapsed time**, tool names, measured tool durations, and model tokens. Check that answer text doesn't include reasoning blocks or tool JSON. Avoid running a separate Away Mode camera monitor simultaneously.

## B. Reuse Piper with text input (no microphone)

Run using our existing Week 09 voice Python environment:

```bash
src/week_09/.venv/bin/python src/week_11/voice_assistant.py \
  --text 'What is 2 + 17?' --no-play
```

This still runs Piper TTS and writes a reply WAV, but doesn't play through Bluetooth. The voice wrapper selects the project-local compact prompt by default; no temporary `/tmp` prompt file is required. If you wish to compare with/without project context, add `--pi-no-context-files` to the voice command. The output defaults to `outputs/week_11/voice/` (ignored by Git).

## C. Push-to-talk voice

```bash
src/week_09/.venv/bin/python src/week_11/voice_assistant.py \
  --mic-device plughw:0,0 --duration 5
```

Press Enter to record and `q` to quit. The existing timing log reports `record_s`, `stt_s`, `llm_s`, `tts_s`, `play_s`, `total_s`; Pi adds `[pi_rpc] elapsed_s`, selected tools, and tool-duration fields. The first request may include Ollama and YOLOE warm-up. Repeat the same sensor query before comparing warm timings with Week 10.

Pi session storage is persistent by default. To resume the most recent project Pi session (e.g., after restarting the voice app), add `--pi-continue`. This does **not** import Week 10's old conversation store; only environmental SQLite history is shared.

## Design and safety notes

- `--no-mcp` and the tool allowlist are a **controlled voice/remote-agent profile**, not the full Pi development environment. Use the normal Pi TUI separately to explore its built-in coding tools.
- The RPC client waits for `agent_settled`, not merely prompt acceptance or `agent_end`. It reads the authoritative `message_end` text, ignoring tool-call, reasoning, and streaming delta blocks. It refuses to speak incomplete `length`, `error`, or `aborted` responses.
- Pi RPC uses one child process and preserves context between consecutive turns. It does not start Ollama independently or duplicate the Python hardware bridge. Initial loading time is included in the first Pi agent turn.
- RPC stderr and errors are shown in the terminal, not synthesized into speech. After a timeout/error, the RPC process is closed to avoid mixing responses from different turns.
- This is **not yet** a Telegram daemon or multi-client Pi gateway; voice and TUI are separate processes with separate live conversations. We should coordinate sessions before adding Telegram.
- Voice WAV files, transcriptions, Pi session logs and camera evidence may contain sensitive information; don't commit them. The default Week 11 voice output is in the repository's ignored `outputs/` tree.

## Compare with Week 10

For a useful comparison, use the same set of general, live-presence, memory and contextual follow-up questions. Record warm-up separately, report median of ≥3 warm runs, and compare **speech-to-speech `total_s`** rather than only the `llm_s` stage. Inspect actual tool calls and groundedness, not just whether a final sentence sounds plausible.

Pi RPC wire format and lifecycle: <https://pi.dev/docs/latest/rpc> and <https://pi.dev/docs/latest/json>.
