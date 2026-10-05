# Small-model speed and accuracy benchmark (reviewed v2)

Keep all four files together. The Python entry point uses the Ollama SDK. The Bash entry point uses curl plus the same Python dataset/scorer; it requires python3 and curl but does not require the SDK. Ollama and the models must already be installed.

```bash
source .venv/bin/activate
python -m pip install -U ollama
python3 ollama_benchmark.py qwen3.5:4b 3 --think off > qwen4b.json
bash ollama_benchmark.sh qwen3.5:4b 3 --think off > qwen4b_curl.json
python3 ollama_benchmark.py hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M 3 > lfm12b.json
python3 ollama_benchmark.py hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M 3 > lfm26b.json
```

## Repetitions

`MODEL 3` means one unloaded-model speed request, three warm speed requests and **one pass through the 32 accuracy questions**. A preliminary run with `1` is optional, useful for checking setup or avoiding spending time on an unsuitable model. Three is a practical timing sample, not a statistically guaranteed sample size; add repetitions if speed varies substantially. Use `--quality-runs 3` only to check answer consistency. Repeating deterministic questions does not create more independent evidence about knowledge or reasoning.

Progress goes to stderr; stdout is a JSON report. Run models sequentially with consistent cooling and background workload. Before each model, check `ollama ps` and use `ollama stop MODEL` for other loaded models so they do not compete for RAM. Extensive reasoning can make the benchmark take much longer. One client is sufficient for model comparisons; run both only to compare client behavior. The old optional third positional prompt now overrides only the speed prompt.

## Dataset

`benchmark_cases.json` has eight cases each for knowledge, reasoning, grounding and uncertainty. It includes basic controls, multistep arithmetic, midnight rollover, logical implications, rule priority, sensor/log extraction, a malicious instruction inside a log, corrected memory, device capabilities and switching from F1 to football. The uncertainty category has four answerable and four unanswerable/false-premise cases, so always replying UNKNOWN cannot earn a perfect category score. Primary references for factual questions are stored with the cases; expected answers and references are never sent to the model.

This is a screening test tailored to our Pi assistant. It is **not validated as a general ranking benchmark**. Each question changes a one-pass score by 3.125 percentage points. Do not treat a one-question lead as decisive. If models tie or nearly tie, inspect failures and add varied tasks from the intended application before choosing. Avoid changing the dataset after seeing one model's results unless all models are rerun. Factual questions are dated or stable; “best player right now” would lack an objective gold answer.

## Scores and timing

- `quality_summary.accuracy_percent`: completed, correct, strictly formatted answers as a percentage of all attempts. API errors, incomplete answers and truncations count as failed attempts; inspect the status counts to distinguish setup/runtime failures from model mistakes.
- `answer_accuracy_percent`: correct completed answer values, accepting an outer JSON code fence or extra object keys for this metric only. Those still fail strict formatting. Free prose is not automatically judged. Wrong types (for example `"10.53"` instead of `10.53`) fail. Case/whitespace, dictionary key order and listed name aliases are accepted; array order matters.
- `format_valid_percent`: output was one strict JSON object with exactly one `answer` key. Duplicate keys, NaN, Infinity, Markdown and extra keys fail. This tests instruction following; the API's JSON/schema constraint is deliberately not used.
- `false_abstentions`: UNKNOWN or NONE on an answerable case. Raw content and thinking are retained for inspecting unsupported claims. Neither the overall failure rate nor all wrong answers should be called a hallucination rate.
- `median_response_s`: complete nonempty final-answer latency, including the request. It excludes errors, truncations and incomplete/empty answers, so always read it alongside `completed_answers` and accuracy. It is not time-to-first-token. Individual wall times for failed/truncated calls remain recorded.
- `speed_summary.tokens_per_second`: total warm generated tokens divided by total warm generation seconds. Ollama durations are converted from nanoseconds. Truncated speed runs are valid samples if the response is done and contains valid timing metrics; incomplete, failed or zero-token/zero-duration runs are excluded and counted under `invalid_warm_runs`. The initial loading request is excluded from warm speed. The min/max reveal spread across runs.

The long speed prompt deliberately exceeds its 128-token cap to obtain a useful sample. It is not quality-scored. Unloading does not clear the OS disk cache. Warm speed requests repeat the prompt and may use prompt caching. Quality cases use fresh messages, except the explicit context-switch case; their deterministic shuffled order is identical across models for each round.

## Thinking and fairness

`--think auto` disables thinking only when `/api/show` advertises support for false; otherwise it leaves the model default. `think_sent: null` means the parameter was omitted. Explicit unsupported on/off settings are rejected when the server supplies capability metadata. Older servers may omit metadata; use `--think off` explicitly for models known to support it. Mandatory-thinking models retain their thinking.

Defaults: context 2048, temperature 0, speed budget 128 and quality budget 512 tokens. The reasoning and final answer share the generation budget. If reasoning exhausts it, the task is recorded as truncated rather than as proof of missing knowledge. Use `--tokens 1024` equally across models to compare with a larger budget. Speed tokens include thinking when present, so a model can generate quickly yet take longer to give a usable answer. Compare latency and accuracy as well as tokens/sec. Different tokenizers also make tokens/sec an approximate cross-model metric.

Keep dataset hash, budgets and thinking policy consistent. The report records Ollama version and model quantization details. Record Pi configuration and temperature separately. A loaded-model speed result is not end-to-end agent latency. No tool calling, live retrieval, camera input or hardware control is tested here.

```bash
jq '{model, ollama_version, settings, speed_summary, quality_summary, by_category}' qwen4b.json
jq '.quality_runs[] | select(.passed == false) | {id, status, correct, valid_format, content, thinking, error}' qwen4b.json
```

Implementation verification: syntax checks and local mocked HTTP tests of SDK/curl request parity, grading, repeats, thinking controls, nanosecond timing, truncation, errors and path handling. Actual model accuracy and performance still need to be measured on the Pi.

API references: https://docs.ollama.com/api/chat and https://docs.ollama.com/capabilities/thinking.
