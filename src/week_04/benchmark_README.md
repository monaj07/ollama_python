# Local and API model speed and accuracy benchmark

Keep all four files together. The Python entry point supports the Ollama and OpenAI SDKs. The Bash entry point uses curl plus the same Python dataset/scorer for Ollama; it requires python3 and curl but does not require the SDK. OpenAI benchmarks use the Python entry point. Install only the SDK for the provider you use.

```bash
source .venv/bin/activate
python -m pip install -U ollama
python3 ollama_benchmark.py qwen3.5:4b 3 --think off > qwen4b.json
bash ollama_benchmark.sh qwen3.5:4b 3 --think off > qwen4b_curl.json
python3 ollama_benchmark.py qwen3.5:2b 3 --think off > qwen2b.json
```

## OpenAI API example

Use an OpenAI API key with access and billing for the selected model. In the Pi's Bash shell:

```bash
source .venv/bin/activate
python -m pip install -U openai
read -rs -p 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
python3 ollama_benchmark.py gpt-6-luna 3 --provider openai --reasoning none > gpt6_luna.json
```

If OPENAI_API_KEY is already exported in this shell, skip the read/export lines. The key is read from the environment; it is not written into the script or report. --host overrides the provider endpoint (default http://localhost:11434 for Ollama, https://api.openai.com/v1 for OpenAI). The OpenAI client validates the model with a metadata request, then uses /v1/responses. SDK retries are disabled so they cannot silently inflate timing. Rate limits and other request failures remain visible in the report.

The same dataset, fresh messages, deterministic case order, scoring and token budgets apply to both providers. No web tools or API JSON/schema constraint are used. store=false is sent on Responses requests. GPT-6 Luna supports reasoning none, and that setting is the default for this backend. Temperature 0 is sent only with reasoning none; higher effort omits temperature because the model family does not support it with reasoning enabled. Use --reasoning low to test a separate reasoning configuration. --think is for Ollama only. Other OpenAI model families may not support this reasoning configuration; this API adapter is intended for models with Responses reasoning controls.

Compare accuracy and quality_summary.median_response_s across providers. speed_summary.end_to_end_tokens_per_second is available for both and equals total generated tokens divided by total complete request wall time over valid repeated speed calls. For API calls it includes network, server queueing, prompt processing and generation. It is **not generation-only throughput**. speed_summary.tokens_per_second and internal timing fields are null for OpenAI because the API does not expose Ollama's eval_duration. OpenAI's output token usage may include reasoning and other non-visible tokens; do not call the count visible answer tokens. Each API row retains usage and reasoning_tokens. Different tokenizers still limit cross-model token-rate comparisons.

## Repetitions

`MODEL 3` means one excluded initial speed request, three repeated speed requests and **one pass through the 32 accuracy questions**. Ollama unloads the selected model before the initial request; OpenAI cannot unload the server's model, so its initial request is not labelled cold and server load state is unknown. warm_runs_count keeps its name for compatibility, but means valid repeated requests after the initial call for the API. A preliminary run with `1` is optional, useful for checking setup or avoiding spending time on an unsuitable model. Three is a practical timing sample, not a statistically guaranteed sample size; add repetitions if speed varies substantially. Use `--quality-runs 3` only to check answer consistency. Repeating deterministic questions does not create more independent evidence about knowledge or reasoning.

Progress goes to stderr; stdout is a JSON report. Run models sequentially with consistent cooling and background workload. Before each model, check `ollama ps` and use `ollama stop MODEL` for other loaded models so they do not compete for RAM. Extensive reasoning can make the benchmark take much longer. One client is sufficient for model comparisons; run both only to compare client behavior. The old optional third positional prompt now overrides only the speed prompt.

## Dataset

`benchmark_cases.json` has eight cases each for knowledge, reasoning, grounding and uncertainty. It includes basic controls, multistep arithmetic, midnight rollover, logical implications, rule priority, sensor/log extraction, a malicious instruction inside a log, corrected memory, device capabilities and switching from F1 to football. The uncertainty category has four answerable and four unanswerable/false-premise cases, so always replying UNKNOWN cannot earn a perfect category score. Primary references for factual questions are stored with the cases; expected answers and references are never sent to the model.

The current dataset is `edge-ai-mini-v3`. It keeps the v2 questions and gold answers, but clarifies in the system message and every question that requested types describe the value inside `{"answer": ...}`. This also applies to UNKNOWN and NONE. Fresh comparisons must use v3 for every model; v2 and v3 prompts have different hashes and should not be mixed. Existing v2 reports can instead be rescored together without rerunning inference, as described below.

This is a screening test tailored to our Pi assistant. It is **not validated as a general ranking benchmark**. Each question changes a one-pass score by 3.125 percentage points. Do not treat a one-question lead as decisive. If models tie or nearly tie, inspect failures and add varied tasks from the intended application before choosing. Avoid changing the dataset after seeing one model's results unless all models are rerun. Factual questions are dated or stable; “best player right now” would lack an objective gold answer.

## Scores and timing

- `quality_summary.accuracy_percent`: completed, correct, strictly formatted answers as a percentage of all attempts. API errors, incomplete answers and truncations count as failed attempts; inspect the status counts to distinguish setup/runtime failures from model mistakes.
- `answer_accuracy_percent`: correct completed answer values, independent of the outer format. An exact bare value (such as `42`, `UNKNOWN` or a sensor object), a JSON code fence, or extra wrapper keys can count here while failing strict formatting. `correct_answers` records the count. Explanatory prose is not searched for a correct substring. Wrong types (for example `"10.53"` instead of `10.53`) fail. Case/whitespace, dictionary key order and listed name aliases are accepted; array order matters.
- `format_valid_percent`: output was one strict JSON object with exactly one `answer` key. Duplicate keys, NaN, Infinity, Markdown and extra keys fail. This tests instruction following; the API's JSON/schema constraint is deliberately not used.
- `false_abstentions`: UNKNOWN or NONE on an answerable case. Raw content and thinking are retained for inspecting unsupported claims. Neither the overall failure rate nor all wrong answers should be called a hallucination rate.
- `median_response_s`: complete nonempty final-answer latency, including the request. It excludes errors, truncations and incomplete/empty answers, so always read it alongside `completed_answers` and accuracy. It is not time-to-first-token. Individual wall times for failed/truncated calls remain recorded.
- `speed_summary.tokens_per_second`: Ollama-only total warm generated tokens divided by total warm generation seconds, converted from nanoseconds. API generation-only timing is unavailable and is reported as null. Truncated speed runs are valid timing samples when the request has finished and metrics are valid; failed, unfinished or zero-token/zero-duration samples are excluded under `invalid_warm_runs`. API budget exhaustion is a finished truncated request, not an unfinished HTTP request. The initial request is excluded. The min/max are Ollama generation-only rates.

The long speed prompt deliberately exceeds its 128-token cap to obtain a useful sample. It is not quality-scored. Unloading does not clear the OS disk cache. Warm speed requests repeat the prompt and may use prompt caching. Quality cases use fresh messages, except the explicit context-switch case; their deterministic shuffled order is identical across models for each round.

## Rescore saved reports without inference

The scorer version is `typed-answer-values-v2`. Earlier versions undercounted correct bare values in `answer_accuracy_percent`, because they required an `answer` wrapper even for that metric. Strict `accuracy_percent` and `format_valid_percent` keep their existing meanings.

```bash
python3 ollama_benchmark.py --rescore gpt6_luna.json > gpt6_luna_rescored.json
python3 ollama_benchmark.py --rescore qwen4b.json > qwen4b_rescored.json
```

Use a different output filename from the input. This requires only Python's standard library: no API key, SDK or running Ollama server. It checks case IDs and expected values against the dataset, updates row scores and quality summaries, and preserves original responses, timings, speed results and dataset identity. `scorer_version`, `rescored` and `scoring_dataset_sha256` identify the rescoring operation. Rescore every model's saved report with the same scorer when comparing the original experiment.

## Thinking and fairness

`--think auto` disables thinking only when `/api/show` advertises support for false; otherwise it leaves the model default. `think_sent: null` means the parameter was omitted. Explicit unsupported on/off settings are rejected when the server supplies capability metadata. Older servers may omit metadata; use `--think off` explicitly for models known to support it. Mandatory-thinking models retain their thinking.

Defaults: context 2048, temperature 0, speed budget 128 and quality budget 512 tokens. The reasoning and final answer share the generation budget. If reasoning exhausts it, the task is recorded as truncated rather than as proof of missing knowledge. Use `--tokens 1024` equally across models to compare with a larger budget. Speed tokens include thinking when present, so a model can generate quickly yet take longer to give a usable answer. Compare latency and accuracy as well as tokens/sec. Different tokenizers also make tokens/sec an approximate cross-model metric.

Keep dataset hash, budgets and thinking policy consistent. The report records provider, model metadata and Ollama version/quantization where available. OpenAI's context size cannot be configured with Ollama's num_ctx, so that setting is null for API runs; actual input messages are identical and short. Record Pi configuration and temperature separately. A loaded-model speed result is not end-to-end agent latency. No tool calling, live retrieval, camera input or hardware control is tested here.

```bash
jq '{model, ollama_version, settings, speed_summary, quality_summary, by_category}' qwen4b.json
jq '.quality_runs[] | select(.passed == false) | {id, status, correct, valid_format, content, thinking, error}' qwen4b.json
```

Implementation verification: syntax checks and local mocked HTTP tests of SDK/curl request parity, grading, repeats, thinking controls, nanosecond timing, truncation, errors and path handling. The API extension passed 219 mocked HTTP requests covering both providers, matching prompts/order/scoring, effort/temperature controls, token limits, rate limits, refusals and credential validation. Regression checks cover the nine reported failures, bare typed values, strict JSON validation and offline rescoring with preserved inputs/timing and rejected gold mismatches. No paid API requests were made during verification. Actual model accuracy and performance still need measurement.

API references: https://docs.ollama.com/api/chat, https://docs.ollama.com/capabilities/thinking, https://developers.openai.com/api/docs/models/gpt-6-luna, https://developers.openai.com/api/docs/guides/latest-model and https://developers.openai.com/api/docs/guides/token-counting.
