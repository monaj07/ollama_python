# Week 13 — Prompt Caching and Context Management for an Edge Agent

**Learning track:** Module 3.4–3.5, local inference optimisation.
**Hardware:** Raspberry Pi 5 (8 GB RAM), CPU-only Ollama / llama.cpp.
**Measured:** 27–28 October 2026.
**Main model:** `hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M`.

> **Finding:** Keeping a model in RAM is different from reusing computed prompt
> state. On our Pi, repeating an identical request reused **270/274 tokens** and
> cut prefill from **5.063 s to 0.094 s**. Extending a conversation reused
> **310/330 tokens**. In a context-management experiment, full conversation
> history was **faster** to prefill than truncated history because most tokens
> were already cached—and full history preserved a fact the truncated version
> could no longer answer.

This week is about **prompt-prefix reuse, recurrent-model cache behaviour, and
when to compact an agent's active conversation**, not weight or KV precision
benchmarking. For the latter, see Module 3's earlier F16/Q8/Q4 experiments.

## Contents

1. [Run the experiments](#run-the-experiments)
2. [What is being cached?](#what-is-being-cached)
3. [Prompt caching: four controlled probes](#prompt-caching-four-controlled-probes)
4. [Context management: three policies](#context-management-three-policies)
5. [Practical policy for the Raspberry Pi agent](#practical-policy-for-the-raspberry-pi-agent)
6. [Gotchas and limitations](#gotchas-and-limitations)
7. [Source files and next work](#source-files-and-next-work)

## Run the experiments

Use a working local Ollama service and an installed LFM2.5 model. These scripts
use only Python's standard library, and **do not access GPIO, camera, PIR,
SQLite or personal files**.

```bash
cd ~/ollama_python
ollama --version
ollama list
ollama ps

# Offline tests first; Ollama is NOT needed for these tests.
python3 -m unittest discover -s src/week_13/tests -p 'test_*.py' -v
```

Reproduce the prefix-cache probes separately:

```bash
python3 src/week_13/prefix_cache_repro.py --experiment changed-short
python3 src/week_13/prefix_cache_repro.py --experiment identical
python3 src/week_13/prefix_cache_repro.py --experiment continuation
python3 src/week_13/prefix_cache_repro.py --experiment changed-long
```

Or run them consecutively and save the **new** measurements outside the source
folder (each experiment resets the runner context independently):

```bash
mkdir -p outputs/week_13
python3 src/week_13/prefix_cache_repro.py --experiment all \
  --output outputs/week_13/prefix_cache.json
```

Measure the three context-management policies:

```bash
python3 src/week_13/context_policy_repro.py \
  --output outputs/week_13/context_policies.json
```

Both scripts default to the actual local Ollama model tag used during these
sessions. Override it with `--model` if necessary. The model should be present
in `ollama list`. Both scripts accept `--base-url` to target a different Ollama
server. **Do not run a different model and compare its exact cache counts to the
recorded LFM results as if the runtime were identical.**

To examine current model residency:

```bash
ollama ps
```

### Common benchmark settings

The Python probes use `num_ctx=2048`, `num_thread=2`, `temperature=0`,
`seed=42`, `think=false`, `stream=false`, `keep_alive=10m`. The prefix probes
use `num_predict=16` for simple repetitions and 48 for the continuation case;
context-policy tests use 48. The runner is explicitly unloaded, then warmed
with an empty generation request before each independent experiment. This
excludes the **initial model load** from the reported chat prefill time.

Recorded observations came from **Ollama 0.40.6**, with the Pi running
CPU-only. Different Ollama/llama.cpp versions, number of parallel runners,
model tags, chat templates or runtime cache state can change results.

## What is being cached?

These three concepts are often confused:

| Mechanism | What it saves | What it does *not* guarantee |
| --- | --- | --- |
| **Model residency** (`keep_alive`) | Avoids reloading model weights from storage | Prompt token computations need not be reused |
| **Prompt-prefix cache** | Reuses retained attention KV and/or compatible recurrent computation state for an identical initial token sequence | Arbitrary edits to old turns can invalidate reuse |
| **Agent memory** (e.g., SQLite) | Stores historical events and facts across restarts | Does not itself make prompt prefill cheaper |

Prompt caching means computation from a **compatible previous prefix** can be
reused. It is not merely a textual cache of the last answer, and it is not
synonymous with weight quantisation or KV-cache **precision**.

Relevant Ollama response fields:

```json
{
  "prompt_eval_count": 330,
  "prompt_eval_cached_count": 310,
  "prompt_eval_duration": 410000000,
  "eval_count": 0,
  "eval_duration": 0,
  "load_duration": 0,
  "total_duration": 0
}
```

The example above illustrates **field names and units**, not a real complete
response: duration fields are in **nanoseconds**; zeroes are placeholders.
`prompt_eval_cached_count` reports reused prompt tokens;
`prompt_eval_count - prompt_eval_cached_count` is the number not served from
cache. Prefill should be evaluated separately from token decoding and cold
model-loading time. An actual cached-token count can be nonzero without a
strictly proportional latency improvement.

## Prompt caching: four controlled probes

**Provenance:** Transcribed from the terminal output of the original interactive
experiments (rather than invented/re-simulated). Raw measurements are also in
[`results/observed_2026-10-28.json`](results/observed_2026-10-28.json).
There was **one observed run** per case; no uncertainty intervals.

### Probe A — Shared short system prefix, changed user query

A shared system prompt (15 repeated safety rules), with only the final user
request changed in B. C also changed the beginning of the system prompt.

| Case | Prompt tokens | Cached | Prefill | Load |
| --- | ---: | ---: | ---: | ---: |
| A — initial | 283 | 0 | 5.231 s | 0.001 s |
| B — same system prefix, different question | 283 | **0** | 5.231 s | 0.001 s |
| C — system prefix changed | 281 | 0 | 5.165 s | 0.002 s |

**Surprise:** Sharing the entire system prefix did **not** guarantee reuse.
The almost-zero load time confirms the model was resident, not that context
state was reusable. The model also did not reliably follow the `READY/DONE/OK`
format instructions; agent instruction-following should be evaluated separately.

### Probe B — Byte-for-byte identical request repeated

| Case | Prompt tokens | Cached | Prefill |
| --- | ---: | ---: | ---: |
| First | 274 | 0 | 5.063 s |
| Repeated identical request | 274 | **270** | **0.094 s** |

Reusing **98.5%** of the prompt reduced measured prefill by approximately
**54×**. This directly establishes that **prompt caching works** on the Pi;
the earlier miss cannot be explained as an entirely disabled cache.

### Probe C — Growing conversation, append new turn

The second API call included the original system and user messages, then the
**real assistant reply**, then a new user question.

| Turn | Prompt tokens | Cached | Prefill |
| --- | ---: | ---: | ---: |
| First | 263 | 0 | 4.866 s |
| Second | 330 | **310** | **0.410 s** |

The continuing conversation reused **93.9%** of the second prompt even though
its total token count grew. The cached count exceeded the first prompt length
because the previous assistant response had also become part of the context.

**Implication for agents:** Append prior assistant and tool messages as stable
history instead of reconstructing/editing them unnecessarily. This creates
opportunities for efficient continuation, subject to runtime behaviour.

### Probe D — Longer shared prefix, changed user query

Identical structure to Probe A but with 64 repeated rules in the system prefix.

| Case | Prompt tokens | Cached | Uncached | Prefill |
| --- | ---: | ---: | ---: | ---: |
| A — initial | 1,116 | 0 | 1,116 | 21.272 s |
| B — different question | 1,116 | **604** | **512** | **10.060 s** |

The longer shared prefix enabled **partial reuse** and reduced prefill by
**52.7%**. This disproves the idea that changing the user query necessarily
forces total recomputation.

**Explanation and its limit:** LFM2.5 includes recurrent processing as well as
attention. A runtime may need a reusable recurrent-state checkpoint to roll
back to a common prefix before processing a changed suffix. The conspicuous
**512-token uncached tail** is consistent with checkpoint/chunk effects.
However, we did **not** trace the checkpoint implementation of our particular
Ollama runner; these experiments alone **do not prove** the 512-token policy,
its exact mechanism, or a universal threshold across versions/models. Do not
pad short prompts artificially to 512 tokens on this evidence: first-request
compute, memory, and latency may increase more than subsequent savings.

### Comparison: what the probes actually establish

| Request pattern | Observed cache reuse | What we know |
| --- | --- | --- |
| Short common prefix + new user question | None | Prefix equality alone was insufficient |
| Exact same request repeated | Very high | Existing compatible prompt state was reused |
| Existing conversation extended | Very high | Appending can preserve reusable state |
| Longer common prefix + changed question | Partial | Some previously computed state could be reused |

**Open investigation:** To conclusively explain the short-prefix miss at the
implementation level, collect runner-level checkpoint decisions for this
specific Ollama/llama.cpp build. We deliberately stop short of claiming that
our tests have established the exact backend rule.

## Context management: three policies

The experiment injected a synthetic maintenance code (`VEX-7319-LUMA`) early in
a conversation, then added **ten routine status updates**. The model received
an initial chat request with the complete conversation so we could test
subsequent prefix reuse. Before each policy, we unloaded/reloaded the runner
to avoid cross-policy cache contamination.

For the second chat request:

- **Full:** Preserve the original history, previous answer and new question.
- **Recent:** Drop older messages and keep the last two routine exchanges,
  previous answer and new question.
- **Summary + recent:** Replace old dialogue with a **manually authored**
  summary containing the code; keep the recent exchanges.

| Policy | Prompt | Cached | Uncached | Prefill | Decode | Recalled code? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **Full history** | 556 | **528** | **28** | **0.605 s** | 0.512 s | **Yes** |
| Recent only | 170 | 13 | 157 | 2.891 s | 0.176 s | No; returned `UNKNOWN` |
| Summary + recent | 200 | 13 | 187 | 3.463 s | 0.471 s | **Yes** |

**Unexpected result:** Full history was approximately **4.8× faster** to
prefill than the recent-only strategy, even though its prompt had more than
three times as many tokens. It only needed to evaluate 28 uncached tokens.

**Interpret correctness carefully:** `UNKNOWN` was an appropriate grounded
response when the model had no access to the old code. The **policy** lost
important information; this is not evidence of the model hallucinating.

**The summary case is an optimistic control:** A human-provided summary was
constructed to contain the exact code. We did **not** benchmark a model
actually generating the summary, its additional prefill/decode costs, or the
probability that it would omit critical details. This single-turn result also
does not show which policy is best after 20+ future turns.

## Practical policy for the Raspberry Pi agent

**Recommended starting design (not yet implemented/validated as production):**

1. **Append-first:** Preserve the system instructions, stable tool definitions,
   and earlier chat/tool messages where possible. This favours prompt-prefix
   reuse. Do not rewrite earlier evidence or timestamp the top of every prompt.
2. **Keep live observations structured:** Tool output should distinguish
   `present`, `absent`, `unknown`, timeouts, errors and timestamps. Historical
   SQLite records should be retrieved with explicit memory tools when needed.
3. **Prune bulky expendable outputs:** Large camera detection dumps and terminal
   logs need not remain verbatim in active context after their useful facts have
   been recorded. Pruning old blocks can invalidate cache prefixes, so do it
   sparingly and measure subsequent latency.
4. **Compact only when needed:** Monitor actual token counts and reserve room
   for tool calls, returned observations and model output. For `num_ctx=2048`,
   **75–80% occupancy** is a *tentative heuristic*, not a model/framework
   default or a benchmark-proven optimum.
5. **Summary + recent turns:** Once compaction is warranted, retain a concise
   summary of prior goals and decisions and preserve the most useful recent
   messages. Do not assume LLM summaries preserve critical sensor evidence.
6. **Persist facts outside the prompt:** Our Pi already has SQLite history; use
   it as the source of truth for observations and time-sensitive events. A short
   summary is a navigation aid, not a replacement for structured records.
7. **Evaluate quality with efficiency:** Track per-round prompt/cached counts,
   prefill/decode/loading, successful tool execution, grounding and loss of old
   facts. Full context can be faster *now* but must eventually be compacted to
   stay within the allocated window.

The broad pattern resembles the context-management ideas used in contemporary
agent harnesses such as **Pi** and **Hermes**: keep active history while useful,
then compress or summarise older dialogue while preserving important recent
context, with external memory/search for recovery. Their precise default
thresholds and retention budgets are version-dependent and should **not** be
copied unchanged to an 8 GB Pi with a 2K context.

## Gotchas and limitations

- `keep_alive` controls model residency, **not** prompt-cache hits.
- Prefix reuse matches **tokenised context under the runner's rules**, not just
  two visually identical sentences. Chat templates, tools and system-message
  placement may change the token sequence.
- Different models can have very different cache behaviour; recurrent models
  especially may require special state/checkpoint handling.
- The exact repeated-prompt cache count (270/274) is not evidence that every
  new query sharing those first 270 tokens will reuse them.
- A cache miss can cost more than a shorter prompt saves: compare **uncached
  tokens and prefill time**, not prompt length alone.
- Rewriting or summarising past messages may destroy some cache reuse; that is
  sometimes worthwhile to regain context capacity or preserve essential facts.
- In the context experiment the policy (recent-only) lost a fact, but the model
  did not fabricate it; `UNKNOWN` was correct given the missing evidence.
- These observations are **single-run results**. Temperature, frequency,
  runner version, prior sessions and allocator state can affect repeatability.
- No direct system-wide memory/power measurements were collected in these
  specific prefix/compaction tests.
- This work does **not** claim automatic context compaction was implemented in
  the production Pi agent; it studies the decision and its consequences.

## Source files and next work

| File | Purpose |
| --- | --- |
| [`prefix_cache_repro.py`](prefix_cache_repro.py) | Recreates short-prefix, identical, continuation and long-prefix tests; optional JSON logging |
| [`context_policy_repro.py`](context_policy_repro.py) | Recreates full/recent/summary policies; optional JSON logging |
| [`results/observed_2026-10-28.json`](results/observed_2026-10-28.json) | Exact measurements transcribed from the learning sessions |
| [`tests/test_context_repro.py`](tests/test_context_repro.py) | Offline unit tests for scenario construction and metrics |

Your **existing** `src/week_13/prompt_caching.py`,
`src/week_13/context_compaction.py`, and `src/week_13/kv_probe.py` are
left untouched. The new scripts are named differently to avoid overwriting
local experiments.

**Next work:** Run a multi-turn agent scenario that gradually grows toward a
2K token budget, measure when compaction pays off *over several subsequent
turns*, and add a retrieval-backed summary/memory policy with correctness
checks. Separately, Module 3's remaining quantisation-quality capstone should
compare the verified FP16 and Q4_K_M models on the same held-out agent tasks.

**Top-level README link (optional):**

```markdown
## Week 13 — Prompt caching and context management

[Week 13 experiments and findings](src/week_13/README.md): reproducible
prefix-cache tests, full/recent/summary context policies, measured prefill
latencies, cache-behaviour gotchas, and a practical Pi-agent memory policy.
```

This patch intentionally does **not** edit the root `README.md`, to avoid
conflicts with independently edited repository documentation.
