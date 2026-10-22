## Week 10 vs Week 11: Routing robustness and performance

### Observed routing failure (22 October 2026)

In a Raspberry Pi 5 benchmark using `gemma4:e2b-8k`, the Week 10 custom Python voice agent received:

> How many person-positive observation events have been recorded?

Its diagnostic output showed `[route] general`, followed by: “I do not have any person-positive observation events recorded at this time.” The SQLite ground truth was **14 person-positive events**. Thus, the legacy agent did not select its historical-memory path and returned an incorrect answer in **1.600 s** (`llm_s`).

For the identical question, the Week 11 Pi agent selected `query_memory` and correctly returned **14**. Its agent latency was **50.369 s**, of which **0.049 s** was reported as memory-tool execution time. The remaining delay was outside the SQLite lookup, in the overall Pi/model execution path.

A second historical query exposed a similar issue: when asked when a person was last detected, Week 10 answered that it lacked recent readings; Pi retrieved stored observations and identified the latest positive detection at **2026-10-21 19:10:23 AEDT**.

### Interpretation

- **Routing coverage:** Our Week 10 hand-maintained intent router proved sensitive to phrasing. A legitimate saved-history question was classified as `general` instead of being routed to memory, producing a fast but ungrounded answer. This is an observed weakness of *our implementation*, not evidence that all explicit routing architectures are inherently unreliable.
- **Correctness before latency:** Week 10's 1.600 s versus Pi's 50.369 s is **not a fair latency comparison of equivalent successful retrievals**. Only Pi performed the requested historical-memory lookup correctly in that test.
- **Pi's latency limitation remains real:** Database execution was approximately 49 ms, yet the overall Pi agent took over 50 s. A persistent session showed improved latency for later questions, but sometimes reused earlier answers without re-querying SQLite; the speed/grounding trade-off must be kept in mind.
- **Comparable sensor task:** Both agents produced appropriately cautious live-presence answers. Their warm median measured agent latencies were **7.269 s (Week 10)** and **25.132 s (Week 11)**, so Pi was markedly slower in this scenario.
- **Simple arithmetic:** Their warm median measured agent latencies were **1.089 s (Week 10)** and **0.508 s (Week 11)**. Pi was faster at the model/agent stage, though not necessarily in total process wall time.

### Decision

**Retain Week 10 unchanged as the historical baseline.** Do not patch its routing rules merely to improve this benchmark. Document the failure as a useful lesson: hand-maintained intent-routing coverage needs thorough paraphrase and boundary-case tests, and latency comparisons must exclude incorrect or incomplete answers. Continue developing Week 11 around Pi's extensibility, tool-backed grounding, and persistent sessions, while reporting its context-processing and sensor latency costs transparently.
