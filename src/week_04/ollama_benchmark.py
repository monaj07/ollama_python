#!/usr/bin/env python3
"""Usage: python3 ollama_benchmark.py MODEL RUNS [PROMPT]. Requires ollama."""
import argparse
import json
from dataclasses import asdict, dataclass
from statistics import mean

import ollama

DEFAULT_PROMPT = "Explain in three short bullet points what an offline Raspberry Pi AI assistant can do."
OPTIONS = {"num_ctx": 2048, "num_predict": 120, "temperature": 0, "draft_num_predict": 0}


@dataclass
class LLMResponse:
    content: str
    total_time_s: float
    load_time_s: float
    generated_tokens: int
    tokens_per_second: float
    stop_reason: str | None


class OllamaBackend:
    def __init__(self, model):
        self.model = model
        self.client = ollama.Client(host="http://localhost:11434", timeout=600)

    def generate(self, messages) -> LLMResponse:
        output = self.client.chat(
            model=self.model, messages=messages, think=False, stream=False,
            keep_alive="10m", options=OPTIONS,
        )
        content = output.message.content
        if not output.done or not content or not content.strip():
            raise ValueError("Incomplete or empty response from the model")
        generation_time_s = output.eval_duration / 1e9
        return LLMResponse(
            content=content, total_time_s=output.total_duration / 1e9,
            load_time_s=output.load_duration / 1e9,
            generated_tokens=output.eval_count,
            tokens_per_second=(output.eval_count / generation_time_s if generation_time_s > 0 else 0),
            stop_reason=output.done_reason,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("runs", type=int)
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be a positive integer")
    llm = OllamaBackend(args.model)
    # Unload once. This does not clear the OS disk cache.
    llm.client.generate(model=args.model, keep_alive=0, stream=False)
    results = []
    for run in range(1, args.runs + 1):
        output = llm.generate([{"role": "user", "content": args.prompt}])
        results.append({"run": run, **asdict(output)})
    warm = [row for row in results[1:] if row["stop_reason"] == "stop"]
    metrics = ["total_time_s", "load_time_s", "generated_tokens", "tokens_per_second"]
    print(json.dumps({
        "model": args.model, "prompt": args.prompt,
        "settings": {"think": False, **OPTIONS}, "runs": results,
        "warm_runs_count": len(warm),
        "warm_means": {key: mean(row[key] for row in warm) for key in metrics} if warm else None,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
