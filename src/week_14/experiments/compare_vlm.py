"""Small reproducible Ollama SDK VLM runner.

Use --mode compare for an ordered BEFORE/AFTER pair, or --mode single to ask
about mug vs glass bowl on the AFTER photograph. Saves model response and raw
Ollama counters; timings are per run, NOT performance guarantees.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
from ollama import Client

COMPARE_PROMPT = (
    'These are two photos of the same scene, ordered BEFORE then AFTER. '
    'Describe only the visible changes between them. Be specific and concise. '
    'If uncertain, say so. Do not invent changes.'
)
# The exact single-image prompt used previously was not preserved.
SINGLE_PROMPT = (
    'Where is the patterned mug relative to the glass bowl? '
    'Describe only what is visible and avoid inventing details.'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='lfm3-vl:2b')
    parser.add_argument('--mode', choices=('compare', 'single'), default='compare')
    parser.add_argument('--before', type=Path)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--host', default='http://127.0.0.1:11434')
    parser.add_argument('--output', type=Path, default=Path('vlm_run.json'))
    args = parser.parse_args()
    if args.mode == 'compare' and args.before is None:
        parser.error('--before is required for --mode compare')
    images = [args.before, args.after] if args.mode == 'compare' else [args.after]
    for path in images:
        if not path.is_file():
            parser.error(f'Missing image: {path}')
    prompt = COMPARE_PROMPT if args.mode == 'compare' else SINGLE_PROMPT
    client = Client(host=args.host, timeout=600)
    t0 = perf_counter()
    result = client.chat(
        model=args.model,
        messages=[{
            'role': 'user', 'content': prompt,
            'images': [str(path.resolve()) for path in images],
        }],
        stream=False, think=False,
        options={'num_ctx': 4096, 'num_predict': 128,
                 'num_thread': 2, 'temperature': 0},
    )
    client_s = perf_counter() - t0

    # Ollama server counters are nanoseconds; preserve both raw and seconds.
    raw = {key: getattr(result, key, None) for key in (
        'load_duration', 'prompt_eval_duration', 'eval_duration',
        'total_duration', 'prompt_eval_count', 'eval_count'
    )}
    saved = {
        'model': args.model, 'mode': args.mode,
        'images': [str(p) for p in images], 'prompt': prompt,
        'options': {'num_ctx': 4096, 'num_predict': 128,
                    'num_thread': 2, 'temperature': 0, 'think': False},
        'answer': result.message.content,
        'ollama_raw': raw,
        'ollama_seconds': {k.replace('_duration', '_seconds'): v / 1e9
                           for k, v in raw.items() if k.endswith('_duration') and v is not None},
        'client_wall_seconds': client_s,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(saved, indent=2) + '\n')
    print(saved['answer'])
    print(json.dumps({**saved['ollama_seconds'],
                      'prompt_tokens': raw['prompt_eval_count'],
                      'output_tokens': raw['eval_count'],
                      'client_wall_seconds': client_s}, indent=2))
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
