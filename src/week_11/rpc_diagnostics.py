#!/usr/bin/env python3
"""Week 11: compare Pi RPC prompt/tool configurations in the SAME working dir.

Standalone experiment; does not modify the user's repository or Pi settings.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time

WEEK11 = Path.home() / 'ollama_python' / 'src' / 'week_11'
TOOLS = 'detect_motion,observe_scene,check_presence,query_memory'
BASE = [
    'pi', '--mode', 'rpc', '--model', 'ollama/gemma4:e2b-8k',
    '--no-mcp', '--no-session',
]
PROFILES = {
    'minimal': ['--no-tools', '--no-extensions', '--no-context-files'],
    'instructions_only': ['--no-tools', '--no-extensions'],
    'extension_inactive': ['--no-tools', '--no-context-files'],
    'tools_only': ['--no-context-files', '--tools', TOOLS],
    'full': ['--tools', TOOLS],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=1, help='Independent sessions per profile (default 1)')
    parser.add_argument('--profiles', default=','.join(PROFILES), help='Comma-separated profile names')
    parser.add_argument('--question', default='What is 2 + 17?')
    parser.add_argument('--expected', type=int, default=19, help='Expected final numeric answer')
    parser.add_argument('--timeout', type=float, default=180.0)
    parser.add_argument('--output', type=Path, default=Path('/tmp/week11_rpc_ablation.csv'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be at least 1')
    names = [x.strip() for x in args.profiles.split(',') if x.strip()]
    if not names or any(n not in PROFILES for n in names):
        parser.error(f'Profiles must be from: {", ".join(PROFILES)}')
    if not WEEK11.is_dir():
        parser.error(f'Pi project directory not found: {WEEK11}')

    print(f'Pi working directory (ALL profiles): {WEEK11}', flush=True)
    for name in names:
        print(f'{name}: {" ".join(BASE + PROFILES[name])}')
    if args.dry_run:
        return 0

    sys.path.insert(0, str(WEEK11))
    from pi_rpc import PiRPCClient  # local tested bridge

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(1, args.runs + 1):
        for name in names:
            row = {
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'run': rep, 'profile': name, 'question': args.question,
                'answer': '', 'correct': False, 'elapsed_s': '',
                'input_tokens': '', 'output_tokens': '', 'error': '',
            }
            print(f'\n[{name} #{rep}]', flush=True)
            try:
                with PiRPCClient(
                    command=BASE + PROFILES[name],
                    cwd=WEEK11,
                    timeout_s=args.timeout,
                ) as client:
                    turn = client.ask(args.question)
                # Last standalone integer is the answer for this arithmetic test.
                nums = re.findall(r'(?<!\d)-?\d+(?!\d)', turn.text)
                correct = bool(nums and int(nums[-1]) == args.expected)
                row.update(answer=turn.text, correct=correct,
                           elapsed_s=turn.elapsed_s, input_tokens=turn.input_tokens,
                           output_tokens=turn.output_tokens)
                print(f'answer={turn.text!r} correct={correct} '
                      f'elapsed={turn.elapsed_s:.3f}s input={turn.input_tokens} '
                      f'output={turn.output_tokens}', flush=True)
            except Exception as exc:
                row['error'] = str(exc)
                print(f'ERROR: {exc}', flush=True)
            rows.append(row)
            with args.output.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
    print(f'\nResults saved to {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
