#!/usr/bin/env bash
# Usage: bash ollama_benchmark.sh MODEL SPEED_RUNS [--quality-runs N]
# Shared dataset/scorer, with curl making the Ollama requests.
# Requires python3 and curl; the Python ollama package is unnecessary here.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$script_dir/ollama_benchmark.py" "$@" --transport curl
