#!/usr/bin/env bash
# Usage: bash ollama_benchmark.sh MODEL RUNS [PROMPT]
# Requires curl and jq. Prints a JSON report; errors stop the script.
set -euo pipefail
[[ $# -ge 2 && $# -le 3 && ${2:-} =~ ^[1-9][0-9]*$ ]] || {
  echo 'Usage: bash ollama_benchmark.sh MODEL RUNS [PROMPT]' >&2; exit 2;
}
model=$1
runs=$2
prompt=${3:-Explain in three short bullet points what an offline Raspberry Pi AI assistant can do.}
api=http://localhost:11434/api
results=$(mktemp)
trap 'rm -f "$results"' EXIT

# Unload once, so run 1 includes loading. OS disk caches remain intact.
jq -nc --arg m "$model" '{model:$m,keep_alive:0,stream:false}' |
  curl -fsS --max-time 600 -H 'Content-Type: application/json' \
    -d @- "$api/generate" | jq -e '.done==true and .error==null' >/dev/null

payload=$(jq -nc --arg m "$model" --arg p "$prompt" '
  {model:$m,messages:[{role:"user",content:$p}],think:false,stream:false,
   keep_alive:"10m",options:{num_ctx:2048,num_predict:120,temperature:0}}')
for ((i=1; i<=runs; i++)); do
  echo "Run $i/$runs: $model" >&2
  curl -fsS --max-time 600 -H 'Content-Type: application/json' \
    -d "$payload" "$api/chat" |
  jq -ce --argjson run "$i" '
    if .error or .done!=true or ((.message.content // "")|test("\\S")|not)
    then error(.error // "Incomplete or empty response")
    else {run:$run,content:.message.content,stop_reason:.done_reason,
      total_time_s:(.total_duration/1e9),load_time_s:(.load_duration/1e9),
      generated_tokens:.eval_count,
      tokens_per_second:(if .eval_duration>0 then .eval_count/(.eval_duration/1e9) else 0 end)} end
  ' >> "$results"
done

# Average only completed subsequent runs; truncated answers remain in runs.
jq -s --arg model "$model" --arg prompt "$prompt" '
  [.[]|select(.run>1 and .stop_reason=="stop")] as $warm |
  {model:$model,prompt:$prompt,
   settings:{think:false,num_ctx:2048,num_predict:120,temperature:0},runs:.,
   warm_runs_count:($warm|length),
   warm_means:(if ($warm|length)==0 then null else
     reduce ["total_time_s","load_time_s","generated_tokens","tokens_per_second"][] as $k
       ({}; .[$k]=([$warm[]|.[$k]]|add/length)) end)}
' "$results"
