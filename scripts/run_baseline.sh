#!/usr/bin/env bash
# Run the FCFS baseline: scenarios A (uniform) and B (mixed), repeated runs.
# Usage: bash scripts/run_baseline.sh [NUM_REQUESTS] [RATE] [RUNS]
# Env:   TAG (default fcfs), MODEL (default Qwen/Qwen2.5-0.5B-Instruct)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
source venv/bin/activate

N="${1:-50}"
RATE="${2:-0.5}"
RUNS="${3:-3}"
TAG="${TAG:-fcfs}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

# Make sure the server is up
if ! curl -sf http://localhost:8000/v1/models > /dev/null; then
  echo "vLLM server is not reachable on localhost:8000" >&2
  exit 1
fi

declare -A SCEN=( [A]=A_uniform [B]=B_mixed )

# Warm-up (results are discarded)
python -m workload.generator --scenario B_mixed --num-requests 4 --rate 1.0 \
  --seed 123 --out results/warmup.jsonl
python -m benchmarks.run_benchmark --workload results/warmup.jsonl \
  --model "$MODEL" --out-dir results/warmup > /dev/null

for S in A B; do
  WL="results/${TAG}_workload_${S}.jsonl"
  python -m workload.generator --scenario "${SCEN[$S]}" --num-requests "$N" \
    --rate "$RATE" --seed 0 --out "$WL"
  for i in $(seq 1 "$RUNS"); do
    echo "=== ${TAG} scenario ${S} run ${i}/${RUNS} ==="
    python -m benchmarks.run_benchmark --workload "$WL" \
      --model "$MODEL" --out-dir "results/${TAG}_${S}_run${i}" > /dev/null
  done
done

python -m benchmarks.aggregate results "$TAG" | tee "results/${TAG}_summary.txt"