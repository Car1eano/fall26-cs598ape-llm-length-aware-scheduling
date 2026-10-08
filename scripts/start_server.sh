#!/usr/bin/env bash
# Start vLLM server (CPU) with the default FCFS scheduler.
# Optional env: MODEL, DTYPE, VLLM_CPU_KVCACHE_SPACE, MAX_NUM_SEQS
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_DIR/venv/bin/activate"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-4}"
# leave one core for the benchmark client
export VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-0-2}"

EXTRA_ARGS=()
if [ -n "${MAX_NUM_SEQS:-}" ]; then
  EXTRA_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

vllm serve "$MODEL" \
  --port 8000 \
  --dtype "${DTYPE:-float32}" \
  --scheduling-policy fcfs \
  "${EXTRA_ARGS[@]}"