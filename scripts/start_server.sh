#!/usr/bin/env bash
# Start vLLM server (CPU) with the default FCFS scheduler.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_DIR/venv/bin/activate"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-4}"
# leave one core for the benchmark client
export VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-0-2}"

vllm serve "$MODEL" \
  --port 8000 \
  --dtype "${DTYPE:-float32}" \
  --scheduling-policy fcfs