#!/usr/bin/env bash
set -euo pipefail

cd ~/fall26-cs598ape-llm-length-aware-scheduling

source venv-vllm-test/bin/activate

# Use physical GPU 1 only after confirming permission.
export CUDA_VISIBLE_DEVICES=1

# Avoid conflicting with another user's API server.
PORT=18080

# Save the complete startup log.
LOG="gpu_experiments/logs/gpu_smoke_test.log"

echo "Starting vLLM GPU smoke test..."
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Port: $PORT"
echo "Log: $LOG"

vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype float16 \
  --scheduling-policy fcfs \
  --max-num-seqs 4 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.30 \
  --enforce-eager \
  2>&1 | tee "$LOG"
