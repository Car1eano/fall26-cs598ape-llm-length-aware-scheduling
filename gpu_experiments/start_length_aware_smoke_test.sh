#!/usr/bin/env bash
set -euo pipefail

cd ~/fall26-cs598ape-llm-length-aware-scheduling
source venv-vllm-test/bin/activate

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 18081 \
  --dtype float16 \
  --scheduling-policy fcfs \
  --scheduler-cls gpu_experiments.length_aware_scheduler.LengthAwareScheduler \
  --no-enable-prefix-caching \
  --max-num-seqs 4 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.30 \
  --enforce-eager \
  2>&1 | tee gpu_experiments/logs/length_aware_smoke_test.log
