#!/usr/bin/env bash
# One-time setup of a fresh course VM: Python 3.12 venv, vLLM CPU wheel,
# the C++ compiler needed at warm-up, and test dependencies.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

VLLM_VERSION="${VLLM_VERSION:-0.31.0}"
WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}%2Bcpu-cp38-abi3-manylinux_2_39_x86_64.whl"

# vLLM's CPU backend compiles kernels at warm-up and needs g++
if ! command -v g++ > /dev/null; then
  sudo apt update
  sudo apt install -y build-essential
fi

# uv provides Python 3.12 without touching the system Python (3.14)
if ! command -v uv > /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d venv ]; then
  uv venv --python 3.12 venv
fi
source venv/bin/activate

uv pip install aiohttp numpy pytest
uv pip install "$WHEEL" --torch-backend cpu

python -c "import vllm; print('vllm', vllm.__version__)"
python -m pytest tests/ -q