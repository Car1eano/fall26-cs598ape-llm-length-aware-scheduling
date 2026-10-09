#!/usr/bin/env bash
# Package a completed A/B benchmark with three runs per scenario.
# Usage: bash scripts/package_results.sh TAG SERVER_LOG [BENCHMARK_LOG]
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/package_results.sh TAG SERVER_LOG [BENCHMARK_LOG]" >&2
  exit 1
fi
TAG="$1"
if [[ ! "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "TAG must contain only letters, digits, underscores or hyphens." >&2
  exit 1
fi
FILES=("results/${TAG}_summary.txt")
for S in A B; do
  FILES+=("results/${TAG}_workload_${S}.jsonl")
  for I in 1 2 3; do
    FILES+=("results/${TAG}_${S}_run${I}/raw_results.jsonl")
    FILES+=("results/${TAG}_${S}_run${I}/summary.json")
  done
done
# Keep logs inside the project so archive paths remain portable.
for LOG in "${@:2}"; do
  if [[ "$LOG" == /* || "$LOG" == -* || "/$LOG/" == *"/../"* ]]; then
    echo "Use a log path relative to the project, without '..': $LOG" >&2
    exit 1
  fi
  FILES+=("$LOG")
done
for F in "${FILES[@]}"; do
  if [[ ! -s "$F" ]]; then
    echo "Missing or empty file: $F" >&2
    exit 1
  fi
done
TMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TMP_DIR"' EXIT
# This reads existing results; it does not rerun the benchmark.
PYTHONDONTWRITEBYTECODE=1 "$REPO_DIR/venv/bin/python" -m benchmarks.audit results "$TAG" | tee "$TMP_DIR/audit.txt"
{
  echo "tag=$TAG"
  echo "packaged_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "checkout_commit=$(git rev-parse HEAD)"
  echo "checkout_status:"
  git status --short
  echo "server_log=$2"
  echo "Environment below is the packaging-time snapshot, not historical proof:"
  uname -a
  PYTHONDONTWRITEBYTECODE=1 "$REPO_DIR/venv/bin/python" -c 'import platform, importlib.metadata as m; print("Python", platform.python_version()); print("vLLM", m.version("vllm"))'
} > "$TMP_DIR/metadata.txt"
mkdir -p artifacts
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="artifacts/${TAG}_${STAMP}_$(basename "$TMP_DIR").tar.gz"
tar -czf "$ARCHIVE" -C "$REPO_DIR" "${FILES[@]}" -C "$TMP_DIR" audit.txt metadata.txt
printf 'Saved: %s\n' "$ARCHIVE"
