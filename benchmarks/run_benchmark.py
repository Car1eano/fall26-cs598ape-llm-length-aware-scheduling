"""Run benchmark against a running vLLM server: python -m benchmarks.run_benchmark --help."""
import argparse
import asyncio
import json
from pathlib import Path
from benchmarks.client import run_workload
from benchmarks.metrics import summarize
from workload.generator import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    rows = asyncio.run(run_workload(read_jsonl(args.workload), args.url, args.model))
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, output_dir / "raw_results.jsonl")
    summary = summarize(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
