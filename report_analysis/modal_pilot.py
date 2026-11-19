"""Summarize downloaded FCFS pilot runs. Usage: python report_analysis/modal_pilot.py"""

import csv
import json
import sys
from pathlib import Path


def queue_counter(path, suffix):
    metric = f"vllm:request_queue_time_seconds_{suffix}"
    values = [float(line.split()[1]) for line in path.read_text().splitlines()
              if line.startswith(metric + "{") or line.startswith(metric + " ")]
    if not values:
        raise ValueError(f"Missing {metric} in {path}")
    return sum(values)


def summarize_run(path):
    def load(name):
        return json.loads((path / name).read_text())

    status = load("status.json")
    if not status["ok"]:
        raise ValueError(f"Failed audit: {path}")
    config = load("config.json")
    summary = load("summary.json")
    rows = [json.loads(line) for line in (path / "raw_results.jsonl").read_text().splitlines()]
    before, after = path / "metrics_before.txt", path / "metrics_after.txt"
    count = queue_counter(after, "count") - queue_counter(before, "count")
    if count != len(rows):
        raise ValueError(f"Queue histogram covers {count} of {len(rows)} requests in {path}")
    sends = [r["send_time"] for r in rows]
    return {
        "mix": config["mix"], "target_req_s": config["target_rate"],
        "actual_req_s": (len(rows) - 1) / (max(sends) - min(sends)),
        "requests": len(rows), "mean_queue_s": (queue_counter(after, "sum") -
                                                queue_counter(before, "sum")) / count,
        "short_p95_ttft_s": summary["by_bucket"]["short"]["ttft"]["p95"],
        "long_p95_ttft_s": summary["by_bucket"]["long"]["ttft"]["p95"],
        "throughput_tok_s": summary["overall"]["throughput_tokens_per_s"],
        "benchmark_s": summary["overall"]["duration_s"], "function_s": status["elapsed_s"],
        "run": str(path),
    }


if __name__ == "__main__":
    paths = sorted(Path("results/modal_pilot").glob("*/*/status.json"))
    records = [summarize_run(path.parent) for path in paths]
    if records:
        writer = csv.DictWriter(sys.stdout, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: (row["mix"], row["target_req_s"])))
