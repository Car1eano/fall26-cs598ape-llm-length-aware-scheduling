"""Summarize vLLM scheduler CPU time, batch composition and GPU utilization."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

MARKER = "CS598_SCHED_PROFILE "


def profile_run(run):
    status = json.loads((run / "status.json").read_text())
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    snapshots = run / "scheduler_profile.jsonl"
    if snapshots.is_file():
        samples = [json.loads(line) for line in snapshots.read_text().splitlines()]
    else:
        with (run / "server.log").open() as log:
            samples = [json.loads(line.split(MARKER, 1)[1]) for line in log if MARKER in line]
    if not status["ok"] or not samples:
        raise ValueError(f"No valid scheduler profile in {run}")
    profile = samples[-1]
    with (run / "gpu_utilization.csv").open() as metrics:
        gpu = [float(value) for _, value, _ in csv.reader(line for line in metrics if line[:1].isdigit())]
    if not gpu:
        raise ValueError(f"No GPU utilization samples in {run}")
    return {
        "block": config["block"], "policy": config["policy"],
        "schedule_calls": profile["schedule_calls"],
        "nonempty_steps": profile["steps"], "max_waiting": profile["max_waiting"],
        "mean_schedule_us": profile["schedule_ns"] / profile["schedule_calls"] / 1000,
        "mean_sort_us": (profile["sort_ns"] / profile["sort_calls"] / 1000
                         if profile["sort_calls"] else 0.0),
        "sort_share_of_scheduler": profile["sort_ns"] / profile["schedule_ns"],
        "steps_with_queue_ge_100": profile["deep_queue_steps"],
        "sort_share_when_queue_ge_100": (
            profile["deep_queue_sort_ns"] / profile["deep_queue_schedule_ns"]
            if profile["deep_queue_schedule_ns"] else None
        ),
        "mean_scheduled_sequences": profile["slots"] / profile["steps"],
        "scheduled_prompt_token_share": profile["prefill_tokens"] / (
            profile["prefill_tokens"] + profile["decode_tokens"]
        ),
        "mean_gpu_sm_utilization_pct": mean(gpu), "gpu_samples": len(gpu),
        "throughput_tok_s": summary["overall"]["throughput_tokens_per_s"],
        "short_p95_ttft_s": summary["by_bucket"]["short"]["ttft"]["p95"],
        "long_p95_ttft_s": summary["by_bucket"]["long"]["ttft"]["p95"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    with (args.batch / "progress.jsonl").open() as journal:
        entries = [json.loads(line) for line in journal]
    rows = [profile_run(args.batch.parent / row["path"]) for row in entries if row["ok"]]
    destination = args.batch / "scheduler_profiles.csv"
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


if __name__ == "__main__":
    main()
