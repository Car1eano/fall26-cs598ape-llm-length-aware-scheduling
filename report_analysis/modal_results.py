"""Aggregate audited Modal runs and paired differences against native FCFS."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from report_analysis.modal_pilot import summarize_run


def load_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def queue_buckets(path):
    counts = defaultdict(float)
    for line in path.read_text().splitlines():
        if line.startswith("vllm:request_queue_time_seconds_bucket{"):
            bound = float(re.search(r'\ble="([^"]+)"', line).group(1))
            counts[bound] += float(line.split()[1])
    return counts


def maximum_queue_interval(run, count):
    before = queue_buckets(run / "metrics_before.txt")
    after = queue_buckets(run / "metrics_after.txt")
    if before.keys() != after.keys() or not after:
        raise ValueError(f"Queue histogram boundaries differ in {run}")
    lower = 0.0
    for upper in sorted(after):
        if after[upper] - before[upper] == count:
            return lower, upper if math.isfinite(upper) else None
        lower = upper
    raise ValueError(f"Queue histogram does not cover all requests in {run}")


def calibration_medians(run):
    if not load_json(run / "status.json")["ok"]:
        raise ValueError(f"Calibration failed audit: {run}")
    rows = sorted(read_rows(run / "raw_results.jsonl"), key=lambda row: row["send_time"])
    if any(a["finish_time"] > b["send_time"] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"Calibration requests overlap: {run}")
    return {bucket: median(r["finish_time"] - r["send_time"] for r in rows
                           if r["bucket"] == bucket) for bucket in ("short", "long")}


def add_metrics(run, baselines):
    config = load_json(run / "config.json")
    summary = load_json(run / "summary.json")
    rows = read_rows(run / "raw_results.jsonl")
    result = summarize_run(run)
    result.update(block=config["block"], policy=config["policy"],
                  replicate=config["replicate"], max_num_seqs=config.get("max_num_seqs", 4))
    result["gpu_environment"] = config["gpu_actual"]
    result["max_queue_lower_s"], result["max_queue_upper_s"] = maximum_queue_interval(run, len(rows))
    trace_file = run / "queue_times.jsonl"
    trace = {r["id"]: r["queue_time_s"] for r in read_rows(trace_file)} if trace_file.exists() else {}
    if trace and trace.keys() != {row["id"] for row in rows}:
        raise ValueError(f"Queue trace IDs differ from benchmark request IDs in {run}")
    result["max_queue_exact_s"] = max(trace.values(), default=None)
    for bucket, stats in summary["by_bucket"].items():
        for metric in ("ttft", "latency"):
            for percentile in ("p50", "p95", "p99"):
                result[f"{bucket}_{percentile}_{metric}_s"] = stats[metric][percentile]

    duration = summary["overall"]["duration_s"]
    hits = sum(row["first_token_time"] - row["send_time"] < 1.0 for row in rows)
    result["ttft_slo_fraction"] = hits / len(rows)
    result["ttft_goodput_req_s"] = hits / duration
    if baselines and set(summary["by_bucket"]) <= baselines.keys():
        normalized = defaultdict(list)
        for row in rows:
            latency = row["finish_time"] - row["send_time"]
            normalized[row["bucket"]].append(latency / baselines[row["bucket"]])
        result.update({f"{bucket}_mean_slowdown": mean(values)
                       for bucket, values in normalized.items()})
        # Equal weight for each length class, using inverse mean slowdown as utility.
        utilities = [1 / mean(values) for values in normalized.values()]
        result["jain_inverse_class_mean_slowdown"] = (
            sum(utilities) ** 2 / (len(utilities) * sum(value ** 2 for value in utilities))
        )
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    baselines = calibration_medians(args.calibration) if args.calibration else None
    rows = []
    for batch in args.batch:
        for item in read_rows(batch / "progress.jsonl"):
            if item["ok"]:
                rows.append(add_metrics(batch.parent / item["path"], baselines))
    if not rows:
        raise ValueError("No downloaded runs passed audit")
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "individual_runs.csv", rows)
    keys = ("mix", "target_req_s", "max_num_seqs", "policy")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    def comparable_metrics(runs):
        # Histogram bounds are per run; averaging endpoints obscures their meaning.
        return [
            key for key, value in runs[0].items()
            if type(value) is float and key != "target_req_s"
            and not key.startswith(("max_queue_lower_", "max_queue_upper_"))
            and all(run.get(key) is not None for run in runs)
        ]

    conditions = [
        {**dict(zip(keys, group)), "n": len(runs),
         **{metric: mean(run[metric] for run in runs) for metric in comparable_metrics(runs)}}
        for group, runs in sorted(groups.items())
    ]
    write_csv(args.out / "condition_means.csv", conditions)

    fcfs = {(row["block"], row["max_num_seqs"]): row for row in rows if row["policy"] == "native_fcfs"}
    differences = []
    for row in rows:
        reference = fcfs.get((row["block"], row["max_num_seqs"]))
        if row["policy"] != "native_fcfs" and reference:
            differences.append({**{key: row[key] for key in (*keys, "block", "replicate")},
                                "gpu_environment": row["gpu_environment"],
                                "fcfs_gpu_environment": reference["gpu_environment"],
                                "same_gpu_environment": (
                                    row["gpu_environment"] == reference["gpu_environment"]
                                ),
                                **{f"{metric}_delta": row[metric] - reference[metric]
                                   for metric in comparable_metrics([row, reference])}})
    if differences:
        write_csv(args.out / "paired_differences.csv", differences)


if __name__ == "__main__":
    main()
