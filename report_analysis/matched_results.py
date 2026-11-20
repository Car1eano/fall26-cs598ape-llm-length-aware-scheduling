"""Verify and summarize same-container Modal blocks."""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

METRICS = ("short_p95_ttft_s", "long_p95_ttft_s", "long_p99_ttft_s", "throughput_tok_s")
QUEUE_SAMPLE = re.compile(r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Waiting: (\d+) reqs")


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def queue_depth_rows(run, row):
    log = run / "server.log"
    if not log.is_file():
        return []
    with log.open(errors="replace") as stream:
        samples = [match.groups() for line in stream if (match := QUEUE_SAMPLE.search(line))]
    if not samples:
        return []
    first = datetime.strptime(samples[0][0], "%m-%d %H:%M:%S")
    return [{
        **{k: row[k] for k in ("block", "policy", "run")},
        "offset_s": (datetime.strptime(stamp, "%m-%d %H:%M:%S") - first).total_seconds(),
        "waiting": int(waiting),
    } for stamp, waiting in samples]


def read_block(path):
    block = read_json(path)
    if not block["ok"]:
        return block, []
    successful = {r["policy"]: r for r in block["runs"] if r["ok"]}
    if set(successful) != set(block["policy_order"]):
        raise ValueError(f"Incomplete successful block: {path}")

    volume_root = path.parents[2]
    identity_keys = ("hostname", "gpu_identity", "modal_task_id")
    expected_identity = tuple(block[k] for k in identity_keys)
    calibration = volume_root / block["calibration"]["path"]
    calibration_config = read_json(calibration / "config.json")
    calibration_summary = read_json(calibration / "summary.json")
    calibration_capacity = calibration_summary["overall"]["throughput_tokens_per_s"]
    if (not block["calibration"]["ok"]
            or tuple(calibration_config[k] for k in identity_keys) != expected_identity
            or calibration_capacity != block["capacity_tok_s"]):
        raise ValueError(f"Invalid or different host for calibration: {calibration}")

    workload_hashes = set()
    rows = []
    for policy, record in successful.items():
        run = volume_root / record["path"]
        config = read_json(run / "config.json")
        identity = tuple(config[k] for k in identity_keys)
        if (identity != expected_identity or config["capacity_tok_s"] != block["capacity_tok_s"]
                or config["gpu_actual"] != calibration_config["gpu_actual"]):
            raise ValueError(f"Different host or capacity at {run}")
        if config["policy"] != policy:
            raise ValueError(f"Policy label differs at {run}")
        if config["target_rate"] != block["target_rate"]:
            raise ValueError(f"Arrival rate differs inside the block at {run}")
        workload_hashes.add(hashlib.sha256((run / "workload.jsonl").read_bytes()).hexdigest())
        raw = read_jsonl(run / "raw_results.jsonl")
        queue_rows = read_jsonl(run / "queue_times.jsonl")
        queues = {r["id"]: r["queue_time_s"] for r in queue_rows}
        if (len(raw) != 900 or len(queue_rows) != 900 or len(queues) != 900
                or {r["id"] for r in raw} != queues.keys() or any(r["error"] for r in raw)):
            raise ValueError(f"Bad client or queue trace at {run}")
        summary = read_json(run / "summary.json")
        shorts, longs = (summary["by_bucket"][kind] for kind in ("short", "long"))
        throughput = summary["overall"]["throughput_tokens_per_s"]
        row = {
            **{k: block[k] for k in ("block", "campaign", "rho", "replicate", "target_rate",
                                     "capacity_tok_s", "hostname", "gpu_identity", "modal_task_id")},
            "policy": policy, "run": run.as_posix(),
            "short_p95_ttft_s": shorts["ttft"]["p95"],
            "long_p95_ttft_s": longs["ttft"]["p95"],
            "long_p99_ttft_s": longs["ttft"]["p99"],
            "throughput_tok_s": throughput,
            "throughput_over_capacity": throughput / block["capacity_tok_s"],
            "ttft_slo_fraction": mean(r["first_token_time"] - r["send_time"] < 1 for r in raw),
            "mean_queue_s": mean(queues.values()),
        }
        if block["campaign"] == "stress":
            last_short = max(r["send_time"] for r in raw if r["bucket"] == "short")
            probes = [r for r in raw if r["bucket"] == "long"]
            row.update(
                long_probes=len(probes),
                long_after_last_short=sum(r["first_token_time"] > last_short for r in probes),
                long_mean_queue_s=mean(queues[r["id"]] for r in probes),
                long_max_queue_s=max(queues[r["id"]] for r in probes),
            )
        rows.append(row)

    if len(workload_hashes) != 1:
        raise ValueError(f"Different workload copies inside {path.parent}")
    return block, rows


def analyze(batches, output):
    audit = []
    rows = []
    seen = set()
    for batch in batches:
        recorded = set()
        for path in sorted(batch.glob("*/block.json")):
            try:
                block, block_rows = read_block(path)
            except FileNotFoundError as error:
                audit.append({"batch": batch.name, "block": path.parent.name, "complete": False,
                              "error": f"Missing artifact: {error.filename}"})
                recorded.add(path.parent.name)
                continue
            recorded.add(block["block"])
            audit.append({
                "batch": batch.name, "block": block["block"], "complete": block["ok"],
                "successes": sum(r["ok"] for r in block["runs"]),
                "failed_attempts": sum(not r["ok"] for r in block["runs"]),
                "capacity_tok_s": block.get("capacity_tok_s"), "error": block.get("error"),
                "elapsed_s": block.get("elapsed_s"),
            })
            if block_rows:
                if block["block"] in seen:
                    raise ValueError(f"Duplicate complete block: {block['block']}")
                seen.add(block["block"])
                rows.extend(block_rows)
        journal = batch / "progress.jsonl"
        if journal.is_file():
            for entry in read_jsonl(journal):
                if entry["block"] in recorded:
                    continue
                audit.append({
                    "batch": batch.name, "block": entry["block"], "complete": False,
                    "error": entry.get("error") or "Block result has not been downloaded",
                })
                recorded.add(entry["block"])
        plan = batch / "plan.json"
        if plan.is_file():
            for entry in read_json(plan)["blocks"]:
                if entry["block"] not in recorded:
                    audit.append({
                        "batch": batch.name, "block": entry["block"], "complete": False,
                        "error": "No block outcome recorded",
                    })
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "block_audit.csv", audit)
    if not rows:
        return []

    write_csv(output / "individual_runs.csv", rows)
    baselines = {r["block"]: r for r in rows if r["policy"] == "native_fcfs"}
    deltas = []
    for r in rows:
        if r["policy"] == "native_fcfs":
            continue
        base = baselines[r["block"]]
        deltas.append({
            **{key: r[key] for key in ("block", "campaign", "rho", "replicate", "policy", "gpu_identity",
                                      "capacity_tok_s", "target_rate", "run")},
            "fcfs_run": base["run"],
            **{f"{m}_delta": r[m] - base[m] for m in METRICS},
        })
    write_csv(output / "paired_differences.csv", deltas)
    variants = [r for r in deltas if r["campaign"] == "variants"]
    write_csv(output / "table3_paired_blocks.csv", variants)
    variant_groups = defaultdict(list)
    for row in variants:
        variant_groups[row["policy"]].append(row)
    table_rows = []
    for policy, pairs in sorted(variant_groups.items()):
        comparison = {"policy": policy, "n": len(pairs)}
        for metric in METRICS:
            deltas_for_metric = [row[f"{metric}_delta"] for row in pairs]
            comparison.update({
                f"{metric}_mean_delta": mean(deltas_for_metric),
                f"{metric}_min_delta": min(deltas_for_metric),
                f"{metric}_max_delta": max(deltas_for_metric),
                f"{metric}_negative_blocks": sum(value < 0 for value in deltas_for_metric),
            })
        table_rows.append(comparison)
    write_csv(output / "table3_paired_summary.csv", table_rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r["campaign"], r["rho"], r["policy"]].append(r)
    means = []
    for (kind, rho, policy), runs in groups.items():
        stats = {key: mean(r[key] for r in runs) for key in
                 (*METRICS, "target_rate", "capacity_tok_s", "throughput_over_capacity",
                  "ttft_slo_fraction", "mean_queue_s")}
        for key in METRICS:
            stats[f"{key}_sd"] = stdev(r[key] for r in runs) if len(runs) > 1 else None
        if kind == "stress":
            stats.update(late_probes=sum(r["long_after_last_short"] for r in runs),
                         total_probes=sum(r["long_probes"] for r in runs),
                         long_mean_queue_s=mean(r["long_mean_queue_s"] for r in runs),
                         max_long_wait_s=max(r["long_max_queue_s"] for r in runs))
        means.append({"campaign": kind, "rho": rho, "policy": policy, "n": len(runs), **stats})
    write_csv(output / "condition_means.csv", means)
    traces = [
        sample
        for row in rows if row["rho"] == 1.1
        for sample in queue_depth_rows(Path(row["run"]), row)
    ]
    write_csv(output / "queue_depth_samples.csv", traces)
    probe_rows = []
    for r in rows:
        if r["campaign"] != "stress":
            continue
        run = Path(r["run"])
        raw = read_jsonl(run / "raw_results.jsonl")
        queue = {p["id"]: p["queue_time_s"] for p in read_jsonl(run / "queue_times.jsonl")}
        last_short = max(p["send_time"] for p in raw if p["bucket"] == "short")
        for p in raw:
            if p["bucket"] == "long":
                probe_rows.append({
                    **{k: r[k] for k in ("block", "policy", "gpu_identity", "run")},
                    "id": p["id"], "ttft_s": p["first_token_time"] - p["send_time"],
                    "queue_s": queue[p["id"]],
                    "first_token_after_last_short": p["first_token_time"] > last_short,
                })
    write_csv(output / "stress_long_probes.csv", probe_rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batches", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(f"{len(analyze(args.batches, args.out))} audited same-container policy runs")


if __name__ == "__main__":
    main()
