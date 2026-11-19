"""Summarize the experience of long probes under sustained short traffic."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    batch = args.batch
    results = []
    for line in (batch / "progress.jsonl").read_text().splitlines():
        record = json.loads(line)
        if not record["ok"]:
            continue
        path = batch.parent / record["path"]
        rows = [json.loads(r) for r in (path / "raw_results.jsonl").read_text().splitlines()]
        queues = {r["id"]: r["queue_time_s"] for r in
                  map(json.loads, (path / "queue_times.jsonl").read_text().splitlines())}
        last_short_send = max(r["send_time"] for r in rows if r["bucket"] == "short")
        probes = [r for r in rows if r["bucket"] == "long"]
        waiting = [queues[r["id"]] for r in probes]
        results.append({
            "block": record["block"], "policy": record["policy"], "long_probes": len(probes),
            "long_first_token_after_short_stream": sum(
                r["first_token_time"] > last_short_send for r in probes
            ),
            "long_mean_queue_s": mean(waiting), "long_max_queue_s": max(waiting),
            "long_max_ttft_s": max(r["first_token_time"] - r["send_time"] for r in probes),
        })
    with (batch / "long_probe_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
