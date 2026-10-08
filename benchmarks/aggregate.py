"""Aggregate summary.json files from repeated runs into one ASCII table.

Usage:
    python -m benchmarks.aggregate results fcfs
Looks for results/<prefix>_<scenario>_run<N>/summary.json
"""
import glob
import json
import os
import sys

import numpy as np

HEADER = ["scen", "group", "n_req", "ttft_p50", "ttft_p95", "ttft_p99",
          "lat_p50", "lat_p95", "lat_p99", "tok/s", "runs"]


def load_runs(out_dir, prefix, scenario):
    pattern = os.path.join(out_dir, f"{prefix}_{scenario}_run*", "summary.json")
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def find_scenarios(out_dir, prefix):
    names = set()
    for path in glob.glob(os.path.join(out_dir, f"{prefix}_*_run*")):
        base = os.path.basename(path)[len(prefix) + 1:]
        names.add(base.rsplit("_run", 1)[0])
    return sorted(names)


def group_stats(run, group):
    if group == "overall":
        return run["overall"]
    return run["by_bucket"].get(group)


def mean_of(runs, group, metric, key):
    vals = []
    for r in runs:
        g = group_stats(r, group)
        if g is None:
            continue
        v = g[metric][key] if key else g[metric]
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def fmt(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"


def build_rows(out_dir, prefix):
    rows = []
    for scen in find_scenarios(out_dir, prefix):
        runs = load_runs(out_dir, prefix, scen)
        if not runs:
            continue
        groups = ["overall"] + sorted({b for r in runs for b in r["by_bucket"]})
        for group in groups:
            first = next(group_stats(r, group) for r in runs if group_stats(r, group))
            tput = mean_of(runs, group, "throughput_tokens_per_s", None) \
                if group == "overall" else None
            rows.append([
                scen, group, str(first["num_requests"]),
                fmt(mean_of(runs, group, "ttft", "p50")),
                fmt(mean_of(runs, group, "ttft", "p95")),
                fmt(mean_of(runs, group, "ttft", "p99")),
                fmt(mean_of(runs, group, "latency", "p50")),
                fmt(mean_of(runs, group, "latency", "p95")),
                fmt(mean_of(runs, group, "latency", "p99")),
                fmt(tput, 1), str(len(runs)),
            ])
    return rows


def render(rows):
    table = [HEADER] + rows
    widths = [max(len(r[i]) for r in table) for i in range(len(HEADER))]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines = [sep]
    for idx, r in enumerate(table):
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")
        if idx == 0:
            lines.append(sep)
    lines.append(sep)
    return "\n".join(lines)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "fcfs"
    rows = build_rows(out_dir, prefix)
    if not rows:
        print("no results found")
        return
    print("All times in seconds; values are means over runs.")
    print(render(rows))


if __name__ == "__main__":
    main()