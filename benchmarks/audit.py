"""Audit raw results of runs against their workload file.

Usage:
    python -m benchmarks.audit results christine_fcfs_cap4

For every results/<tag>_{A,B}_run<N>/ it checks: row count, ids, errors,
output_tokens vs max_tokens, timestamp order, and that summary.json
matches a recomputation from raw_results.jsonl.
"""
import glob
import json
import os
import sys

from benchmarks.metrics import summarize


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def audit_run(run_dir, workload):
    rows = read_jsonl(os.path.join(run_dir, "raw_results.jsonl"))
    with open(os.path.join(run_dir, "summary.json")) as f:
        stored = json.load(f)

    ids = [r["id"] for r in rows]
    bad_ids = len(ids) - len(set(ids)) + sum(1 for i in ids if i not in workload)
    failed = sum(1 for r in rows if r.get("error") is not None)
    mismatches = sum(
        1 for r in rows
        if r["id"] in workload and r.get("output_tokens") != workload[r["id"]]["max_tokens"]
    )
    bad_time = sum(
        1 for r in rows
        if not (
            isinstance(r.get("first_token_time"), (int, float))
            and r["send_time"] <= r["first_token_time"] <= r["finish_time"]
        )
    )
    summary_ok = summarize(rows) == stored
    ok = (
        len(rows) == len(workload)
        and bad_ids == 0
        and failed == 0
        and mismatches == 0
        and bad_time == 0
        and summary_ok
    )
    return ok, len(rows), failed, mismatches, bad_time, bad_ids, summary_ok


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    tag = sys.argv[2] if len(sys.argv) > 2 else "fcfs_cap4"

    all_ok = True
    total_rows = 0
    print("run | rows | failed | tok_mismatch | bad_time | bad_ids | summary | ok")
    for scen in ("A", "B"):
        wl_path = os.path.join(out_dir, f"{tag}_workload_{scen}.jsonl")
        if not os.path.exists(wl_path):
            print(f"missing workload file: {wl_path}")
            all_ok = False
            continue
        workload = {r["id"]: r for r in read_jsonl(wl_path)}
        run_dirs = sorted(glob.glob(os.path.join(out_dir, f"{tag}_{scen}_run[0-9]*")))
        if not run_dirs:
            print(f"no runs found for scenario {scen}")
            all_ok = False
        for d in run_dirs:
            ok, n, failed, mism, bad_t, bad_i, s_ok = audit_run(d, workload)
            total_rows += n
            all_ok = all_ok and ok
            print(
                f"{os.path.basename(d)} | {n} | {failed} | {mism} | "
                f"{bad_t} | {bad_i} | {'match' if s_ok else 'DIFF'} | "
                f"{'OK' if ok else 'FAIL'}"
            )
    print(f"total rows: {total_rows}")
    print("ALL OK" if all_ok else "PROBLEMS FOUND")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()