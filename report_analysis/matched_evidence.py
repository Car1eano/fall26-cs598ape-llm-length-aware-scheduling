"""Build the numerical claim-to-evidence table for the environment correction."""

import argparse
from pathlib import Path
from statistics import mean

from report_analysis.matched_results import read_csv, write_csv

LEGACY = Path("results/modal_analysis/individual_runs.csv")


def index(rows, *fields):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in fields)
        groups.setdefault(key, []).append(row)
    return groups


def reproduction_evidence(directory):
    rows = []
    old_base = [
        r for r in read_csv(LEGACY)
        if r["mix"] == "balanced" and float(r["target_req_s"]) == 5
        and r["max_num_seqs"] == "4"
    ]
    for (policy, env), group in sorted(index(old_base, "policy", "gpu_environment").items()):
        if policy not in ("native_fcfs", "prompt_only", "aging_alpha1", "aging_alpha5"):
            continue
        summary = ", ".join(
            f"{name} {mean(float(r[field]) for r in group):.2f}"
            for name, field in (("tokens/s", "throughput_tok_s"), ("mean queue s", "mean_queue_s"),
                                ("short P95 s", "short_p95_ttft_s"), ("long P95 s", "long_p95_ttft_s"))
        )
        rows.append({
            "claim": f"Historical environment assignment for {policy}",
            "estimate": f"{env}, n={len(group)}, {summary}",
            "evidence_file": str(LEGACY),
            "blocks": ", ".join(r["block"] for r in group),
            "runs": " | ".join(r["run"] for r in group),
        })

    paired = directory / "paired_differences.csv"
    if paired.is_file():
        for (kind, rho, policy), group in sorted(index(read_csv(paired), "campaign", "rho", "policy").items()):
            if kind == "stress":
                continue
            metrics = [f"{name} Δ {mean(float(r[key]) for r in group):+.2f}"
                       for name, key in (("short P95 s", "short_p95_ttft_s_delta"),
                                         ("long P95 s", "long_p95_ttft_s_delta"),
                                         ("long P99 s", "long_p99_ttft_s_delta"),
                                         ("tokens/s", "throughput_tok_s_delta"))]
            rows.append({
                "claim": f"{policy} versus within-container FCFS at rho={rho}",
                "estimate": f"n={len(group)}, " + ", ".join(metrics),
                "evidence_file": str(paired),
                "blocks": ", ".join(r["block"] for r in group),
                "runs": " | ".join(f"{r['run']} vs. {r['fcfs_run']}" for r in group),
            })

    matched = directory / "individual_runs.csv"
    if matched.is_file():
        runs = read_csv(matched)
        fcfs = [r for r in runs if r["policy"] == "native_fcfs"]
        capacities = [float(r["capacity_tok_s"]) for r in fcfs]
        rows.append({
            "claim": "Calibrated service capacity varied across completed L40S blocks",
            "estimate": f"n={len(fcfs)}, {min(capacities):.2f}–{max(capacities):.2f} tokens/s",
            "evidence_file": str(directory / "block_audit.csv"),
            "blocks": ", ".join(r["block"] for r in fcfs),
            "runs": " | ".join(r["run"] for r in fcfs),
        })
        for (rho,), group in sorted(index([r for r in fcfs if r["rho"]], "rho").items()):
            short = mean(float(r["short_p95_ttft_s"]) for r in group)
            long = mean(float(r["long_p95_ttft_s"]) for r in group)
            throughput = mean(float(r["throughput_tok_s"]) for r in group)
            relative = mean(float(r["throughput_over_capacity"]) for r in group)
            rows.append({
                "claim": f"Balanced FCFS at rho={rho}",
                "estimate": f"n={len(group)}, short P95 {short:.2f} s, long P95 {long:.2f} s, "
                            f"{throughput:.2f} tokens/s, {relative:.3f} times calibrated capacity",
                "evidence_file": str(matched),
                "blocks": ", ".join(r["block"] for r in group),
                "runs": " | ".join(r["run"] for r in group),
            })
        stress = [r for r in runs if r["campaign"] == "stress"]
        for (policy,), group in sorted(index(stress, "policy").items()):
            late = sum(int(r["long_after_last_short"]) for r in group)
            total = sum(int(r["long_probes"]) for r in group)
            avg = mean(float(r["long_mean_queue_s"]) for r in group)
            rows.append({
                "claim": f"Stress test long probes under {policy}",
                "estimate": f"{late}/{total} first tokens after last short send, mean queue {avg:.2f} s",
                "evidence_file": str(matched),
                "blocks": ", ".join(r["block"] for r in group),
                "runs": " | ".join(r["run"] for r in group),
            })
    write_csv(directory / "claim_evidence.csv", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    reproduction_evidence(args.directory)


if __name__ == "__main__":
    main()
