import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("report_analysis/raw_results")
OUT = Path("report_analysis")

SCENARIOS = [
    "fairness_900_rate30",
    "balanced_900_rate30_seed123",
]

POLICIES = {
    "native_fcfs": "Native FCFS",
    "queue_only": "Queue-Only",
    "adaptive_alpha05": "Aging alpha=0.5",
    "adaptive_alpha01": "Aging alpha=0.1",
    "hybrid_conditional": "Hybrid",
}

# The original fairness Queue-Only run1 was anomalous.
# Keep it in the audit, but use the documented retest
# for the primary five-run comparison.
EXCLUDED = {
    ("fairness_900_rate30", "queue_only_run1"):
        "Original anomalous run; retest used",
}

EXPECTED = {
    "fairness_900_rate30": (900, 24144, 813, 87),
    "balanced_900_rate30_seed123": (900, 64240, 455, 445),
}

METRICS = [
    "throughput",
    "short_p95_ttft",
    "long_p95_ttft",
    "short_mean_latency",
    "long_mean_latency",
]

rows = []
audit = []

for scenario in SCENARIOS:
    folder = ROOT / scenario

    for path in sorted(folder.glob("*/summary.json")):
        run = path.parent.name
        policy = None

        for prefix, name in POLICIES.items():
            if run.startswith(prefix + "_run"):
                policy = name
                break

        if policy is None:
            audit.append([scenario, run, "OTHER", "Not primary policy"])
            continue

        data = json.loads(path.read_text())
        overall = data["overall"]
        short = data["by_bucket"]["short"]
        long = data["by_bucket"]["long"]

        expected = EXPECTED[scenario]
        actual = (
            overall["num_requests"],
            overall["output_tokens"],
            short["num_requests"],
            long["num_requests"],
        )

        reason = EXCLUDED.get((scenario, run))

        if actual != expected:
            reason = f"Workload mismatch: {actual}"

        if overall["failed_requests"] != 0:
            reason = "Failed requests present"

        if reason:
            audit.append([scenario, run, "EXCLUDED", reason])
            continue

        rows.append({
            "scenario": scenario,
            "policy": policy,
            "run": run,
            "throughput": overall["throughput_tokens_per_s"],
            "short_p95_ttft": short["ttft"]["p95"],
            "long_p95_ttft": long["ttft"]["p95"],
            "short_mean_latency": short["latency_mean"],
            "long_mean_latency": long["latency_mean"],
        })

        audit.append([scenario, run, "INCLUDED", "Validated"])

def save_csv(path, fieldnames, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

save_csv(
    OUT / "individual_runs.csv",
    ["scenario", "policy", "run"] + METRICS,
    rows,
)

groups = defaultdict(list)

for row in rows:
    groups[(row["scenario"], row["policy"])].append(row)

summary = []

for (scenario, policy), group in sorted(groups.items()):
    result = {
        "scenario": scenario,
        "policy": policy,
        "n": len(group),
    }

    for metric in METRICS:
        values = [r[metric] for r in group]
        result[metric + "_mean"] = statistics.mean(values)
        result[metric + "_sd"] = (
            statistics.stdev(values) if len(values) > 1 else 0
        )

    summary.append(result)

columns = ["scenario", "policy", "n"]

for metric in METRICS:
    columns.extend([metric + "_mean", metric + "_sd"])

save_csv(OUT / "summary_statistics.csv", columns, summary)

with (OUT / "run_audit.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["scenario", "run", "status", "reason"])
    writer.writerows(audit)

print("\n===== FINAL COMPARISON =====")

for scenario in SCENARIOS:
    print(f"\n{scenario}")
    print(
        f"{'Policy':<20} {'N':>3} "
        f"{'Throughput':>15} "
        f"{'Short P95':>15} "
        f"{'Long P95':>15}"
    )

    for row in summary:
        if row["scenario"] != scenario:
            continue

        print(
            f"{row['policy']:<20} "
            f"{row['n']:>3} "
            f"{row['throughput_mean']:>8.2f}"
            f" +/- {row['throughput_sd']:<5.2f} "
            f"{row['short_p95_ttft_mean']:>8.3f}"
            f" +/- {row['short_p95_ttft_sd']:<5.3f} "
            f"{row['long_p95_ttft_mean']:>8.3f}"
            f" +/- {row['long_p95_ttft_sd']:<5.3f}"
        )

print("\n===== AUDIT =====")
for item in audit:
    if item[2] != "INCLUDED":
        print(" | ".join(item))

print("\nFiles created:")
print("report_analysis/individual_runs.csv")
print("report_analysis/summary_statistics.csv")
print("report_analysis/run_audit.csv")
