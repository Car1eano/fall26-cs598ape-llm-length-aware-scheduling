import csv
import json
import statistics
from pathlib import Path

ROOT = Path("report_analysis/raw_results/fairness_900_rate30")
OUTPUT = Path("report_analysis/sensitivity_analysis.csv")

RUNS = [
    "queue_only_run1",
    "queue_only_run1_retest",
    "queue_only_run2",
    "queue_only_run3",
    "queue_only_run4",
    "queue_only_run5",
]

GROUPS = {
    "Original five runs": [
        "queue_only_run1",
        "queue_only_run2",
        "queue_only_run3",
        "queue_only_run4",
        "queue_only_run5",
    ],
    "Retest five runs": [
        "queue_only_run1_retest",
        "queue_only_run2",
        "queue_only_run3",
        "queue_only_run4",
        "queue_only_run5",
    ],
    "All six runs": RUNS,
}

data = {}

for run in RUNS:
    path = ROOT / run / "summary.json"
    result = json.loads(path.read_text())

    data[run] = {
        "throughput": result["overall"]["throughput_tokens_per_s"],
        "short_p95_ttft": result["by_bucket"]["short"]["ttft"]["p95"],
        "long_p95_ttft": result["by_bucket"]["long"]["ttft"]["p95"],
    }

rows = []

for group, runs in GROUPS.items():
    for metric in [
        "throughput",
        "short_p95_ttft",
        "long_p95_ttft",
    ]:
        values = [data[run][metric] for run in runs]

        rows.append({
            "group": group,
            "metric": metric,
            "n": len(values),
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values),
        })

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["group", "metric", "n", "mean", "sd"],
    )
    writer.writeheader()
    writer.writerows(rows)

print("===== SENSITIVITY ANALYSIS =====")

for row in rows:
    print(
        f'{row["group"]:<20} '
        f'{row["metric"]:<16} '
        f'N={row["n"]} '
        f'{row["mean"]:.4f} +/- {row["sd"]:.4f}'
    )

print(f"\nSaved: {OUTPUT}")
