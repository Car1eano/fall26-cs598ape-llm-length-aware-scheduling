import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("report_analysis")
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

with (ROOT / "summary_statistics.csv").open() as f:
    rows = list(csv.DictReader(f))

scenarios = [
    ("fairness_900_rate30", "90% Short"),
    ("balanced_900_rate30_seed123", "50% Short"),
]

policies = [
    "Native FCFS",
    "Queue-Only",
    "Aging alpha=0.1",
    "Aging alpha=0.5",
    "Hybrid",
]

metrics = [
    ("short_p95_ttft", "Short Request P95 TTFT", "Seconds"),
    ("long_p95_ttft", "Long Request P95 TTFT", "Seconds"),
    ("throughput", "Output Token Throughput", "Tokens / second"),
]

lookup = {
    (r["scenario"], r["policy"]): r
    for r in rows
}

for metric, title, ylabel in metrics:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (scenario, label) in zip(axes, scenarios):
        means = []
        errors = []

        for policy in policies:
            row = lookup[(scenario, policy)]
            means.append(float(row[metric + "_mean"]))
            errors.append(float(row[metric + "_sd"]))

        x = np.arange(len(policies))

        ax.bar(
            x,
            means,
            yerr=errors,
            capsize=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            ["FCFS", "Queue", "Aging\n0.1", "Aging\n0.5", "Hybrid"]
        )

        ax.set_title(label)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    fig.suptitle(title)
    fig.tight_layout()

    path = OUT / f"{metric}.pdf"
    fig.savefig(path, bbox_inches="tight")

    png_path = OUT / f"{metric}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    print("Created:", path)
    print("Created:", png_path)

print("\nAll figures generated.")
