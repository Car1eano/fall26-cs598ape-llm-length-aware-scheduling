"""Render report figures from audited Modal run-level CSV data."""

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

POLICIES = {
    "native_fcfs": ("FCFS", "#355a92"),
    "queue_only": ("Queue-Only", "#137d68"),
    "aging_alpha05": ("Aging α=0.5", "#8e4c77"),
}
MIXES = {"short_heavy": "Short-heavy", "balanced": "Balanced"}


def policy_legend():
    return [Line2D([], [], color=color, marker="o", label=label)
            for label, color in POLICIES.values()]


def plot_panel(ax, rows, levels, dimension, metric, symbols=None):
    for index, (policy, (_, color)) in enumerate(POLICIES.items()):
        means, positions = [], []
        for x, level in enumerate(levels):
            cell = sorted(
                [row for row in rows if float(row[dimension]) == level and row["policy"] == policy],
                key=lambda row: int(row["replicate"]),
            )
            if not cell:
                continue
            position = x + (index - 1) * 0.20
            values = [float(row[metric]) for row in cell]
            means.append(mean(values))
            positions.append(position)
            for j, (row, value) in enumerate(zip(cell, values)):
                offset = (j - (len(cell) - 1) / 2) * 0.026
                symbol = symbols[row["gpu_environment"]] if symbols else "o"
                ax.scatter(position + offset, value, s=16, marker=symbol, color=color,
                           alpha=0.54, zorder=3)
        ax.plot(positions, means, color=color, marker="o", markersize=4,
                markerfacecolor="white", linewidth=1.35, zorder=4)
    ax.set_xticks(range(len(levels)), [str(n) for n in levels])
    ax.set_xlim(-0.48, len(levels) - 0.52)
    ax.grid(axis="y", color="#dbe2e8", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.csv_file.open(newline="") as f:
        rows = list(csv.DictReader(f))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if r["policy"] in POLICIES]
    main_runs = [r for r in rows if r["max_num_seqs"] == "4" and r["mix"] in MIXES]
    rates = sorted({int(float(r["target_req_s"])) for r in main_runs})
    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold", "axes.labelcolor": "#273340",
                         "xtick.color": "#273340", "ytick.color": "#273340"})

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.1, 4.7), layout="constrained")
    for column, (mix, label) in enumerate(MIXES.items()):
        subset = [r for r in main_runs if r["mix"] == mix]
        axes[0][column].set_title(label)
        for row_index, bucket in enumerate(("short", "long")):
            ax = axes[row_index][column]
            plot_panel(ax, subset, rates, "target_req_s", f"{bucket}_p95_ttft_s")
            ax.set_yscale("log")
            if column == 0:
                ax.set_ylabel(f"{bucket.title()} P95 TTFT (s)")
    fig.supxlabel("Target arrival rate (requests/s)  ·  dots: individual runs; lines: means", fontsize=9)
    axes[0][0].legend(handles=policy_legend(), loc="upper left", frameon=False, fontsize=7)
    fig.savefig(args.out / "modal_ttft_by_load.png", dpi=330)
    plt.close(fig)

    environments = sorted({r["gpu_environment"] for r in main_runs})
    symbols = dict(zip(environments, ("o", "s", "^", "D", "v"), strict=False))
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.1, 3.3))
    for ax, (mix, label) in zip(axes, MIXES.items()):
        subset = [r for r in main_runs if r["mix"] == mix]
        plot_panel(ax, subset, rates, "target_req_s", "throughput_tok_s", symbols)
        ax.set_title(label)
    axes[0].set_ylabel("Generated tokens/s")
    fig.supxlabel("Target arrival rate (requests/s)", y=0.195, fontsize=9)
    fig.legend(handles=policy_legend(), loc="lower center", bbox_to_anchor=(0.5, 0.105),
               ncol=3, frameon=False, fontsize=8)
    env_handles = [Line2D([], [], color="#475566", marker=symbols[env], linestyle="none",
                          label=env.removeprefix("NVIDIA L40S, "))
                   for env in environments]
    fig.legend(handles=env_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               title="Driver and visible memory", ncol=3, frameon=False, fontsize=7, title_fontsize=7)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.31, top=0.89, wspace=0.13)
    fig.savefig(args.out / "modal_throughput_by_load.png", dpi=330)
    plt.close(fig)

    subset = [r for r in rows if r["mix"] == "balanced" and float(r["target_req_s"]) == 5]
    caps = sorted({int(r["max_num_seqs"]) for r in subset})
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.6, 2.9), layout="constrained")
    for ax, bucket in zip(axes, ("short", "long")):
        plot_panel(ax, subset, caps, "max_num_seqs", f"{bucket}_p95_ttft_s")
        ax.set_title(f"{bucket.title()} requests")
        ax.set_yscale("log")
        ax.set_xlabel("Maximum concurrent sequences")
    axes[0].set_ylabel("P95 TTFT (s)")
    axes[0].legend(handles=policy_legend(), loc="upper right", frameon=False, fontsize=7)
    fig.savefig(args.out / "modal_cap_sensitivity.png", dpi=330)
    plt.close(fig)


if __name__ == "__main__":
    main()
