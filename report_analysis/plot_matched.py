"""Render report-ready figures for matched and historical Modal data."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from report_analysis.matched_results import read_csv, read_jsonl

CORE = {"native_fcfs": ("FCFS", "#315d93"), "queue_only": ("Queue-Only", "#087d6e"),
        "aging_alpha05": ("Aging α=0.5", "#9a4870")}
LABELS = {
    **{p: label for p, (label, _) in CORE.items()},
    "aging_alpha01": "Aging α=0.1", "aging_alpha1": "Aging α=1",
    "aging_alpha2": "Aging α=2", "aging_alpha5": "Aging α=5", "aging_alpha10": "Aging α=10",
    "native_priority": "Native priority", "prompt_only": "Prompt only", "noisy_length": "±30% noise",
    "wait_promotion": "10 s promotion", "wait_limit_5": "5 s promotion", "hrrn": "HRRN",
    "hybrid": "Hybrid",
}


def save(fig, destination, name):
    for extension in ("pdf", "png"):
        fig.savefig(destination / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_axes(ax):
    ax.grid(alpha=.17)
    ax.spines[["right", "top"]].set_visible(False)


def decimal_log_ticks(axis):
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=8))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_formatter(NullFormatter())


def formats():
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold",
                         "axes.labelcolor": "#273340", "axes.edgecolor": "#5e6876",
                         "xtick.color": "#273340", "ytick.color": "#273340",
                         "legend.frameon": False, "pdf.fonttype": 42})


def pareto(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), layout="constrained")
    groups = defaultdict(list)
    for row in rows:
        if row["rho"]:
            groups[float(row["rho"]), row["policy"]].append(row)
    palette = plt.get_cmap("tab20")
    for ax, rho in zip(axes, (0.9, 1.1)):
        for i, (policy, label) in enumerate(LABELS.items()):
            cell = groups[rho, policy]
            if not cell:
                continue
            x = [float(r["long_p99_ttft_s"]) for r in cell]
            y = [float(r["short_p95_ttft_s"]) for r in cell]
            color = CORE[policy][1] if policy in CORE else palette(i)
            marker = ("o", "s", "^", "D", "v", "P", "X")[i % 7]
            ax.scatter(x, y, color=color, alpha=.33, s=17, marker=marker)
            ax.scatter(mean(x), mean(y), label=label, color=color, marker=marker,
                       s=50, edgecolor="white", linewidth=.65, zorder=4)
        ax.set(xscale="log", yscale="log", xlabel="Long P99 TTFT (s)",
               ylabel="Short P95 TTFT (s)", title=f"Balanced · ρ={rho:g}")
        decimal_log_ticks(ax.xaxis)
        decimal_log_ticks(ax.yaxis)
        style_axes(ax)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best" if rho == .9 else "upper left",
                      bbox_to_anchor=None if rho == .9 else (1.01, 1), fontsize=8)
    fig.suptitle("Short–long latency trade-off", fontweight="bold")
    fig.supxlabel("Small markers show blocks. Large markers show arithmetic means.", fontsize=9)
    save(fig, out, "matched_pareto")


def queue_depth(ax, rows):
    by_block = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["policy"] in CORE:
            by_block[row["block"]][row["policy"]].append(row)
    candidates = sorted(block for block, policies in by_block.items() if policies.keys() == CORE.keys())
    if not candidates:
        ax.set_visible(False)
        return
    for policy, (label, color) in CORE.items():
        samples = by_block[candidates[0]][policy]
        ax.plot([float(r["offset_s"]) for r in samples], [int(r["waiting"]) for r in samples],
                label=label, color=color, linewidth=1.5)
    replicate = candidates[0].rsplit("-r", 1)[-1]
    ax.set(title=f"Queue depth at ρ=1.1 · block {replicate}", xlabel="Seconds since first server sample",
           ylabel="Waiting requests")
    ax.legend(fontsize=8)
    style_axes(ax)


def rho_sweep(rows, queue_samples, out):
    metrics = (("short_p95_ttft_s", "Short P95 TTFT (s)"),
               ("long_p95_ttft_s", "Long P95 TTFT (s)"),
               ("throughput_tok_s", "Generated tokens/s"))
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.4), layout="constrained")
    labels = sorted({float(row["rho"]) for row in rows if row["rho"]})
    for ax, (key, title) in zip(axes.flat, metrics):
        for policy, (label, color) in CORE.items():
            cells = defaultdict(list)
            for r in rows:
                if r["rho"] and r["policy"] == policy:
                    cells[float(r["rho"])].append(float(r[key]))
            for rho, values in cells.items():
                ax.scatter([rho]*len(values), values, s=14, color=color, alpha=.35)
            x = sorted(cells)
            ax.plot(x, [mean(cells[rho]) for rho in x], color=color, marker="o", label=label)
        ax.set(xlabel="Offered load ρ", title=title)
        ax.set_xticks(labels, [f"{value:g}" for value in labels])
        if "ttft" in key:
            ax.set_yscale("log")
            decimal_log_ticks(ax.yaxis)
        style_axes(ax)
    axes[0, 0].legend(fontsize=8)
    queue_depth(axes[1, 1], queue_samples)
    fig.suptitle("Capacity-calibrated load sweep · balanced mix", fontweight="bold")
    save(fig, out, "matched_rho_sweep")


def empirical_cdf(ax, values, label, color):
    ordered = np.sort(values)
    if len(ordered):
        ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), color=color,
                label=label, where="post", linewidth=1.85)


def cdf(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True, layout="constrained")
    selected = [r for r in rows if r["rho"] == "1.1" and r["policy"] in CORE]
    for ax, bucket in zip(axes, ("short", "long")):
        for policy, (label, color) in CORE.items():
            values = []
            for r in selected:
                if r["policy"] == policy:
                    values.extend(row["first_token_time"] - row["send_time"]
                                  for row in read_jsonl(Path(r["run"]) / "raw_results.jsonl")
                                  if row["bucket"] == bucket)
            empirical_cdf(ax, values, label, color)
        ax.set(xscale="log", xlabel="TTFT (s)", title=f"{bucket.title()} requests")
        decimal_log_ticks(ax.xaxis)
        style_axes(ax)
    axes[0].set_ylabel("Cumulative fraction")
    axes[0].legend(fontsize=8)
    fig.suptitle("Request-level TTFT · balanced, ρ=1.1", fontweight="bold")
    save(fig, out, "matched_ttft_cdf")


def fairness(legacy, out):
    # Historical slowdown uses the isolated-service calibration already in
    # the CSV. The plot keeps the pooled historical environment label visible.
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.7), layout="constrained")
    for ax, (key, label) in zip(axes.flat, [
        ("short_mean_slowdown", "Short mean slowdown"),
        ("long_mean_slowdown", "Long mean slowdown"),
        ("jain_inverse_class_mean_slowdown", "Jain index"),
        ("ttft_slo_fraction", "TTFT < 1 s fraction"),
    ]):
        rates = (2, 5, 10, 15, 20, 30)
        for i, (policy, (name, color)) in enumerate(CORE.items()):
            selected = [
                [float(r[key]) for r in legacy if r["mix"] == "balanced"
                 and r["max_num_seqs"] == "4" and r["policy"] == policy
                 and float(r["target_req_s"]) == rate and r.get(key)]
                for rate in rates
            ]
            x = [j + (i-1)*.24 for j, values in enumerate(selected) if values]
            means = [mean(values) for values in selected if values]
            ax.bar(x, means, width=.22, color=color, label=name)
        ax.set_title(label)
        ax.set_xticks(range(len(rates)), [str(r) for r in rates])
        if key in ("short_mean_slowdown", "long_mean_slowdown"):
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.05)
        style_axes(ax)
    axes[0, 1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Historical balanced run metrics · cap 4 · pooled GPU environments",
                 fontweight="bold")
    fig.supxlabel("Target arrival rate (requests/s)")
    save(fig, out, "historical_slowdown_fairness")


def stress(rows, out):
    fig, ax = plt.subplots(figsize=(7.9, 4), layout="constrained")
    colors = {**{p: color for p, (_, color) in CORE.items()},
              "wait_limit_5": "#65a319", "wait_promotion": "#7967aa", "hrrn": "#d96110"}
    for policy, color in colors.items():
        values = [
            raw["first_token_time"] - raw["send_time"]
            for r in rows if r["policy"] == policy
            for raw in read_jsonl(Path(r["run"]) / "raw_results.jsonl")
            if raw["bucket"] == "long"
        ]
        title = LABELS[policy]
        empirical_cdf(ax, values, title, color)
    ax.set(xlabel="Long-probe TTFT (s)", ylabel="Cumulative fraction",
           title="Long probes under sustained short arrivals")
    style_axes(ax)
    ax.legend(fontsize=9)
    save(fig, out, "matched_long_probe_cdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched", type=Path)
    parser.add_argument("--legacy", type=Path, default=Path("results/modal_analysis/individual_runs.csv"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    formats()
    fairness(read_csv(args.legacy), args.out)
    if args.matched:
        rows = read_csv(args.matched)
        queue_path = args.matched.with_name("queue_depth_samples.csv")
        queue_samples = read_csv(queue_path) if queue_path.exists() else []
        if any(r["rho"] for r in rows):
            pareto(rows, args.out)
            rho_sweep(rows, queue_samples, args.out)
        if any(r["rho"] == "1.1" for r in rows):
            cdf(rows, args.out)
        probes = [r for r in rows if r["campaign"] == "stress"]
        if probes:
            stress(probes, args.out)


if __name__ == "__main__":
    main()
