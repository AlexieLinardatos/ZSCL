"""
Replay budget figure for the CVPR paper (fig:q2a): Last and Transfer vs. memory.

Run from repo root, after bash mtil/scripts/fetch_results.sh:
    python mtil/scripts/plot_budget_curve.py
Saves figs/q2a_budget.{pdf,png}
"""

import csv
import os

ROOT = "results_sync/ckpt/11task"
KB_PER_IMAGE = 150.5  # uint8 224x224x3
OUT = "figs/q2a_budget"

# (samples, run dir) per series. All use the v4 per-task iteration schedule.
SERIES = {
    "Replay + ZSCL": [
        (1000, "robustness/buf_1k_nord"),
        (3000, "robustness/buf_3k_nord"),
        (5500, "robustness/buf_5500_nord"),
        (11000, "robustness/buf_11k_nord"),
    ],
    "Replay + ZSCL + RD": [
        (1000, "robustness/buf_1k_exrd"),
        (3000, "robustness/buf_3k_exrd"),
        (5500, "robustness/buf_5500_exrd"),
        (11000, "phase3_no_lora_v4"),
    ],
}

# Read from the Nibi output on 2026-11-05; used only until fetch_results.sh has
# copied this run into results_sync. (Transfer, Last)
PINNED = {"robustness/buf_11k_nord": (68.93, 86.87)}

# Reported Order I numbers.
GENERATIVE = {"Last": (86.0, "GIFT / LoRA-Loop"), "Transfer": (69.3, "GIFT")}


def metrics(run):
    """(Transfer, Last): per-task upper-triangle mean and final-row mean."""
    path = os.path.join(ROOT, run, "task_summary.csv")
    if not os.path.exists(path):
        print(f"WARNING: {path} missing, using pinned values")
        return PINNED[run]
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    order = [r["task_name"] for r in rows]
    per_task = [
        sum(float(r[task]) for r in rows[:j]) / j
        for j, task in enumerate(order) if j > 0
    ]
    last = [float(rows[-1][task]) for task in order]
    return sum(per_task) / len(per_task), sum(last) / len(last)


def load():
    return {
        name: [(n * KB_PER_IMAGE / 1000, *metrics(run)) for n, run in points]
        for name, points in SERIES.items()
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    })
    style = {
        "Replay + ZSCL": dict(color="#2a78d6", marker="o"),
        "Replay + ZSCL + RD": dict(color="#eb6834", marker="s"),
    }
    data = load()
    ticks = [p[0] for p in data["Replay + ZSCL"]]

    fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.75))
    for ax, metric, col, ylim in (
        (axes[0], "Last", 2, (80.5, 87.5)),
        (axes[1], "Transfer", 1, (68.0, 69.5)),
    ):
        ref, label = GENERATIVE[metric]
        ax.axhline(ref, color="#52514e", lw=0.7, ls="--", zorder=1)
        ax.text(ticks[0], ref, label, color="#52514e", fontsize=5.5,
                va="bottom", ha="left")
        for name, pts in data.items():
            ax.plot([p[0] for p in pts], [p[col] for p in pts], label=name,
                    lw=1.2, ms=4, mec="white", mew=0.5, zorder=3, **style[name])
        ax.set_xscale("log")
        ax.set_xticks(ticks, [f"{int(t + 0.5)}" for t in ticks], fontsize=6.5)
        ax.minorticks_off()
        ax.set_ylim(*ylim)
        ax.set_title(metric, fontsize=8, pad=3)
        ax.grid(axis="y", color="#e4e3df", lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(width=0.6, length=2.5, pad=2, labelsize=6.5)
    fig.supxlabel("Buffer memory (MB, log scale)", fontsize=8, y=0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=7, handlelength=1.8, columnspacing=1.2,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(pad=0.3, rect=(0, 0.04, 1, 0.90), w_pad=1.0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {OUT}.pdf and .png")


if __name__ == "__main__":
    for name, pts in load().items():
        for mb, transfer, last in pts:
            print(f"{name:<20} {mb:7.1f} MB  Transfer {transfer:.2f}  Last {last:.2f}")
    main()
