"""
Storage figure for the CVPR paper (fig:pareto): Last vs. buffer memory per form.

Run from repo root, after bash mtil/scripts/fetch_results.sh:
    python mtil/scripts/plot_memory_curve.py
Saves figs/q3_memory.{pdf,png}. Runs not yet in results_sync are skipped and
named in the figure as running.
"""

import os

from plot_budget_curve import ROOT, metrics

OUT = "figs/q3_memory"

# (MB, run dir) per series. Images use RD, matching the feature and token runs.
SERIES = {
    "Images": [
        (151, "robustness/buf_1k_exrd"),
        (452, "robustness/buf_3k_exrd"),
        (828, "robustness/buf_5500_exrd"),
        (1656, "phase3_no_lora_v4"),
    ],
    "Embeddings": [
        (2.8, "featrep_v0_int4"),
        (5.6, "featrep_v0_int8"),
        (11.3, "featrep_v0"),
        (22.5, "featrep_v0_fp32"),
    ],
    "Tokens (PQ, tap 2)": [
        (69, "remind_l2_m32"),
        (277, "remind_l2_m128"),
        (1638, "remind_l2_m128_b65000"),
    ],
}


def load():
    data, running = {}, []
    for name, points in SERIES.items():
        data[name] = []
        for mb, run in points:
            if os.path.exists(os.path.join(ROOT, run, "task_summary.csv")):
                data[name].append((mb, metrics(run)[1]))
            else:
                running.append(run)
    return data, running


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    })
    style = {
        "Images": dict(color="#2a78d6", marker="o"),
        "Embeddings": dict(color="#eb6834", marker="s"),
        "Tokens (PQ, tap 2)": dict(color="#1f9d74", marker="^"),
    }
    data, running = load()

    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    for name, pts in data.items():
        ax.plot([p[0] for p in pts], [p[1] for p in pts], label=name, lw=1.2,
                ms=4.5, mec="white", mew=0.5, zorder=3, **style[name])
    ax.set_xscale("log")
    ticks = [3, 10, 30, 100, 300, 1000, 2000]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(2, 2400)
    ax.set_ylim(80.5, 87)
    ax.set_xlabel("Buffer memory (MB, log scale)", fontsize=8)
    ax.set_ylabel("Last accuracy (%)", fontsize=8)
    ax.grid(axis="y", color="#e4e3df", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.6, length=2.5, pad=2, labelsize=6.5)
    if running:
        ax.text(0.03, 0.05, f"{len(running)} points running", color="#52514e",
                fontsize=6.5, transform=ax.transAxes)
    ax.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.8)
    fig.tight_layout(pad=0.3)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {OUT}.pdf and .png; running: {running}")


if __name__ == "__main__":
    main()
