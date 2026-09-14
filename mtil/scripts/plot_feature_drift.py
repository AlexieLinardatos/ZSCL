"""
Feature drift figure: how stale stored embeddings become, and what it costs.

Two panels, because the interesting claim is the link between them:

  (a) Drift trajectories. For every stored task, the cosine between its features
      as originally stored and what the model produces for those same probe
      images after each subsequent task. One line per stored task, starting at
      its own storage point. 1.0 means no drift.

  (b) Drift vs. damage. Final drift per task against that task's accuracy
      difference between feature replay and pixel replay. If drift is what costs
      accuracy, points fall on a line through the origin — and the task that
      drifted most is the task that lost most.

Series are coloured by task position on a single-hue ramp rather than as
categorical hues: eleven tasks is well past the point where distinct hues stay
distinguishable, and position is ordered data, so a ramp encodes it honestly.
The one task carrying the story is drawn in a contrasting accent.

Requires a run with --drift_probe_size > 0, which writes feature_drift.csv.

Usage (from mtil/):
    python scripts/plot_feature_drift.py \
        --drift   ckpt/11task/featrep_v0/feature_drift.csv \
        --feature ckpt/11task/featrep_v0/task_summary.csv \
        --pixel   ckpt/11task/phase3_no_lora_v4/task_summary.csv \
        --out     feature_drift

Outputs <out>.pdf and <out>.png.
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK_ORDER = [
    "Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT",
    "Flowers", "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397",
]

ABBREV = {
    "Aircraft": "AC", "Caltech101": "Cal", "CIFAR100": "C100",
    "DTD": "DTD", "EuroSAT": "ES", "Flowers": "Fl", "Food": "Fo",
    "MNIST": "MN", "OxfordPet": "Pet", "StanfordCars": "Cars", "SUN397": "SUN",
}

# Single-hue sequential ramp, darkest = stored earliest. Clipped below 0.35 so
# the lightest line still reads against white.
RAMP = plt.cm.Blues
RAMP_LO, RAMP_HI = 0.95, 0.35
ACCENT = "#D95F02"   # orange; CVD-safe against every step of a blue ramp
INK = "#222222"
MUTED = "#777777"


def ramp_color(task_idx, n_tasks):
    t = task_idx / max(1, n_tasks - 1)
    return RAMP(RAMP_LO + (RAMP_HI - RAMP_LO) * t)


def load_csv(path, required=True):
    if not os.path.exists(path):
        msg = f"missing {path}"
        if required:
            sys.exit(f"ERROR: {msg}")
        print(f"WARNING: {msg}", file=sys.stderr)
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drift", required=True,
                   help="feature_drift.csv from a --drift_probe_size run.")
    p.add_argument("--feature", default=None,
                   help="task_summary.csv of the feature-replay run (panel b).")
    p.add_argument("--pixel", default=None,
                   help="task_summary.csv of the pixel-replay baseline (panel b).")
    p.add_argument("--accent-task", default="Aircraft",
                   help="Task drawn in the accent colour.")
    p.add_argument("--out", default="feature_drift")
    return p.parse_args()


def panel_trajectories(ax, drift_rows, accent_task):
    """Panel (a): one drift curve per stored task."""
    by_task = {}
    for r in drift_rows:
        by_task.setdefault(int(r["stored_task_idx"]), []).append(r)

    n_tasks = len(TASK_ORDER)
    for stored_idx in sorted(by_task):
        rows = sorted(by_task[stored_idx], key=lambda r: int(r["measured_after_idx"]))
        xs = [int(r["measured_after_idx"]) for r in rows]
        ys = [float(r["mean_cos"]) for r in rows]
        if not xs:
            continue

        name = TASK_ORDER[stored_idx] if stored_idx < n_tasks else str(stored_idx)
        is_accent = name == accent_task
        ax.plot(
            xs, ys,
            color=ACCENT if is_accent else ramp_color(stored_idx, n_tasks),
            lw=2.0 if is_accent else 1.4,
            marker="o", markersize=3.5 if is_accent else 2.5,
            zorder=5 if is_accent else 3,
            solid_capstyle="round",
        )
        # Direct-label only the extremes; an eleven-entry legend would be noise.
        if is_accent or stored_idx == max(by_task):
            ax.annotate(
                ABBREV.get(name, name),
                xy=(xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                fontsize=8, va="center",
                color=ACCENT if is_accent else MUTED,
                fontweight="bold" if is_accent else "normal",
            )

    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels([ABBREV[t] for t in TASK_ORDER], rotation=45, ha="right")
    ax.set_xlabel("Measured after training task")
    ax.set_ylabel("Cosine to stored feature")
    ax.set_title("(a) Stored features go stale", fontsize=10, loc="left")
    ax.grid(axis="y", color="#E6E6E6", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":", zorder=1)


def panel_drift_vs_damage(ax, drift_rows, feat_rows, pix_rows, accent_task):
    """Panel (b): final drift per task against accuracy lost versus pixels."""
    # Final drift = the last measurement of each stored task.
    final_drift = {}
    for r in drift_rows:
        idx = int(r["stored_task_idx"])
        step = int(r["measured_after_idx"])
        if idx not in final_drift or step > final_drift[idx][0]:
            final_drift[idx] = (step, float(r["mean_cos"]))

    feat_last, pix_last = feat_rows[-1], pix_rows[-1]
    n_tasks = len(TASK_ORDER)

    xs, ys, labels, colors = [], [], [], []
    for idx, name in enumerate(TASK_ORDER):
        if idx not in final_drift:
            continue
        if not feat_last.get(name) or not pix_last.get(name):
            continue
        xs.append(final_drift[idx][1])
        ys.append(float(feat_last[name]) - float(pix_last[name]))
        labels.append(name)
        colors.append(ACCENT if name == accent_task
                      else ramp_color(idx, n_tasks))

    if not xs:
        ax.text(0.5, 0.5, "no overlapping tasks", ha="center", va="center",
                transform=ax.transAxes, color=MUTED)
        return

    ax.axhline(0.0, color=MUTED, lw=0.8, ls=":", zorder=1)
    for x, y, name, c in zip(xs, ys, labels, colors):
        accent = name == accent_task
        ax.scatter(x, y, s=48 if accent else 30, color=c,
                   edgecolor="white", linewidth=0.8,
                   zorder=5 if accent else 3)
        ax.annotate(
            ABBREV.get(name, name), xy=(x, y), xytext=(5, 3),
            textcoords="offset points", fontsize=8,
            color=ACCENT if accent else MUTED,
            fontweight="bold" if accent else "normal",
        )

    if len(xs) >= 3:
        r = np.corrcoef(xs, ys)[0, 1]
        ax.annotate(f"r = {r:.2f}", xy=(0.04, 0.06), xycoords="axes fraction",
                    fontsize=9, color=INK)

    ax.set_xlabel("Final cosine to stored feature")
    ax.set_ylabel("Accuracy vs. pixel replay (pp)")
    ax.set_title("(b) Drift predicts the damage", fontsize=10, loc="left")
    ax.grid(color="#E6E6E6", lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def main():
    args = parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#999999",
        "text.color": INK,
        "axes.labelcolor": INK,
    })

    drift_rows = load_csv(args.drift)
    feat_rows = load_csv(args.feature, required=False) if args.feature else None
    pix_rows = load_csv(args.pixel, required=False) if args.pixel else None
    two_panel = bool(feat_rows and pix_rows)

    if two_panel:
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
        fig.subplots_adjust(wspace=0.30)
        panel_trajectories(axes[0], drift_rows, args.accent_task)
        panel_drift_vs_damage(axes[1], drift_rows, feat_rows, pix_rows,
                              args.accent_task)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4))
        panel_trajectories(ax, drift_rows, args.accent_task)
        print("Only panel (a): pass --feature and --pixel for the damage panel.",
              file=sys.stderr)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = f"{args.out}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
