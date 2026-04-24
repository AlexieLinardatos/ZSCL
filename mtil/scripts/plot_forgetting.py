"""
Forgetting profile figure for the paper.

Shows accuracy of two high-forgetting tasks (Aircraft, CIFAR100) over every
subsequent training stage for three configurations: ZSCL only, +proportional
replay, and Ours (full method with RD).

Usage (from mtil/):
    python scripts/plot_forgetting.py

Looks for these task_summary.csv files under ckpt/11task/:
    ablation_zscl_only/          ZSCL only (no replay, no RD)
    ablation_prop_replay/        ZSCL + proportional replay (no RD)
    phase3_no_lora_v4/           Ours full method

Outputs: forgetting_profile.pdf  (and .png for quick preview)
"""
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

TASK_ORDER = [
    'Aircraft', 'Caltech101', 'CIFAR100', 'DTD', 'EuroSAT',
    'Flowers', 'Food', 'MNIST', 'OxfordPet', 'StanfordCars', 'SUN397'
]

ABBREV = {
    'Aircraft': 'AC', 'Caltech101': 'Cal', 'CIFAR100': 'C100',
    'DTD': 'DTD', 'EuroSAT': 'ES', 'Flowers': 'Fl',
    'Food': 'Fo', 'MNIST': 'MN', 'OxfordPet': 'Pet',
    'StanfordCars': 'Cars', 'SUN397': 'SUN'
}

CONFIGS = [
    {
        'label': 'ZSCL only',
        'path': 'ckpt/11task/ablation_zscl_only/task_summary.csv',
        'color': '#D62728',
        'ls': '--',
        'marker': 'o',
        'zorder': 3,
    },
    {
        'label': '+Replay (prop.)',
        'path': 'ckpt/11task/ablation_prop_replay/task_summary.csv',
        'color': '#2CA02C',
        'ls': ':',
        'marker': 's',
        'zorder': 4,
    },
    {
        'label': 'Ours (full)',
        'path': 'ckpt/11task/phase3_no_lora_v4/task_summary.csv',
        'color': '#1F77B4',
        'ls': '-',
        'marker': '^',
        'zorder': 5,
    },
]

# The two tasks to show: (column_name, panel_title, y_label)
FOCUS_TASKS = [
    ('Aircraft',  'Aircraft (task 1)',  'Accuracy (%)'),
    ('CIFAR100',  'CIFAR-100 (task 3)', None),
]


def load_csv(path):
    if not os.path.exists(path):
        print(f"WARNING: missing {path}", file=sys.stderr)
        return None
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def get_trajectory(rows, target_task):
    """
    Return (x_labels, accuracies) for target_task starting from
    the stage it was trained and every subsequent stage.
    """
    start = TASK_ORDER.index(target_task)
    stages_after = TASK_ORDER[start:]
    xs, ys = [], []
    for row in rows:
        if row['task_name'] in stages_after:
            val = row.get(target_task, '')
            if val:
                xs.append(ABBREV[row['task_name']])
                ys.append(float(val))
    return xs, ys


def main():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), sharey=False)
    fig.subplots_adjust(wspace=0.32)

    for ax, (task_col, panel_title, y_label) in zip(axes, FOCUS_TASKS):
        task_start = TASK_ORDER.index(task_col)

        all_xs = None
        plotted = []

        for cfg in CONFIGS:
            rows = load_csv(cfg['path'])
            if rows is None:
                continue
            xs, ys = get_trajectory(rows, task_col)
            if not ys:
                continue
            if all_xs is None:
                all_xs = xs
            x_idx = np.arange(len(xs))
            ax.plot(x_idx, ys,
                    color=cfg['color'], linestyle=cfg['ls'],
                    marker=cfg['marker'], markersize=5,
                    linewidth=1.6, zorder=cfg['zorder'],
                    label=cfg['label'])
            plotted.append((cfg['color'], ys))

        if all_xs is None:
            print(f"No data for {task_col}. Skipping panel.", file=sys.stderr)
            continue

        x_idx = np.arange(len(all_xs))

        # Shade gap between ZSCL and Ours
        if len(plotted) >= 2:
            zscl_ys = plotted[0][1]
            ours_ys = plotted[-1][1]
            n = min(len(zscl_ys), len(ours_ys))
            ax.fill_between(x_idx[:n], zscl_ys[:n], ours_ys[:n],
                            alpha=0.10, color='#1F77B4', zorder=2)

        # Mark "trained here" (first x point)
        ax.axvline(x=0, color='#888888', linestyle=':', linewidth=1.0, zorder=1)
        ax.annotate('trained\nhere', xy=(0, ax.get_ylim()[0]),
                    xytext=(0.18, 0.06), textcoords='axes fraction',
                    fontsize=7.5, color='#666666',
                    arrowprops=None)

        ax.set_xticks(x_idx)
        ax.set_xticklabels(all_xs, rotation=40, ha='right')
        ax.set_title(panel_title, fontsize=10, pad=4)
        if y_label:
            ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.22, linestyle=':', axis='y')

    # Single shared legend below both panels
    handles = [
        mlines.Line2D([], [], color=c['color'], linestyle=c['ls'],
                      marker=c['marker'], markersize=5, linewidth=1.6,
                      label=c['label'])
        for c in CONFIGS
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=9, framealpha=0.92,
               bbox_to_anchor=(0.5, -0.06))

    for fmt in ('pdf', 'png'):
        out = f'forgetting_profile.{fmt}'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved {out}")


if __name__ == '__main__':
    main()
