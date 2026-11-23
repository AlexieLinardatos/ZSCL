"""
Buffer lifecycle figure (thesis Methods, Exemplar Memory Management).

Fixed budget K, allocated proportionally to class count after every task
(rebalance_proportional, the default in phase3/finetune_phase3.py): task i keeps
round(K * C_i / sum_{j<=t} C_j) slots. Reduce keeps a prefix of each old slice
(ranked buffer), so the trimmed tail is drawn hatched; Add fills the freed slots.

Run from repo root:  python mtil/scripts/plot_buffer_lifecycle.py
Saves figs/buffer_lifecycle.{pdf,png}.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = "figs/buffer_lifecycle"
K = 11000
TASKS = [  # MTIL order I, (name, number of classes)
    ("Aircraft", 100), ("Caltech101", 101), ("CIFAR100", 100), ("DTD", 47),
    ("EuroSAT", 10), ("Flowers", 102), ("Food", 101), ("MNIST", 10),
    ("OxfordPet", 37), ("StanfordCars", 196), ("SUN397", 397),
]
COLORS = plt.get_cmap("tab10").colors + ("#8c6d31",)


def alloc(t):
    """Slots per task after t tasks (1-based), same rounding as ReplayBuffer.rebalance_proportional."""
    cs = [c for _, c in TASKS[:t]]
    return [max(1, round(K * c / sum(cs))) for c in cs]


H, GAP = 0.55, 0.55


def bar(ax, y, segs, title):
    """segs: list of (task_idx, width, hatched)."""
    x = 0
    for i, w, hatched in segs:
        ax.add_patch(Rectangle((x, y), w, H, facecolor=COLORS[i],
                               alpha=0.25 if hatched else 1.0,
                               hatch="////" if hatched else None,
                               edgecolor="white" if not hatched else COLORS[i],
                               linewidth=0.8))
        if not hatched and w > 1500:
            ax.text(x + w / 2, y + H / 2, f"{TASKS[i][0]}\n{w:,}", ha="center",
                    va="center", fontsize=6.5, color="white")
        x += w
    ax.add_patch(Rectangle((0, y), K, H, fill=False, edgecolor="black", linewidth=0.8))
    ax.text(-250, y + H / 2, title, ha="right", va="center", fontsize=8)


def arrow(ax, y_from, y_to, label):
    ax.annotate("", xy=(K / 2, y_to + H + 0.03), xytext=(K / 2, y_from - 0.03),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.3"))
    ax.text(K / 2 + 200, (y_from + y_to + H) / 2, label, va="center", fontsize=7.5,
            style="italic", color="0.25")


def main():
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    y = 0.0
    prev = alloc(1)
    bar(ax, y, [(0, prev[0], False)], "After task 1")
    for t in (2, 3):
        new = alloc(t)
        y_r = y - H - GAP
        segs = []
        for i, (old, keep) in enumerate(zip(prev, new[:-1])):
            segs += [(i, keep, False), (i, old - keep, True)]
        bar(ax, y_r, segs, "Reduce")
        arrow(ax, y, y_r, f"reduce: trim each task to its share of {sum(c for _, c in TASKS[:t])} classes")
        y_a = y_r - H - GAP
        bar(ax, y_a, [(i, w, False) for i, w in enumerate(new)], f"After task {t}")
        arrow(ax, y_r, y_a, f"add: {TASKS[t - 1][0]} fills the freed slots")
        y, prev = y_a, new
    y_end = y - H - 1.1
    ax.text(K / 2, (y + y_end + H) / 2, "⋮", ha="center", va="center", fontsize=14)
    final = alloc(len(TASKS))
    bar(ax, y_end, [(i, w, False) for i, w in enumerate(final)], f"After task {len(TASKS)}")

    # budget bracket
    yt = H + 0.25
    ax.annotate("", xy=(0, yt), xytext=(K, yt), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(K / 2, yt + 0.12, f"fixed budget K = {K:,} images  (≈{K // sum(c for _, c in TASKS)} per class after task 11)",
            ha="center", va="bottom", fontsize=8)

    handles = [Rectangle((0, 0), 1, 1, color=COLORS[i]) for i in range(len(TASKS))]
    handles.append(Rectangle((0, 0), 1, 1, facecolor="0.85", hatch="////", edgecolor="0.5"))
    labels = [f"{n} ({final[i]:,})" for i, (n, _) in enumerate(TASKS)] + ["trimmed"]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4,
              fontsize=6.5, frameon=False, handlelength=1.2, columnspacing=1.0)

    ax.set_xlim(-2600, K + 100)
    ax.set_ylim(y_end - 0.1, yt + 0.5)
    ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved {OUT}.pdf/.png; final allocation:", dict(zip([n for n, _ in TASKS], final)))


if __name__ == "__main__":
    main()
