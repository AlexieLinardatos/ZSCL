"""Schematic of the MTIL problem: zero-shot CLIP -> fine-tune task 1 -> ... -> task T.
Transfer (accuracy on not-yet-seen tasks) drops; each learned task fades. No data."""
import matplotlib.pyplot as plt

OUT = "figs/problem_cartoon"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})

GAP, LAST = 4, 5                                   # x of the "..." gap and of Task T
labels = ["Zero-shot\n  CLIP", "Task 1", "Task 2", "Task 3", r"$\cdots$", r"Task $T$"]
transfer = [0.60, 0.49, 0.43, 0.39, 0.355, 0.34]
ZS = transfer[0]
PEAK = 0.92
decay = [PEAK, 0.75, 0.65]                         # 0, 1, 2 steps after learning
end = {1: 0.45, 2: 0.515, 3: 0.58}                  # value at Task T after many tasks
blues = {1: "#08306b", 2: "#1f5fa8", 3: "#3d85c6", LAST: "#5b9bd5"}
marks = {1: "o", 2: "^", 3: "s", LAST: "D"}
ORANGE = "#eb6834"
GAPSTYLE = dict(ls=(0, (1.5, 1.5)), lw=1.0, alpha=0.6)
RISE = dict(ls=":", lw=0.9)

fig, ax = plt.subplots(figsize=(3.5, 2.0))

# zero-shot reference and lost-transfer gap
ax.axhline(ZS, color="#8a8986", lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.fill_between(range(LAST + 1), transfer, ZS, color=ORANGE, alpha=0.10, lw=0, zorder=0)

# tasks 1-3: rise from unseen (on Transfer curve) to peak, fade, continue through gap
for k in (1, 2, 3):
    c, m = blues[k], marks[k]
    ax.plot([k - 1, k], [transfer[k - 1], PEAK], color=c, **RISE, zorder=2)
    xs = list(range(k, 4))
    ys = decay[:len(xs)]
    ax.plot(xs, ys, color=c, lw=1.4, marker=m, ms=3, zorder=3)
    ax.plot([3, LAST], [ys[-1], end[k]], color=c, **GAPSTYLE, zorder=3)
    ax.plot(LAST, end[k], color=c, marker=m, ms=3, zorder=3)
    ax.text(k, PEAK + 0.035, f"Task {k}", color=c, fontsize=6, ha="center")

# Task T: last thing trained, peak with no decline after it
c, m = blues[LAST], marks[LAST]
ax.plot([GAP, LAST], [transfer[GAP], PEAK], color=c, **RISE, zorder=2)
ax.plot(LAST, PEAK, color=c, marker=m, ms=3, zorder=3)
ax.text(LAST, PEAK + 0.035, r"Task $T$", color=c, fontsize=6, ha="center")

# Transfer curve, faded through the gap
ax.plot(range(4), transfer[:4], color=ORANGE, lw=1.6, marker="s", ms=3, zorder=4)
ax.plot([3, LAST], [transfer[3], transfer[LAST]], color=ORANGE, **GAPSTYLE, zorder=4)
ax.plot(LAST, transfer[LAST], color=ORANGE, marker="s", ms=3, zorder=4)
ax.text(2.0, 0.335, "Transfer (unseen tasks)", color=ORANGE, fontsize=6.5, ha="center")

# lost zero-shot, measured at Task T
X_ARR = LAST + 0.28
ax.annotate("", xy=(X_ARR, transfer[LAST]), xytext=(X_ARR, ZS),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9))
ax.text(X_ARR + 0.08, (ZS + transfer[LAST]) / 2, "lost\nzero-shot", color=ORANGE,
        fontsize=6, ha="left", va="center")

# forgetting arrow on task 1
ax.annotate("forgetting", xy=(1.55, 0.83), xytext=(1.5, 1.08), fontsize=6.5,
            color=blues[1], ha="center",
            arrowprops=dict(arrowstyle="->", color=blues[1], lw=0.8))

ax.set_xticks(range(LAST + 1), labels, fontsize=6)
ax.set_yticks([ZS], ["zero-shot"], fontsize=6, color="#52514e")
ax.tick_params(axis="y", length=0)
ax.set_ylim(0.25, 1.13)
ax.set_xlim(-0.3, 6.15)
ax.set_ylabel("Accuracy", fontsize=7)
ax.set_xlabel(r"Fine-tune sequentially $\rightarrow$", fontsize=7)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
