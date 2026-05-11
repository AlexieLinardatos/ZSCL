"""
Two-panel Pareto figure for ExRD paper.

Left:  Full method comparison (all published methods) — Transfer vs Last.
Right: Ablation trajectory showing how adding each component moves the
       operating point in Transfer-Last space.

Run from repo root:
    python mtil/scripts/plot_pareto_two_panel.py
Saves figures/Figure_2_twopanel.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── colour / style constants ──────────────────────────────────────────────────
REPLAY_COLOR   = "#2166ac"   # blue  — replay-based methods
SYNTH_COLOR    = "#b2182b"   # red   — generative/synthetic replay
PARAM_COLOR    = "#4dac26"   # green — parameter-efficient
REG_COLOR      = "#969696"   # grey  — regularisation baselines
EXRD_COLOR     = "#d62728"   # red star — our method

# ── data ─────────────────────────────────────────────────────────────────────
# (name, Transfer, Last, colour, marker, size, label_offset_xy)
ALL_METHODS = [
    # regularisation baselines
    ("Seq. FT",       59.0, 74.0,  REG_COLOR,   "o",  60,  ( 4,  4)),
    ("WiSE-FT",       52.3, 77.7,  REG_COLOR,   "^",  60,  ( 4,  4)),
    ("EWC",           62.1, 76.3,  REG_COLOR,   "s",  60,  ( 4,  4)),
    ("LwF",           56.9, 74.6,  REG_COLOR,   "D",  60,  ( 4,  4)),
    ("iCaRL",         50.4, 80.1,  REPLAY_COLOR,"o",  60,  ( 4,  4)),
    # ZSCL family
    ("ZSCL",          68.1, 83.6,  PARAM_COLOR, "s",  60,  ( 4,  4)),
    ("MoE-Adapters",  68.9, 85.0,  PARAM_COLOR, "^",  60,  ( 4, -9)),
    # generative replay
    ("GIFT",          69.3, 86.0,  SYNTH_COLOR, "D",  80,  ( 4, -9)),
    ("LoRA-Loop",     69.8, 86.0,  SYNTH_COLOR, "s",  80,  ( 4,  4)),
    # ours
    ("ExRD (ours)",   68.37, 86.28, EXRD_COLOR, "*", 260,  (-60, 5)),
]

# Ablation trajectory (Transfer, Last) in order
TRAJECTORY = [
    ("ZSCL",               68.10, 83.60),
    ("+Replay",            68.00, 86.44),
    (r"+RD ($\lambda{=}0.5$)", 68.18, 85.32),
    (r"ExRD ($\lambda{=}0.3$)", 68.37, 86.28),
]

# Reference points for context on the right panel
TRAJ_CONTEXT = [
    ("GIFT",      69.3, 86.0),
    ("LoRA-Loop", 69.8, 86.0),
]

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.5, 4.2))

# ─── LEFT PANEL: all methods ─────────────────────────────────────────────────
for name, tr, last, col, mk, sz, (ox, oy) in ALL_METHODS:
    ax_left.scatter(tr, last, color=col, s=sz, marker=mk, zorder=4,
                    edgecolors="white" if mk == "*" else col, linewidths=0.4)
    ax_left.annotate(name, (tr, last),
                     xytext=(ox, oy), textcoords="offset points",
                     fontsize=7.5, color=col,
                     fontweight="bold" if name.startswith("ExRD") else "normal")

# horizontal dashed line at ExRD Last
ax_left.axhline(y=86.28, color=EXRD_COLOR, linestyle="--", linewidth=0.9, alpha=0.55,
                label="ExRD Last frontier")

# legend patches
legend_items = [
    mpatches.Patch(color=REG_COLOR,   label="Regularisation"),
    mpatches.Patch(color=REPLAY_COLOR, label="Exemplar replay"),
    mpatches.Patch(color=PARAM_COLOR,  label="Param.-efficient"),
    mpatches.Patch(color=SYNTH_COLOR,  label="Generative replay"),
    mpatches.Patch(color=EXRD_COLOR,   label="ExRD (ours)"),
]
ax_left.legend(handles=legend_items, fontsize=7, loc="lower right", framealpha=0.85)

ax_left.set_xlabel("Transfer (ImageNet, %)", fontsize=10)
ax_left.set_ylabel("Last (mean final acc., %)", fontsize=10)
ax_left.set_title("Transfer vs.\ Last: all methods", fontsize=10)
ax_left.grid(True, alpha=0.25)
ax_left.set_xlim(48, 72)
ax_left.set_ylim(72, 88)

# ─── RIGHT PANEL: ablation trajectory ────────────────────────────────────────
# context baselines (grey)
for name, tr, last in TRAJ_CONTEXT:
    ax_right.scatter(tr, last, color="gray", s=55, marker="D", zorder=3,
                     edgecolors="gray", linewidths=0.4)
    ax_right.annotate(name, (tr, last), xytext=(4, 4), textcoords="offset points",
                      fontsize=7.5, color="gray")

# trajectory nodes + connecting arrows
traj_colors = ["#4dac26", REPLAY_COLOR, "#ff7f0e", EXRD_COLOR]
traj_xs = [t[1] for t in TRAJECTORY]
traj_ys = [t[2] for t in TRAJECTORY]

for i in range(len(TRAJECTORY) - 1):
    ax_right.annotate(
        "", xy=(traj_xs[i+1], traj_ys[i+1]), xytext=(traj_xs[i], traj_ys[i]),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4,
                        mutation_scale=12, connectionstyle="arc3,rad=0.08"))

label_offsets = [(-48, 5), (4, -11), (4, 5), (-58, 6)]
for i, (name, tr, last) in enumerate(TRAJECTORY):
    is_final = (i == len(TRAJECTORY) - 1)
    mk = "*" if is_final else "o"
    sz = 230 if is_final else 70
    col = traj_colors[i]
    ax_right.scatter(tr, last, color=col, s=sz, marker=mk, zorder=5,
                     edgecolors="white", linewidths=0.5)
    ox, oy = label_offsets[i]
    ax_right.annotate(name, (tr, last), xytext=(ox, oy), textcoords="offset points",
                      fontsize=7.5, color=col,
                      fontweight="bold" if is_final else "normal")

# dashed frontier line
ax_right.axhline(y=86.28, color=EXRD_COLOR, linestyle="--", linewidth=0.9, alpha=0.55)
ax_right.axhline(y=86.44, color=REPLAY_COLOR, linestyle=":", linewidth=0.9, alpha=0.45,
                 label="+Replay ceiling")

ax_right.set_xlabel("Transfer (ImageNet, %)", fontsize=10)
ax_right.set_ylabel("Last (mean final acc., %)", fontsize=10)
ax_right.set_title("Ablation: component trajectory", fontsize=10)
ax_right.grid(True, alpha=0.25)
ax_right.set_xlim(67.3, 70.5)
ax_right.set_ylim(83.0, 87.1)

import os
os.makedirs("figures", exist_ok=True)
plt.tight_layout(pad=1.5)
plt.savefig("figures/Figure_2_twopanel.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figures/Figure_2_twopanel.png", dpi=300, bbox_inches="tight")
print("Saved figures/Figure_2_twopanel.{pdf,png}")
