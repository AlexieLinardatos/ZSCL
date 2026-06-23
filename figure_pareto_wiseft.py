"""Pareto frontier: WiSE-FT sweep on the LLM-anchor (ZSCL-text dropped) model
vs. prior work (GIFT, LoRA-Loop) and ExRD.

x = Transfer (ImageNet held-out retention), y = Last (mean over 11 trained tasks).
Up-and-to-the-right is better; a curve passing above/right of GIFT & LoRA-Loop
means our method Pareto-dominates them.

Fill SWEEP with the 7 alpha points from the job summary:
  Transfer = ImageNet accuracy at that alpha
  Last     = mean of the 11 task accuracies at that alpha
Then:  python figure_pareto_wiseft.py  ->  pareto_wiseft.{pdf,png}
"""
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ---- WiSE-FT sweeps: (alpha, Transfer, Last) per anchor variant ----
# Fill Transfer (=ImageNet acc) and Last (=mean of 11 task accs) from the
# eval.log under each alpha dir. Leave None to skip a point.
SWEEPS = [
    # label, color, points  (dir: ckpt/11task/wiseft_llm_anchor/alpha_*)
    ('LLM-anchor, ZSCL-text dropped', '#4472C4', [
        (0.50, None, None),
        (0.60, None, None),
        (0.70, None, None),
        (0.80, None, None),
        (0.90, None, None),
        (0.95, None, None),
        (1.00, None, None),
    ]),
    # dir: ckpt/11task/wiseft_llm_anchor_added/alpha_*
    ('LLM-anchor, added', '#5BA053', [
        (0.50, None, None),
        (0.60, None, None),
        (0.70, None, None),
        (0.80, None, None),
        (0.90, None, None),
        (0.95, None, None),
        (1.00, None, None),
    ]),
]

# ---- reference points: (Transfer, Last, label, color, marker) ----
REFS = [
    (68.37, 86.17, 'ExRD',      '#C00000', '*'),
    (69.3,  86.0,  'GIFT',      '#E07B39', 'D'),
    (69.8,  86.0,  'LoRA-Loop', '#E07B39', 'D'),
]

fig, ax = plt.subplots(figsize=(6.0, 4.5))

# sweep curves (only plot points that are filled in)
for label, color, sweep in SWEEPS:
    pts = [(t, l, a) for a, t, l in sweep if t is not None and l is not None]
    if not pts:
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, '-o', color=color, markersize=5, linewidth=1.3,
            zorder=4, label=f'{label} + WiSE-FT')
    for t, l, a in pts:
        ax.annotate(f'$\\alpha$={a:.2f}', xy=(t, l), xytext=(4, 4),
                    textcoords='offset points', fontsize=7, color=color)

for t, l, name, color, marker in REFS:
    ax.scatter(t, l, c=color, marker=marker, s=120 if marker == '*' else 60,
               zorder=5, edgecolors='white', linewidths=0.5)
    ax.annotate(name, xy=(t, l), xytext=(5, -9), textcoords='offset points',
                fontsize=8, color=color)

ax.set_xlabel('Transfer (ImageNet zero-shot, %)')
ax.set_ylabel('Last (mean over 11 tasks, %)')
ax.set_title('Last vs. Transfer Pareto frontier', fontsize=11)
ax.grid(True, color='#dddddd', linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=8.5, loc='lower left', framealpha=0.92)

plt.tight_layout()
plt.savefig('pareto_wiseft.pdf', dpi=300, bbox_inches='tight')
plt.savefig('pareto_wiseft.png', dpi=300, bbox_inches='tight')
print('wrote pareto_wiseft.pdf / pareto_wiseft.png')
