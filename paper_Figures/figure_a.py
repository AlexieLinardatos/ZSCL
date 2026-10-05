"""
Option A: Decomposition path scatter.
Zoomed in on the VLM cluster. Shows ZSCL → +Replay → Ours as arrows,
with GIFT and LoRA-Loop as faded context. Arrows labelled with deltas.
"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Faded context points
CONTEXT = [
    (68.9,  85.0, 'MoE-Adapters',        '#aaaaaa', '^', 55),
    (69.3,  86.0, r'GIFT$^\dagger$',      '#aaaaaa', 'D', 55),
    (69.8,  86.0, r'LoRA-Loop$^\dagger$', '#aaaaaa', 'D', 55),
]

# (Transfer, Last, label, color, marker, s, dx_pt, dy_pt, ha, va)
PATH = [
    (68.10, 83.60, 'ZSCL',       '#4472C4', '^', 100,  -10,  -8, 'right', 'top'),
    (68.00, 86.44, '+Replay',    '#2CA02C', 'o', 100,  -10,   6, 'right', 'bottom'),
    (68.37, 86.17, 'Ours (+RD)', '#C00000', '*', 240,    8,   6, 'left',  'bottom'),
]

# Arrow labels: (text, x offset from midpoint in pts, y offset)
ARROW_LABELS = [
    ('Last +2.84\nTransfer −0.10', -60, 10, '#2CA02C'),
    ('Last −0.27\nTransfer +0.37',  10, -22, '#C00000'),
]

fig, ax = plt.subplots(figsize=(5.5, 4.5))

# Faded context
for t, l, name, color, marker, s in CONTEXT:
    ax.scatter(t, l, c=color, marker=marker, s=s, zorder=3,
               edgecolors='white', linewidths=0.5, alpha=0.5, clip_on=False)
    ax.annotate(name, xy=(t, l), xytext=(7, 4), textcoords='offset points',
                fontsize=8, color=color, alpha=0.65, annotation_clip=False)

# Curved arrows between path steps
for i in range(len(PATH) - 1):
    x0, y0 = PATH[i][0], PATH[i][1]
    x1, y1 = PATH[i+1][0], PATH[i+1][1]
    col = PATH[i+1][3]
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=col, lw=2.0,
                                connectionstyle='arc3,rad=0.3'),
                annotation_clip=False)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    txt, odx, ody, tcol = ARROW_LABELS[i]
    ax.annotate(txt, xy=(mx, my), xytext=(odx, ody),
                textcoords='offset points', fontsize=8.5, color=tcol,
                annotation_clip=False, ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85, ec='none'))

# Path points + labels
for t, l, name, color, marker, s, dx, dy, ha, va in PATH:
    ax.scatter(t, l, c=color, marker=marker, s=s, zorder=6,
               edgecolors='white', linewidths=0.6, clip_on=False)
    ax.annotate(name, xy=(t, l), xytext=(dx, dy), textcoords='offset points',
                fontsize=10, color=color, fontweight='bold',
                ha=ha, va=va, annotation_clip=False)

ax.set_xlabel('Transfer — ImageNet zero-shot (%)')
ax.set_ylabel('Last — mean final accuracy (%)')
ax.set_xlim(67.0, 70.5)
ax.set_ylim(82.5, 87.5)
ax.grid(True, alpha=0.22, linestyle=':')

handles = [
    mlines.Line2D([],[],marker='^', color='w', markerfacecolor='#4472C4',
                  markersize=8, label='ZSCL (baseline)'),
    mlines.Line2D([],[],marker='o', color='w', markerfacecolor='#2CA02C',
                  markersize=8, label='+Replay'),
    mlines.Line2D([],[],marker='*', color='w', markerfacecolor='#C00000',
                  markersize=12, label='Ours (+RD)'),
    mlines.Line2D([],[],marker='D', color='w', markerfacecolor='#aaaaaa',
                  markersize=7, label='Generative baselines'),
]
ax.legend(handles=handles, fontsize=8, loc='lower right',
          bbox_to_anchor=(1.0, 0.0), framealpha=0.92,
          handlelength=1.2, handletextpad=0.4, labelspacing=0.35)

plt.tight_layout()
plt.savefig('figure_a.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure_a.png', dpi=300, bbox_inches='tight')
print("Saved figure_a.pdf / figure_a.png")
plt.show()
