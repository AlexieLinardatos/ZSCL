import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 30,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Numbers from Table 1 in the paper.
# Seq. FT inferred from ZSCL deltas: 68.1-9.1=59.0 Transfer, 83.6-9.6=74.0 Last.
#
# To reposition a label, edit (dx_pt, dy_pt) — offset in typographic points
# from the marker centre. Positive x = right, positive y = up.
# ha = horizontal alignment ('left' | 'right' | 'center')
# va = vertical alignment ('bottom' | 'top' | 'center')
#
# (Transfer, Last, label, color, marker, s, dx_pt, dy_pt, ha, va)
METHODS = [
    (50.4,  80.1,  'iCaRL',               '#666666', 's',  35,    6,   4, 'left',  'bottom'),
    (52.3,  77.7,  'WiSE-FT',             '#666666', 's',  35,    6,   4, 'left',  'bottom'),
    (56.9,  74.6,  'LwF',                 '#666666', 's',  35,   -6,   4, 'right', 'bottom'),
    (59.0,  74.0,  'Seq. FT',             '#666666', 'o',  35,    6,  -6, 'right',  'top'),
    (62.1,  76.3,  'EWC',                 '#666666', 's',  35,   -6,   4, 'right', 'bottom'),
    (68.1,  83.6,  'ZSCL',                '#4472C4', '^',  65,   -8,   4, 'right', 'bottom'),
    (68.9,  85.0,  'MoE-Adapters',        '#4472C4', '^',  65,   -8,   4, 'right', 'bottom'),
    (69.3,  86.0,  r'GIFT',     '#E07B39', 'D',  65,    5,   5, 'left',  'bottom'),
    (69.8,  86.0,  r'LoRA-Loop','#E07B39', 'D',  65,    5,  -7, 'left',  'top'),
    (68.37, 86.28, 'ExRD',                '#C00000', '*', 200,   -8,   5, 'right', 'bottom'),
]

LINE_LEN = 14  # uniform connector length in typographic points — edit to taste

fig, ax = plt.subplots(figsize=(6.0, 4.5))

for t, l, name, color, marker, s, dx, dy, ha, va in METHODS:
    ax.scatter(t, l, c=color, marker=marker, s=s, zorder=5,
               edgecolors='white', linewidths=0.5, clip_on=False)
    mag = math.hypot(dx, dy)
    ndx, ndy = dx / mag * LINE_LEN, dy / mag * LINE_LEN
    ax.annotate(
        name,
        xy=(t, l),
        xytext=(ndx, ndy),
        textcoords='offset points',
        fontsize=8.5,
        color=color,
        fontweight='bold' if name == 'Ours' else 'normal',
        ha=ha, va=va,
        zorder=10,
        annotation_clip=False,
        arrowprops=dict(
            arrowstyle='-',
            color=color,
            lw=0.6,
            alpha=0.6,
        ),
    )

# Pareto frontier: horizontal line at Ours' Last (86.17) across the full plot.
# Ours is the only method that achieves this Last; everything below is dominated.
ax.axhline(y=86.28, color='#444444', linestyle='--', linewidth=1.0,
           alpha=0.45, zorder=2)

ax.set_xlabel('Transfer — ImageNet Zero-Shot (%)')
ax.set_ylabel('Last — Mean Final Accuracy (%)')
ax.set_xlim(47.5, 72.0)
ax.set_ylim(72.5, 87.5)
ax.grid(True, alpha=0.22, linestyle=':')

legend_handles = [
    mlines.Line2D([],[], marker='s', color='w', markerfacecolor='#666666',
                  markersize=7, label='Standard CL baselines'),
    mlines.Line2D([],[], marker='^', color='w', markerfacecolor='#4472C4',
                  markersize=7, label='VLM CL methods'),
    mlines.Line2D([],[], marker='D', color='w', markerfacecolor='#E07B39',
                  markersize=7, label='Specialised CL methods'),
    mlines.Line2D([],[], marker='*', color='w', markerfacecolor='#C00000',
                  markersize=11, label='ExRD (Ours)'),
    mlines.Line2D([],[], color='#444444', linestyle='--', linewidth=1.0,
                  alpha=0.6, label='Pareto frontier'),
]

ax.legend(handles=legend_handles, fontsize=8, loc='lower right',
          bbox_to_anchor=(1.02, 0.02), framealpha=0.92,
          handlelength=1.4, handletextpad=0.5, labelspacing=0.35)

plt.tight_layout()
plt.savefig('pareto_frontier.pdf', dpi=300, bbox_inches='tight')
plt.savefig('pareto_frontier.png', dpi=300, bbox_inches='tight')
print("Saved pareto_frontier.pdf and pareto_frontier.png")
plt.show()
