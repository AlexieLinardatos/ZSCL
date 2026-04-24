import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from adjustText import adjust_text

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

methods = [
    (59.0,  74.0,  'Seq. FT',       '#aaaaaa', 'o',  55),
    (67.0,  78.0,  'WiSE-FT',       '#aaaaaa', 's',  55),
    (62.1,  76.3,  'EWC',           '#aaaaaa', 's',  55),
    (66.8,  78.1,  'LwF',           '#aaaaaa', 's',  55),
    (68.1,  83.6,  'ZSCL',          '#4472C4', '^',  80),
    (68.9,  85.0,  'MoE-Adapters',  '#4472C4', '^',  80),
    (69.3,  86.0,  'GIFT$^\\dagger$',      '#E07B39', 'D', 80),
    (69.8,  86.0,  'LoRA-Loop$^\\dagger$', '#E07B39', 'D', 80),
    (68.37, 86.17, 'Ours',          '#C00000', '*', 220),
]

fig, ax = plt.subplots(figsize=(5.5, 4.5))

texts = []

# scatter + text
for t, l, name, color, marker, size in methods:
    ax.scatter(
        t, l,
        c=color,
        marker=marker,
        s=size,
        zorder=5,
        edgecolors='white',
        linewidths=0.5
    )

    txt = ax.text(
        t,
        l,
        name,
        fontsize=8.5,
        color=color,
        fontweight='bold' if name == 'Ours' else 'normal',
        zorder=10
    )

    texts.append(txt)

# =========================
# KEY FIX: strong repulsion
# =========================
adjust_text(
    texts,
    ax=ax,

    # push labels away HARD
    expand_text=(2.5, 2.5),
    expand_points=(2.5, 2.5),

    # stronger repulsion = less overlap
    force_text=(1.5, 2.0),
    force_points=(0.5, 1.0),

    # allow full movement (critical)
    only_move={'text': 'xy'},

    # keep labels OUTSIDE cluster
    lim=200,

    arrowprops=dict(
        arrowstyle='-',
        color='gray',
        lw=0.5,
        alpha=0.6
    )
)

# Pareto frontier
frontier_t = [68.37, 69.3, 69.3, 69.8]
frontier_l = [86.17, 86.17, 86.0, 86.0]

ax.plot(frontier_t, frontier_l, 'k--', linewidth=1.1, alpha=0.45)

# shaded region
shade_t = [56.0, 68.37, 69.3, 69.8, 72.0]
shade_l = [86.17, 86.17, 86.0, 86.0, 86.0]

ax.fill_between(shade_t, 72.0, shade_l, alpha=0.04, color='black')

ax.set_xlabel('Transfer — ImageNet zero-shot (%)')
ax.set_ylabel('Last — mean final accuracy (%)')
ax.set_xlim(56.5, 72.0)
ax.set_ylim(72.5, 87.5)
ax.grid(True, alpha=0.22, linestyle=':')

legend_handles = [
    mlines.Line2D([],[],marker='s',color='w',markerfacecolor='#aaaaaa',
                  markersize=8, label='Baselines'),
    mlines.Line2D([],[],marker='^',color='w',markerfacecolor='#4472C4',
                  markersize=8, label='VLM CL methods'),
    mlines.Line2D([],[],marker='D',color='w',markerfacecolor='#E07B39',
                  markersize=8, label='Generative replay ($\\dagger$)'),
    mlines.Line2D([],[],marker='*',color='w',markerfacecolor='#C00000',
                  markersize=12, label='Ours'),
]

ax.legend(handles=legend_handles, fontsize=8, loc='lower right', framealpha=0.92)

plt.tight_layout()
plt.savefig('pareto_frontier.pdf', dpi=300, bbox_inches='tight')
plt.show()