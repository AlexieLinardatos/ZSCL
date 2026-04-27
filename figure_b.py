"""
Option B: Component delta bar chart.
Shows how much each component (Replay, RD) contributes to Last vs Transfer,
measured as delta from ZSCL baseline. Key insight: orthogonal mechanisms.
  Replay: Last +2.84, Transfer -0.10  (drives retention)
  RD:     Last -0.27, Transfer +0.37  (drives zero-shot)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

components  = ['Replay', 'RD']
last_deltas = [+2.84, -0.27]      # vs ZSCL; RD delta = Ours - (+prop replay)
xfer_deltas = [-0.10, +0.37]

BLUE   = '#2166AC'
ORANGE = '#D6604D'

x = np.arange(len(components))
w = 0.32

fig, ax = plt.subplots(figsize=(5.5, 4.2))

b1 = ax.bar(x - w/2, last_deltas, w, label='Last Δ',
            color=BLUE, alpha=0.88, edgecolor='white', linewidth=0.5, zorder=3)
b2 = ax.bar(x + w/2, xfer_deltas, w, label='Transfer Δ',
            color=ORANGE, alpha=0.88, edgecolor='white', linewidth=0.5, zorder=3)

# Value labels
for bar in list(b1) + list(b2):
    h = bar.get_height()
    va  = 'bottom' if h >= 0 else 'top'
    ypos = h + (0.06 if h >= 0 else -0.06)
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f'{h:+.2f}', ha='center', va=va, fontsize=9.5, fontweight='bold',
            color=bar.get_facecolor())

ax.axhline(0, color='#333333', linewidth=0.9)
ax.set_ylabel('Accuracy change from ZSCL (%)')
ax.set_xticks(x)
ax.set_xticklabels(components)
ax.set_ylim(-0.8, 3.6)
ax.grid(True, alpha=0.22, linestyle=':', axis='y', zorder=0)

ax.legend(fontsize=9, framealpha=0.92, loc='upper right',
          handlelength=1.2, handletextpad=0.4)

# Annotations explaining the story
ax.annotate('drives\nretention', xy=(0 - w/2, 2.84), xytext=(0 - w/2 - 0.28, 3.25),
            fontsize=7.5, color=BLUE, ha='center',
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8))
ax.annotate('drives\nzero-shot', xy=(1 + w/2, 0.37), xytext=(1 + w/2 + 0.28, 0.85),
            fontsize=7.5, color=ORANGE, ha='center',
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.8))

plt.tight_layout()
plt.savefig('figure_b.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure_b.png', dpi=300, bbox_inches='tight')
print("Saved figure_b.pdf / figure_b.png")
plt.show()
