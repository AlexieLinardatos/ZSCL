"""
Option C: 2x2 substitution grid.
Rows = ZSCL on/off, Cols = RD on/off.
Shows Transfer and Last for each cell; pending cells grayed out.
Fills in automatically once no-ZSCL jobs finish on Nibi.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
})

# Grid values: (Transfer, Last) or None if pending
# Rows: ZSCL on (0), ZSCL off (1)
# Cols: No RD (0), With RD (1)
DATA = {
    (0, 0): (68.00, 86.44),   # ZSCL on,  RD off  (+prop replay)
    (0, 1): (68.37, 86.17),   # ZSCL on,  RD on   (Ours)
    (1, 0): None,             # ZSCL off, RD off  (pending)
    (1, 1): None,             # ZSCL off, RD on   (pending)
}

ROW_LABELS = ['ZSCL\non', 'ZSCL\noff']
COL_LABELS = ['No RD', '+RD']

# Color scale based on Transfer (known range ~60-70)
TRANSFER_MIN, TRANSFER_MAX = 60.0, 70.0

def transfer_color(val):
    t = (val - TRANSFER_MIN) / (TRANSFER_MAX - TRANSFER_MIN)
    t = max(0, min(1, t))
    # Blue gradient: light to dark
    return (1 - t * 0.7, 1 - t * 0.7, 1.0)

fig, ax = plt.subplots(figsize=(5.0, 3.8))
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.set_aspect('equal')
ax.axis('off')

CELL_W, CELL_H = 1.0, 1.0

for row in range(2):
    for col in range(2):
        x = col * CELL_W
        y = (1 - row) * CELL_H   # row 0 at top

        val = DATA[(row, col)]
        if val is not None:
            transfer, last = val
            color = transfer_color(transfer)
            rect = mpatches.FancyBboxPatch(
                (x + 0.04, y + 0.04), CELL_W - 0.08, CELL_H - 0.08,
                boxstyle='round,pad=0.02', linewidth=1.2,
                edgecolor='#cccccc', facecolor=color, zorder=2)
            ax.add_patch(rect)
            cx, cy = x + CELL_W/2, y + CELL_H/2
            ax.text(cx, cy + 0.15, f'Transfer\n{transfer:.2f}',
                    ha='center', va='center', fontsize=9.5, fontweight='bold',
                    color='#1a1a2e', zorder=3)
            ax.text(cx, cy - 0.18, f'Last  {last:.2f}',
                    ha='center', va='center', fontsize=8.5,
                    color='#444444', zorder=3)
            # Highlight Ours
            if row == 0 and col == 1:
                rect2 = mpatches.FancyBboxPatch(
                    (x + 0.02, y + 0.02), CELL_W - 0.04, CELL_H - 0.04,
                    boxstyle='round,pad=0.02', linewidth=2.2,
                    edgecolor='#C00000', facecolor='none', zorder=4)
                ax.add_patch(rect2)
                ax.text(cx, y + 0.06, 'Ours', ha='center', va='bottom',
                        fontsize=7.5, color='#C00000', fontweight='bold', zorder=5)
        else:
            rect = mpatches.FancyBboxPatch(
                (x + 0.04, y + 0.04), CELL_W - 0.08, CELL_H - 0.08,
                boxstyle='round,pad=0.02', linewidth=1.2,
                edgecolor='#cccccc', facecolor='#f0f0f0', zorder=2)
            ax.add_patch(rect)
            cx, cy = x + CELL_W/2, y + CELL_H/2
            ax.text(cx, cy, 'Pending', ha='center', va='center',
                    fontsize=9, color='#aaaaaa', style='italic', zorder=3)

# Column headers
for col, label in enumerate(COL_LABELS):
    ax.text(col * CELL_W + CELL_W/2, 2.35, label,
            ha='center', va='center', fontsize=10, fontweight='bold', color='#333333')

# Row headers
for row, label in enumerate(ROW_LABELS):
    ax.text(-0.25, (1 - row) * CELL_H + CELL_H/2, label,
            ha='center', va='center', fontsize=10, fontweight='bold', color='#333333')

# Colorbar legend
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.Blues,
    norm=plt.Normalize(vmin=TRANSFER_MIN, vmax=TRANSFER_MAX))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.02,
                    shrink=0.6, anchor=(0.0, 0.5))
cbar.set_label('Transfer (%)', fontsize=8.5)
cbar.ax.tick_params(labelsize=8)

ax.set_title('Substitution claim: can RD replace ZSCL?',
             fontsize=10, pad=12, color='#333333')

plt.tight_layout()
plt.savefig('figure_c.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure_c.png', dpi=300, bbox_inches='tight')
print("Saved figure_c.pdf / figure_c.png")
plt.show()
