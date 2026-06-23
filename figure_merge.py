"""Grouped bar chart: multi-teacher merge vs Phase 3 v4 baseline.

Metrics are over the 11 MTIL tasks with ImageNet EXCLUDED:
  Transfer = mean zero-shot acc on not-yet-trained tasks (strict upper triangle)
  Avg      = mean of the whole 11x11 accuracy matrix
  Last     = mean of the final row (after all tasks trained)

Run:  python figure_merge.py   ->  writes merge_metrics.{pdf,png}
"""
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

METRICS = ['Transfer', 'Avg', 'Last']

# (label, color, [Transfer, Avg, Last])
RUNS = [
    ('Phase 3 v4 (base)', '#666666', [69.18, 77.17, 86.17]),
    ('Merge v1',          '#E07B39', [68.78, 77.14, 86.31]),
    ('Merge v3',          '#C00000', [68.86, 77.18, 86.33]),
]

n_groups = len(METRICS)
n_runs = len(RUNS)
bar_w = 0.26
x = list(range(n_groups))

fig, ax = plt.subplots(figsize=(6.0, 4.5))

for i, (label, color, vals) in enumerate(RUNS):
    offs = (i - (n_runs - 1) / 2) * bar_w
    positions = [xi + offs for xi in x]
    bars = ax.bar(positions, vals, width=bar_w, color=color, label=label,
                  edgecolor='white', linewidth=0.6, zorder=3)
    for rect, v in zip(bars, vals):
        ax.annotate(f'{v:.2f}', xy=(rect.get_x() + rect.get_width() / 2, v),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7.5, color=color)

# y-axis zoomed so the sub-point differences are visible.
ax.set_ylim(64, 90)
ax.set_xticks(x)
ax.set_xticklabels(METRICS)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Multi-teacher merge vs Phase 3 baseline\n(11 tasks, ImageNet excluded)',
             fontsize=11)
ax.yaxis.grid(True, color='#dddddd', linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=8.5, loc='upper left', framealpha=0.92)

plt.tight_layout()
plt.savefig('merge_metrics.pdf', dpi=300, bbox_inches='tight')
plt.savefig('merge_metrics.png', dpi=300, bbox_inches='tight')
print('wrote merge_metrics.pdf / merge_metrics.png')
