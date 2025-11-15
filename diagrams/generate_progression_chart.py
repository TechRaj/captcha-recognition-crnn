"""
Generate model accuracy progression chart for poster
Outputs high-resolution PNG with rotated x-axis labels
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from model_logs.md
versions = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14']
labels = [
    'v1:\nBaseline',
    'v2:\nOver-reg',
    'v3:\nLR Fix',
    'v4:\nAttention',
    'v5:\nCustom Loss',
    'v6:\nStable',
    'v7:\nLabel Smooth',
    'v8:\nResNet',
    'v9:\nBigger',
    'v10:\nAug+',
    'v11:\nCLAHE',
    'v12:\nFilter',
    'v13:\nCurriculum',
    'v14:\nMulti-scale'
]
accuracies = [35, 22, 42, 34, 17, 42, 42, 50, 41, 55.6, 49, 54, 52, 54.65]

# Create figure
fig, ax = plt.subplots(figsize=(14, 6))

# Plot line with markers
x_pos = np.arange(len(versions))
line = ax.plot(x_pos, accuracies, marker='o', linewidth=2.5, markersize=8, 
               color='#2E7D32', markerfacecolor='#4CAF50', markeredgewidth=2, 
               markeredgecolor='#1B5E20', zorder=3)

# Highlight key milestones
milestones = {
    0: ('Baseline\n35%', '#FF5252'),  # v1
    2: ('LR Fix\n+7%', '#4CAF50'),     # v3
    7: ('ResNet\n+8%', '#4CAF50'),     # v8
    9: ('BEST\n55.6%', '#FFD700'),     # v10
}

for idx, (label, color) in milestones.items():
    ax.scatter(idx, accuracies[idx], s=300, c=color, edgecolors='black', 
               linewidth=2, zorder=5, alpha=0.7)
    ax.annotate(label, (idx, accuracies[idx]), 
                textcoords="offset points", xytext=(0, 15), 
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='black'))

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# Labels and title
ax.set_xlabel('Model Version', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Accuracy Progression: 14 Versions (35% → 55.6%)', 
             fontsize=16, fontweight='bold', pad=20)

# X-axis: rotated labels
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)

# Y-axis: set range and format
ax.set_ylim(10, 60)
ax.set_yticks(range(10, 65, 5))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

# Add horizontal line at baseline (35%)
ax.axhline(y=35, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Baseline (35%)')

# Add horizontal line at best (55.6%)
ax.axhline(y=55.6, color='gold', linestyle=':', linewidth=1.5, alpha=0.7, label='Best (55.6%)')

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

# Tight layout
plt.tight_layout()

# Save high-resolution PNG
output_path = 'model_progression_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Chart saved to: {output_path}")

# Also save as SVG (vector, scalable for poster)
output_svg = 'model_progression_chart.svg'
plt.savefig(output_svg, format='svg', bbox_inches='tight', facecolor='white')
print(f"✅ SVG version saved to: {output_svg}")

plt.show()

