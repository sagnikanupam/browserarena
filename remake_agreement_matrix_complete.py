#!/usr/bin/env python3
"""
Remake the agreement matrix plot with all evaluators using the data from the comprehensive report
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Based on the comprehensive evaluation report data:
# Agreement rates from the report:
# - Baseline ↔ GPT-4o: 68%
# - Baseline ↔ Human: 63%
# - Human ↔ Human: 100% (V1 and V2 identical)
# - GPT-4o ↔ O4-mini: 68%
# - Baseline ↔ O4-mini: 57.9%
# - Human ↔ GPT-4o: 47%
# - Human ↔ O4-mini: 58%

# Create agreement matrix
source_labels = ['Baseline', 'Other Humans', 'O4-mini', 'GPT-4o']
n_sources = len(source_labels)
agreement_matrix = np.zeros((n_sources, n_sources))

# Fill in the agreement matrix based on the report
# Diagonal is always 1.0
for i in range(n_sources):
    agreement_matrix[i, i] = 1.0

# Baseline row/column
agreement_matrix[0, 1] = agreement_matrix[1, 0] = 0.63  # Baseline-Human
agreement_matrix[0, 2] = agreement_matrix[2, 0] = 0.58  # Baseline-O4mini (57.9% rounded)
agreement_matrix[0, 3] = agreement_matrix[3, 0] = 0.68  # Baseline-GPT4o

# Human row/column
agreement_matrix[1, 2] = agreement_matrix[2, 1] = 0.58  # Human-O4mini
agreement_matrix[1, 3] = agreement_matrix[3, 1] = 0.47  # Human-GPT4o

# O4mini-GPT4o
agreement_matrix[2, 3] = agreement_matrix[3, 2] = 0.68  # O4mini-GPT4o

# Create heatmap
im = ax.imshow(agreement_matrix, cmap='RdYlBu', vmin=0, vmax=1, aspect='auto')

# Add text annotations
for i in range(n_sources):
    for j in range(n_sources):
        text = ax.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                      ha="center", va="center", 
                      color="black" if agreement_matrix[i, j] > 0.5 else "white",
                      fontsize=14, fontweight='bold')

ax.set_xticks(np.arange(n_sources))
ax.set_yticks(np.arange(n_sources))
ax.set_xticklabels(source_labels, fontsize=14)
ax.set_yticklabels(source_labels, fontsize=14)

# Rotate x labels for better readability
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

ax.set_title('Pairwise Agreement Matrix', fontsize=18, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Agreement Rate', fontsize=12)
cbar.ax.tick_params(labelsize=12)

# Add grid
ax.set_xticks(np.arange(n_sources)-.5, minor=True)
ax.set_yticks(np.arange(n_sources)-.5, minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
ax.tick_params(which="minor", size=0)

# Highlight perfect agreement (diagonal)
for i in range(n_sources):
    rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)

plt.tight_layout()
plt.savefig('/Users/davisbrown/browserarena/agreement-results/agreement_matrix.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/davisbrown/browserarena/agreement-results/agreement_matrix.png', dpi=300, bbox_inches='tight')
print("Updated agreement matrix with all evaluators saved!")

# Print the actual agreement values for reference
print("\nAgreement Matrix Values:")
print("                    ", end="")
for label in source_labels:
    print(f"{label:>12}", end="")
print()
for i, label1 in enumerate(source_labels):
    print(f"{label1:<15}", end="")
    for j in range(n_sources):
        print(f"{agreement_matrix[i, j]:>12.2f}", end="")
    print()