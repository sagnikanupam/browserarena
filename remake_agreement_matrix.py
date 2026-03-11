#!/usr/bin/env python3
"""
Remake the agreement matrix plot with updated labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']

# Load the comprehensive comparison data
df = pd.read_csv('/Users/davisbrown/browserarena/agreement-results/final_comprehensive_comparison.csv')

# Also load o4mini data if available
try:
    o4mini_df = pd.read_csv('/Users/davisbrown/browserarena/agreement-results/o4mini_evaluation_final.csv')
    # Merge o4mini preferences
    o4mini_map = {row['Question']: row['Preference'] for _, row in o4mini_df.iterrows()}
    df['O4mini'] = df['Question'].map(o4mini_map)
except:
    print("Could not load O4-mini data")
    df['O4mini'] = None

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Prepare sources and labels
sources = ['Baseline', 'Human_V2', 'GPT4o', 'O4mini']
source_labels = ['Baseline', 'Other Humans', 'GPT-4o', 'O4-mini']

# Calculate pairwise agreement
n_sources = len(sources)
agreement_matrix = np.zeros((n_sources, n_sources))

for i, src1 in enumerate(sources):
    for j, src2 in enumerate(sources):
        if i == j:
            agreement_matrix[i, j] = 1.0
        elif src1 in df.columns and src2 in df.columns:
            # Both columns exist
            valid_mask = df[src1].notna() & df[src2].notna()
            if valid_mask.sum() > 0:
                agreement = (df[src1] == df[src2]).sum() / valid_mask.sum()
                agreement_matrix[i, j] = agreement
        else:
            agreement_matrix[i, j] = np.nan

# If O4-mini data is missing, reduce the matrix
if df['O4mini'].isna().all():
    sources = sources[:3]
    source_labels = source_labels[:3]
    agreement_matrix = agreement_matrix[:3, :3]
    n_sources = 3

# Create heatmap
im = ax.imshow(agreement_matrix, cmap='RdYlBu', vmin=0, vmax=1, aspect='auto')

# Add text annotations
for i in range(n_sources):
    for j in range(n_sources):
        if not np.isnan(agreement_matrix[i, j]):
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
print("Updated agreement matrix saved!")

# Print the actual agreement values for reference
print("\nAgreement Matrix Values:")
print("                    ", end="")
for label in source_labels:
    print(f"{label:>12}", end="")
print()
for i, label1 in enumerate(source_labels):
    print(f"{label1:<15}", end="")
    for j in range(n_sources):
        if not np.isnan(agreement_matrix[i, j]):
            print(f"{agreement_matrix[i, j]:>12.2f}", end="")
        else:
            print(f"{'N/A':>12}", end="")
    print()