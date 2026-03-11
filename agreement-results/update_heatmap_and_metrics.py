#!/usr/bin/env python3
"""Update the three-way comparison heatmap with new labels and font sizes, and calculate all reliability metrics"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
import krippendorff

# Set larger default font sizes
plt.rcParams.update({'font.size': 14})

# Read the comparison data
comparison_df = pd.read_csv('three_way_comparison.csv')

# Create the updated heatmap
fig, ax = plt.subplots(figsize=(16, 12))

# Prepare data for heatmap
questions = comparison_df['Question'].tolist()
evaluators = ['Original Vote', 'Human V2 Majority', 'GPT-4.1']
heatmap_data = np.zeros((len(questions), len(evaluators)))

for i, q in enumerate(questions):
    row = comparison_df[comparison_df['Question'] == q].iloc[0]
    for j, (eval_name, col_name) in enumerate([('Original Vote', 'Original_Baseline'),
                                                ('Human V2 Majority', 'Human_V2_Majority'),
                                                ('GPT-4.1', 'GPT4_Eval')]):
        if row[col_name] == 'Agent 1':
            heatmap_data[i, j] = 1
        elif row[col_name] == 'Agent 2':
            heatmap_data[i, j] = 2
        elif row[col_name] == 'Tie':
            heatmap_data[i, j] = 3

# Create custom colormap
colors = ['white', '#3498db', '#e74c3c', '#95a5a6']
n_bins = 4
cmap = sns.color_palette(colors, n_colors=n_bins, as_cmap=True)

# Plot heatmap with larger fonts
sns.heatmap(heatmap_data, 
            xticklabels=evaluators,
            yticklabels=questions,
            cmap=cmap,
            cbar_kws={'label': 'Choice', 'ticks': [0.5, 1, 2, 2.5]},
            vmin=0, vmax=3,
            linewidths=0.5,
            linecolor='gray',
            annot_kws={'size': 16})

# Customize colorbar with larger font
cbar = ax.collections[0].colorbar
cbar.set_ticklabels(['', 'Agent 1', 'Agent 2', 'Tie'])
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Choice', size=18)

# Set title and labels with larger fonts
plt.title('Three-Way Evaluation Comparison', fontsize=20, pad=20)
plt.xlabel('Evaluator', fontsize=18)
plt.ylabel('Question', fontsize=18)

# Increase tick label sizes
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=14)

plt.tight_layout()
plt.savefig('three_way_comparison_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("Updated heatmap saved with new label and larger fonts!")

# Now calculate all reliability metrics
print("\n=== CALCULATING ALL RELIABILITY METRICS ===")

# Convert labels to numeric for calculations
def label_to_numeric(label):
    if label == 'Agent 1':
        return 1
    elif label == 'Agent 2':
        return 2
    elif label == 'Tie':
        return 3
    return None

# 1. Cohen's Kappa between Original Vote and Human V2
original_numeric = [label_to_numeric(row['Original_Baseline']) for _, row in comparison_df.iterrows()]
humanv2_numeric = [label_to_numeric(row['Human_V2_Majority']) for _, row in comparison_df.iterrows()]
gpt4_numeric = [label_to_numeric(row['GPT4_Eval']) for _, row in comparison_df.iterrows()]

kappa_original_v2 = cohen_kappa_score(original_numeric, humanv2_numeric)
kappa_original_gpt4 = cohen_kappa_score(original_numeric, gpt4_numeric)
kappa_v2_gpt4 = cohen_kappa_score(humanv2_numeric, gpt4_numeric)

print(f"\nCohen's Kappa Scores:")
print(f"- Original Vote vs Human V2: {kappa_original_v2:.3f}")
print(f"- Original Vote vs GPT-4.1: {kappa_original_gpt4:.3f}")
print(f"- Human V2 vs GPT-4.1: {kappa_v2_gpt4:.3f}")

# 2. Krippendorff's Alpha for all three raters
# Create reliability data matrix where each row is a rater and each column is an item
reliability_data = np.array([original_numeric, humanv2_numeric, gpt4_numeric])

alpha_all_three = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
print(f"\nKrippendorff's Alpha (all three evaluators): {alpha_all_three:.3f}")

# 3. Pairwise Krippendorff's Alpha
alpha_original_v2 = krippendorff.alpha(reliability_data=np.array([original_numeric, humanv2_numeric]), 
                                       level_of_measurement='nominal')
alpha_original_gpt4 = krippendorff.alpha(reliability_data=np.array([original_numeric, gpt4_numeric]), 
                                         level_of_measurement='nominal')
alpha_v2_gpt4 = krippendorff.alpha(reliability_data=np.array([humanv2_numeric, gpt4_numeric]), 
                                   level_of_measurement='nominal')

print(f"\nPairwise Krippendorff's Alpha:")
print(f"- Original Vote vs Human V2: {alpha_original_v2:.3f}")
print(f"- Original Vote vs GPT-4.1: {alpha_original_gpt4:.3f}")
print(f"- Human V2 vs GPT-4.1: {alpha_v2_gpt4:.3f}")

# Create updated metrics report
with open('reliability_metrics_report.md', 'w') as f:
    f.write("# Complete Reliability Metrics Report\n\n")
    f.write("## Cohen's Kappa (Pairwise Agreement)\n")
    f.write(f"- **Original Vote vs Human V2**: {kappa_original_v2:.3f}\n")
    f.write(f"- **Original Vote vs GPT-4.1**: {kappa_original_gpt4:.3f}\n")
    f.write(f"- **Human V2 vs GPT-4.1**: {kappa_v2_gpt4:.3f}\n\n")
    
    f.write("## Krippendorff's Alpha\n")
    f.write(f"- **All three evaluators**: {alpha_all_three:.3f}\n")
    f.write(f"- **Original Vote vs Human V2**: {alpha_original_v2:.3f}\n")
    f.write(f"- **Original Vote vs GPT-4.1**: {alpha_original_gpt4:.3f}\n")
    f.write(f"- **Human V2 vs GPT-4.1**: {alpha_v2_gpt4:.3f}\n\n")
    
    f.write("## Interpretation Guidelines\n")
    f.write("### Cohen's Kappa:\n")
    f.write("- < 0: Poor agreement\n")
    f.write("- 0.00-0.20: Slight agreement\n")
    f.write("- 0.21-0.40: Fair agreement\n")
    f.write("- 0.41-0.60: Moderate agreement\n")
    f.write("- 0.61-0.80: Substantial agreement\n")
    f.write("- 0.81-1.00: Almost perfect agreement\n\n")
    
    f.write("### Krippendorff's Alpha:\n")
    f.write("- < 0.667: Insufficient agreement\n")
    f.write("- 0.667-0.800: Tentative agreement\n")
    f.write("- > 0.800: Good agreement\n")

print("\nMetrics report saved to reliability_metrics_report.md")