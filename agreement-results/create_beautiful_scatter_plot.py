#!/usr/bin/env python3
"""
Create a beautiful scatter plot of human agreement vs GPT-4o confidence.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Read the data
agreement_df = pd.read_csv('v2_per_question_agreement.csv')
confidence_df = pd.read_csv('question_to_gpt4o_confidence_exact.csv')
eval_df = pd.read_csv('survey_questions_gpt4o_exact_evaluations.csv')

# Merge the data
merged_df = agreement_df.merge(confidence_df, on='question', how='inner')
merged_df = merged_df.merge(eval_df[['question', 'baseline_winner', 'gpt4o_preference', 'agrees_with_baseline']], on='question', how='left')

# Create the figure
fig, ax = plt.subplots(figsize=(12, 9))

# Define color scheme
agree_color = '#2E86AB'  # Nice blue
disagree_color = '#E63946'  # Nice red
neutral_color = '#A8DADC'  # Light blue

# Define marker styles
marker_styles = {
    'Agent 1': {'marker': 'o', 'size': 150},
    'Agent 2': {'marker': 's', 'size': 150},
    'Tie': {'marker': '^', 'size': 180}
}

# Plot points by baseline winner and agreement
for winner in ['Agent 1', 'Agent 2', 'Tie']:
    mask_agree = (merged_df['baseline_winner'] == winner) & (merged_df['agrees_with_baseline'] == True)
    mask_disagree = (merged_df['baseline_winner'] == winner) & (merged_df['agrees_with_baseline'] == False)
    
    # Plot agreeing points
    if mask_agree.any():
        ax.scatter(merged_df[mask_agree]['gpt4o_confidence'], 
                  merged_df[mask_agree]['pairwise_agreement'],
                  s=marker_styles[winner]['size'], 
                  alpha=0.8, 
                  edgecolors='white', 
                  linewidth=2,
                  color=agree_color,
                  marker=marker_styles[winner]['marker'],
                  label=f'{winner} (GPT-4o agrees)' if winner == 'Agent 1' else None)
    
    # Plot disagreeing points
    if mask_disagree.any():
        ax.scatter(merged_df[mask_disagree]['gpt4o_confidence'], 
                  merged_df[mask_disagree]['pairwise_agreement'],
                  s=marker_styles[winner]['size'], 
                  alpha=0.8, 
                  edgecolors='white', 
                  linewidth=2,
                  color=disagree_color,
                  marker=marker_styles[winner]['marker'],
                  label=f'{winner} (GPT-4o disagrees)' if winner == 'Agent 1' else None)

# Add question labels with manual positioning to avoid overlaps
for idx, row in merged_df.iterrows():
    color = disagree_color if not row['agrees_with_baseline'] else 'black'
    
    # Manual offsets for better label positioning
    offset_x = 0.01
    offset_y = 0.01
    
    # Adjust specific overlapping labels
    if row['question'] in ['Q17', 'Q7']:
        offset_y = -0.02
    elif row['question'] in ['Q12', 'Q18']:
        offset_x = -0.02
    elif row['question'] in ['Q11']:
        offset_x = 0.02
        offset_y = 0.02
    
    ax.annotate(row['question'], 
                (row['gpt4o_confidence'] + offset_x, row['pairwise_agreement'] + offset_y),
                fontsize=10,
                color=color,
                weight='bold' if not row['agrees_with_baseline'] else 'normal',
                alpha=0.9,
                ha='center',
                va='center')

# Calculate and plot regression line
z = np.polyfit(merged_df['gpt4o_confidence'], merged_df['pairwise_agreement'], 1)
p = np.poly1d(z)
x_line = np.linspace(0.45, 0.98, 100)
ax.plot(x_line, p(x_line), color='#457B9D', linestyle='--', linewidth=2.5, alpha=0.8)

# Calculate statistics
correlation = merged_df['gpt4o_confidence'].corr(merged_df['pairwise_agreement'])
r2 = r2_score(merged_df['pairwise_agreement'], p(merged_df['gpt4o_confidence']))
_, p_value = stats.pearsonr(merged_df['gpt4o_confidence'], merged_df['pairwise_agreement'])

# Add statistics text box
stats_text = f'Pearson r = {correlation:.3f}\nR² = {r2:.3f}\np-value = {p_value:.3f}'
ax.text(0.48, 0.95, stats_text, 
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9),
        fontsize=12,
        verticalalignment='top')

# Customize axes
ax.set_xlabel('GPT-4o Confidence Score', fontsize=16, weight='bold')
ax.set_ylabel('Human Pairwise Agreement Rate', fontsize=16, weight='bold')
ax.set_title('Human Inter-Annotator Agreement vs GPT-4o Confidence\nby Survey Question', 
            fontsize=18, weight='bold', pad=20)

# Set axis limits with some padding
ax.set_xlim(0.45, 1.02)
ax.set_ylim(0.38, 1.02)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Customize legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=agree_color, 
               markersize=12, label='GPT-4o agrees with baseline', markeredgecolor='white', markeredgewidth=2),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=disagree_color, 
               markersize=12, label='GPT-4o disagrees with baseline', markeredgecolor='white', markeredgewidth=2),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Circle = Agent 1 baseline', alpha=0.5),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=10, label='Square = Agent 2 baseline', alpha=0.5),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
               markersize=10, label='Triangle = Tie baseline', alpha=0.5),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
         frameon=True, fancybox=True, shadow=True)

# Add subtle diagonal reference line
ax.plot([0.4, 1], [0.4, 1], 'k:', alpha=0.2, linewidth=1)

# Improve tick labels
ax.tick_params(axis='both', which='major', labelsize=12)

# Add percentage symbols to y-axis
y_ticks = ax.get_yticks()
ax.set_yticklabels([f'{int(y*100)}%' if y > 0 else '0%' for y in y_ticks])

# Make the plot tighter
plt.tight_layout()

# Save with high quality
plt.savefig('agreement_vs_confidence_beautiful.png', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.close()

print("Beautiful scatter plot saved as agreement_vs_confidence_beautiful.png")

# Also create a version with annotations for specific interesting points
fig, ax = plt.subplots(figsize=(12, 9))

# Replot everything (same as above but simpler)
for winner in ['Agent 1', 'Agent 2', 'Tie']:
    mask = merged_df['baseline_winner'] == winner
    colors = merged_df[mask]['agrees_with_baseline'].map({True: agree_color, False: disagree_color})
    
    ax.scatter(merged_df[mask]['gpt4o_confidence'], 
              merged_df[mask]['pairwise_agreement'],
              s=marker_styles[winner]['size'], 
              alpha=0.8, 
              edgecolors='white', 
              linewidth=2,
              color=colors,
              marker=marker_styles[winner]['marker'])

# Highlight specific interesting cases
interesting_cases = {
    'Q2': 'Highest human agreement\n(98.2%)',
    'Q22': 'Lowest human agreement\n(40.7%)',
    'Q8': 'High GPT-4o confidence (95%)\nbut GPT-4o disagreed',
    'Q11': 'Lowest GPT-4o confidence\n(50%, chose Tie)'
}

for q, desc in interesting_cases.items():
    row = merged_df[merged_df['question'] == q].iloc[0]
    ax.annotate(f'{q}\n{desc}',
                xy=(row['gpt4o_confidence'], row['pairwise_agreement']),
                xytext=(row['gpt4o_confidence'] + 0.05, row['pairwise_agreement'] - 0.05),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'),
                fontsize=10,
                weight='bold')

# Add other labels without descriptions
for idx, row in merged_df.iterrows():
    if row['question'] not in interesting_cases:
        color = disagree_color if not row['agrees_with_baseline'] else 'gray'
        ax.text(row['gpt4o_confidence'], row['pairwise_agreement'], row['question'],
                fontsize=9, color=color, alpha=0.7,
                ha='center', va='center')

# Add regression line and stats (same as before)
ax.plot(x_line, p(x_line), color='#457B9D', linestyle='--', linewidth=2.5, alpha=0.8)
ax.text(0.48, 0.95, stats_text, 
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9),
        fontsize=12,
        verticalalignment='top')

# Same styling as before
ax.set_xlabel('GPT-4o Confidence Score', fontsize=16, weight='bold')
ax.set_ylabel('Human Pairwise Agreement Rate', fontsize=16, weight='bold')
ax.set_title('Human Inter-Annotator Agreement vs GPT-4o Confidence\nwith Highlighted Cases', 
            fontsize=18, weight='bold', pad=20)
ax.set_xlim(0.45, 1.02)
ax.set_ylim(0.38, 1.02)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', labelsize=12)
y_ticks = ax.get_yticks()
ax.set_yticklabels([f'{int(y*100)}%' if y > 0 else '0%' for y in y_ticks])

# Simpler legend for this version
ax.legend(handles=legend_elements[:2], loc='lower right', fontsize=11, 
         frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('agreement_vs_confidence_annotated.png', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.close()

print("Annotated scatter plot saved as agreement_vs_confidence_annotated.png")