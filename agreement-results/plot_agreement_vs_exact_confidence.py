#!/usr/bin/env python3
"""
Plot per-question human agreement vs exact GPT-4o confidence scores.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score

# Read the per-question agreement data
print("Reading V2 per-question agreement data...")
agreement_df = pd.read_csv('v2_per_question_agreement.csv')

# Read the exact GPT-4o confidence data
print("Reading exact GPT-4o confidence data...")
confidence_df = pd.read_csv('question_to_gpt4o_confidence_exact.csv')

# Read the full evaluation results for additional context
eval_df = pd.read_csv('survey_questions_gpt4o_exact_evaluations.csv')

# Merge the data
merged_df = agreement_df.merge(confidence_df, on='question', how='inner')
merged_df = merged_df.merge(eval_df[['question', 'baseline_winner', 'gpt4o_preference', 'agrees_with_baseline']], on='question', how='left')

print(f"\nMerged data has {len(merged_df)} questions")

# Create the scatter plot
plt.figure(figsize=(10, 8))

# Color points by whether GPT-4o agreed with baseline
colors = merged_df['agrees_with_baseline'].map({True: 'steelblue', False: 'coral'})
markers = merged_df['baseline_winner'].map({'Agent 1': 'o', 'Agent 2': 's', 'Tie': '^'})

# Plot each point with different marker based on baseline winner
for winner in ['Agent 1', 'Agent 2', 'Tie']:
    mask = merged_df['baseline_winner'] == winner
    if mask.any():
        plt.scatter(merged_df[mask]['gpt4o_confidence'], 
                   merged_df[mask]['pairwise_agreement'], 
                   s=120, alpha=0.7, edgecolors='black', linewidth=1,
                   color=colors[mask], 
                   marker={'Agent 1': 'o', 'Agent 2': 's', 'Tie': '^'}[winner],
                   label=f'{winner} (n={mask.sum()})')

# Add labels for each point
for idx, row in merged_df.iterrows():
    color = 'red' if not row['agrees_with_baseline'] else 'black'
    plt.annotate(row['question'], 
                (row['gpt4o_confidence'], row['pairwise_agreement']),
                xytext=(5, 5), textcoords='offset points', fontsize=9,
                color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Calculate correlation and R²
correlation = merged_df['gpt4o_confidence'].corr(merged_df['pairwise_agreement'])
r2 = r2_score(merged_df['pairwise_agreement'], 
              np.poly1d(np.polyfit(merged_df['gpt4o_confidence'], 
                                  merged_df['pairwise_agreement'], 1))(merged_df['gpt4o_confidence']))

# Add regression line
z = np.polyfit(merged_df['gpt4o_confidence'], merged_df['pairwise_agreement'], 1)
p = np.poly1d(z)
x_line = np.linspace(0.4, 1.0, 100)
plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear fit (R² = {r2:.3f})')

# Calculate p-value
_, p_value = stats.pearsonr(merged_df['gpt4o_confidence'], merged_df['pairwise_agreement'])

print(f"\n=== CORRELATION ANALYSIS ===")
print(f"Pearson correlation: {correlation:.3f}")
print(f"R² value: {r2:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

# Additional statistics
print(f"\nGPT-4o agreement with baseline: {merged_df['agrees_with_baseline'].sum()}/{len(merged_df)}")
print(f"Questions where GPT-4o disagreed: {list(merged_df[~merged_df['agrees_with_baseline']]['question'])}")

plt.xlabel('GPT-4o Confidence Score', fontsize=14)
plt.ylabel('Human Pairwise Agreement Rate', fontsize=14)
plt.title('Human Inter-Annotator Agreement vs GPT-4o Confidence\n(Exact Evaluations)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, loc='upper left')

# Set axis limits
plt.xlim(0.4, 1.05)
plt.ylim(0.4, 1.05)

# Add diagonal reference line
plt.plot([0.4, 1], [0.4, 1], 'k:', alpha=0.3)

# Add text for disagreements
plt.text(0.42, 0.95, 'Red labels = GPT-4o disagreed with baseline', 
         fontsize=9, color='red', alpha=0.7)

plt.tight_layout()
plt.savefig('agreement_vs_confidence_scatter_exact.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nPlot saved as agreement_vs_confidence_scatter_exact.png")

# Create detailed summary table
summary_df = merged_df[['question', 'pairwise_agreement', 'gpt4o_confidence', 
                       'baseline_winner', 'gpt4o_preference', 'agrees_with_baseline']].copy()
summary_df = summary_df.sort_values('pairwise_agreement', ascending=False)
summary_df['pairwise_agreement'] = summary_df['pairwise_agreement'].round(3)
summary_df['gpt4o_confidence'] = summary_df['gpt4o_confidence'].round(3)
summary_df.to_csv('agreement_confidence_summary_exact.csv', index=False)

print("\nSummary saved to agreement_confidence_summary_exact.csv")

print("\nTop 5 questions by human agreement:")
print(summary_df.head()[['question', 'pairwise_agreement', 'gpt4o_confidence', 'gpt4o_preference']])

print("\nBottom 5 questions by human agreement:")
print(summary_df.tail()[['question', 'pairwise_agreement', 'gpt4o_confidence', 'gpt4o_preference']])

# Analyze relationship between agreement and confidence for different cases
print("\n=== ANALYSIS BY AGREEMENT TYPE ===")
agreed_mask = merged_df['agrees_with_baseline']
print(f"\nWhen GPT-4o agreed with baseline (n={agreed_mask.sum()}):")
print(f"  Avg human agreement: {merged_df[agreed_mask]['pairwise_agreement'].mean():.3f}")
print(f"  Avg GPT-4o confidence: {merged_df[agreed_mask]['gpt4o_confidence'].mean():.3f}")

print(f"\nWhen GPT-4o disagreed with baseline (n=(~agreed_mask).sum()):")
print(f"  Avg human agreement: {merged_df[~agreed_mask]['pairwise_agreement'].mean():.3f}")
print(f"  Avg GPT-4o confidence: {merged_df[~agreed_mask]['gpt4o_confidence'].mean():.3f}")