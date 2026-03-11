#!/usr/bin/env python3
"""
Compare o4-mini and GPT-4o evaluation results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load results
o4mini_df = pd.read_csv('o4mini_evaluation_final.csv')
gpt4o_df = pd.read_csv('gpt4o_proper_evaluation_fixed.csv')

# Merge on question
comparison_df = o4mini_df.merge(
    gpt4o_df[['question', 'gpt4o_preference', 'gpt4o_confidence', 'agrees_with_baseline']],
    on='question',
    suffixes=('_o4mini', '_gpt4o')
)

print("=== O4-MINI vs GPT-4O COMPARISON ===\n")

# Agreement rates with baseline
o4mini_agreement = comparison_df['agrees_with_baseline_o4mini'].mean()
gpt4o_agreement = comparison_df['agrees_with_baseline_gpt4o'].mean()

print(f"Baseline Agreement Rates:")
print(f"  o4-mini: {o4mini_agreement:.1%} ({comparison_df['agrees_with_baseline_o4mini'].sum()}/{len(comparison_df)})")
print(f"  GPT-4o:  {gpt4o_agreement:.1%} ({comparison_df['agrees_with_baseline_gpt4o'].sum()}/{len(comparison_df)})")

# Average confidence
o4mini_confidence = comparison_df['o4mini_confidence'].mean()
gpt4o_confidence = comparison_df['gpt4o_confidence'].mean()

print(f"\nAverage Confidence:")
print(f"  o4-mini: {o4mini_confidence:.3f}")
print(f"  GPT-4o:  {gpt4o_confidence:.3f}")

# Agreement between models
model_agreement = (comparison_df['o4mini_preference'] == comparison_df['gpt4o_preference']).sum()
print(f"\nAgreement between o4-mini and GPT-4o: {model_agreement}/{len(comparison_df)} ({model_agreement/len(comparison_df):.1%})")

# Preference distribution
print("\nPreference Distribution:")
for model, col in [('o4-mini', 'o4mini_preference'), ('GPT-4o', 'gpt4o_preference')]:
    print(f"\n{model}:")
    dist = comparison_df[col].value_counts()
    for pref, count in dist.items():
        print(f"  {pref}: {count} ({count/len(comparison_df):.1%})")

# Cases where models disagree
disagree_mask = comparison_df['o4mini_preference'] != comparison_df['gpt4o_preference']
disagree_df = comparison_df[disagree_mask]

print(f"\n\nDisagreements ({len(disagree_df)} cases):")
for _, row in disagree_df.iterrows():
    print(f"\n{row['question']}: {row['task'][:60]}...")
    print(f"  Baseline: {row['baseline_winner']}")
    print(f"  o4-mini:  {row['o4mini_preference']} (conf: {row['o4mini_confidence']:.2f})")
    print(f"  GPT-4o:   {row['gpt4o_preference']} (conf: {row['gpt4o_confidence']:.2f})")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Agreement rates comparison
ax = axes[0, 0]
models = ['o4-mini', 'GPT-4o']
agreements = [o4mini_agreement, gpt4o_agreement]
bars = ax.bar(models, agreements)
ax.set_ylabel('Agreement with Baseline')
ax.set_title('Model Agreement with Baseline')
ax.set_ylim(0, 1)
for i, (bar, val) in enumerate(zip(bars, agreements)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.1%}', ha='center', va='bottom')

# 2. Confidence comparison
ax = axes[0, 1]
ax.scatter(comparison_df['o4mini_confidence'], comparison_df['gpt4o_confidence'], alpha=0.6)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('o4-mini Confidence')
ax.set_ylabel('GPT-4o Confidence')
ax.set_title('Confidence Score Comparison')
ax.set_xlim(0.4, 1.0)
ax.set_ylim(0.4, 1.0)

# 3. Preference distribution
ax = axes[1, 0]
pref_data = pd.DataFrame({
    'o4-mini': comparison_df['o4mini_preference'].value_counts(),
    'GPT-4o': comparison_df['gpt4o_preference'].value_counts()
}).fillna(0)
pref_data.plot(kind='bar', ax=ax)
ax.set_title('Preference Distribution')
ax.set_ylabel('Count')
ax.set_xlabel('Preference')
ax.legend(title='Model')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# 4. Question-by-question comparison
ax = axes[1, 1]
questions = comparison_df['question'].values
x = range(len(questions))
width = 0.35

ax.bar([i - width/2 for i in x], 
       comparison_df['agrees_with_baseline_o4mini'].astype(int),
       width, label='o4-mini', alpha=0.8)
ax.bar([i + width/2 for i in x], 
       comparison_df['agrees_with_baseline_gpt4o'].astype(int),
       width, label='GPT-4o', alpha=0.8)

ax.set_xlabel('Question')
ax.set_ylabel('Agrees with Baseline (1=Yes, 0=No)')
ax.set_title('Per-Question Baseline Agreement')
ax.set_xticks(x)
ax.set_xticklabels(questions, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('o4mini_gpt4o_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nSaved comparison plot to o4mini_gpt4o_comparison.png")

# Save detailed comparison
comparison_df.to_csv('o4mini_gpt4o_detailed_comparison.csv', index=False)
print(f"Saved detailed comparison to o4mini_gpt4o_detailed_comparison.csv")

# Create summary statistics
summary = pd.DataFrame({
    'Metric': ['Baseline Agreement', 'Average Confidence', 'Inter-model Agreement', 
               'Agent 1 Preferences', 'Agent 2 Preferences', 'Tie Preferences'],
    'o4-mini': [
        f"{o4mini_agreement:.1%}",
        f"{o4mini_confidence:.3f}",
        f"{model_agreement/len(comparison_df):.1%}",
        f"{(comparison_df['o4mini_preference'] == 'Agent 1').sum()}",
        f"{(comparison_df['o4mini_preference'] == 'Agent 2').sum()}",
        f"{(comparison_df['o4mini_preference'] == 'Tie').sum()}"
    ],
    'GPT-4o': [
        f"{gpt4o_agreement:.1%}",
        f"{gpt4o_confidence:.3f}",
        f"{model_agreement/len(comparison_df):.1%}",
        f"{(comparison_df['gpt4o_preference'] == 'Agent 1').sum()}",
        f"{(comparison_df['gpt4o_preference'] == 'Agent 2').sum()}",
        f"{(comparison_df['gpt4o_preference'] == 'Tie').sum()}"
    ]
})

summary.to_csv('o4mini_gpt4o_summary.csv', index=False)
print(f"\nSaved summary to o4mini_gpt4o_summary.csv")