#!/usr/bin/env python3
"""
Create final comprehensive analysis with GPT-4o evaluations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, cohen_kappa_score
import krippendorff

print("=== FINAL COMPREHENSIVE ANALYSIS ===")

# Read all data
baseline_df = pd.read_csv('baseline.csv')
human_v2_df = pd.read_csv('BrowserArenaAgreementv2_July_18_2025_11.01.csv')
gpt4o_df = pd.read_csv('gpt4o_proper_evaluation_final.csv')
agreement_df = pd.read_csv('v2_per_question_agreement.csv')

# Process human V2 data
human_v2_df = human_v2_df.iloc[2:]
human_v2_df = human_v2_df[human_v2_df['Status'] == '0']

# Get survey questions
survey_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 
                   'Q11', 'Q12', 'Q13', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q22']

# Process human V2 majority votes
human_v2_majority = {}
human_v2_stats = {}

for q in survey_questions:
    if q in human_v2_df.columns:
        responses = human_v2_df[q].dropna()
        vote_counts = {'Agent 1': 0, 'Agent 2': 0, 'Tie': 0}
        
        for response in responses:
            response_str = str(response).strip()
            if response_str == '1':
                vote_counts['Agent 1'] += 1
            elif response_str == '2':
                vote_counts['Agent 2'] += 1
            elif response_str == '3':
                vote_counts['Tie'] += 1
        
        total_votes = sum(vote_counts.values())
        if total_votes > 0:
            majority = max(vote_counts.items(), key=lambda x: x[1])
            human_v2_majority[q] = majority[0]
            human_v2_stats[q] = {
                'votes': vote_counts,
                'total': total_votes,
                'majority_pct': (majority[1]/total_votes)*100
            }

# Create comprehensive comparison
comparison_data = []
for _, row in gpt4o_df.iterrows():
    q = row['question']
    baseline_row = baseline_df[baseline_df['question'] == q]
    
    if not baseline_row.empty:
        comparison_data.append({
            'Question': q,
            'Task': row['task'][:100] + '...' if len(row['task']) > 100 else row['task'],
            'Baseline': baseline_row.iloc[0]['winner'],
            'Human_V2': human_v2_majority.get(q, 'N/A'),
            'GPT4o': row['gpt4o_preference'],
            'GPT4o_Confidence': row['gpt4o_confidence'],
            'Human_Agreement_Rate': agreement_df[agreement_df['question'] == q]['pairwise_agreement'].iloc[0] if q in agreement_df['question'].values else 0,
            'All_Agree': (baseline_row.iloc[0]['winner'] == human_v2_majority.get(q) == row['gpt4o_preference']),
            'Baseline_GPT4o_Agree': (baseline_row.iloc[0]['winner'] == row['gpt4o_preference']),
            'Human_GPT4o_Agree': (human_v2_majority.get(q) == row['gpt4o_preference'])
        })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('final_comprehensive_comparison.csv', index=False)

# Calculate agreement statistics
print("\n=== AGREEMENT STATISTICS ===")
total = len(comparison_df)
all_agree = comparison_df['All_Agree'].sum()
baseline_gpt4o = comparison_df['Baseline_GPT4o_Agree'].sum()
baseline_human = sum((comparison_df['Baseline'] == comparison_df['Human_V2']))
human_gpt4o = comparison_df['Human_GPT4o_Agree'].sum()

print(f"Three-way agreement: {all_agree}/{total} ({all_agree/total*100:.1f}%)")
print(f"\nPairwise agreements:")
print(f"Baseline vs Human V2: {baseline_human}/{total} ({baseline_human/total*100:.1f}%)")
print(f"Baseline vs GPT-4o: {baseline_gpt4o}/{total} ({baseline_gpt4o/total*100:.1f}%)")
print(f"Human V2 vs GPT-4o: {human_gpt4o}/{total} ({human_gpt4o/total*100:.1f}%)")

# Calculate Cohen's Kappa
labels_baseline = comparison_df['Baseline'].tolist()
labels_human = comparison_df['Human_V2'].tolist()
labels_gpt4o = comparison_df['GPT4o'].tolist()

print("\n=== COHEN'S KAPPA ===")
print(f"Baseline vs Human V2: {cohen_kappa_score(labels_baseline, labels_human):.3f}")
print(f"Baseline vs GPT-4o: {cohen_kappa_score(labels_baseline, labels_gpt4o):.3f}")
print(f"Human V2 vs GPT-4o: {cohen_kappa_score(labels_human, labels_gpt4o):.3f}")

# Create beautiful scatter plot
plt.figure(figsize=(14, 10))

# Define colors and markers
agree_color = '#2E86AB'
disagree_color = '#E63946'
marker_styles = {
    'Agent 1': {'marker': 'o', 'size': 200},
    'Agent 2': {'marker': 's', 'size': 200},
    'Tie': {'marker': '^', 'size': 220}
}

# Plot points
for _, row in comparison_df.iterrows():
    baseline_winner = row['Baseline']
    agrees = row['Baseline_GPT4o_Agree']
    
    plt.scatter(row['GPT4o_Confidence'], row['Human_Agreement_Rate'],
                s=marker_styles.get(baseline_winner, {'size': 200})['size'],
                marker=marker_styles.get(baseline_winner, {'marker': 'o'})['marker'],
                color=agree_color if agrees else disagree_color,
                alpha=0.85, edgecolors='white', linewidth=2.5)
    
    # Add labels
    color = 'black' if agrees else disagree_color
    weight = 'normal' if agrees else 'bold'
    plt.annotate(row['Question'], 
                (row['GPT4o_Confidence'] + 0.005, row['Human_Agreement_Rate'] + 0.005),
                fontsize=11, color=color, weight=weight, alpha=0.9)

# Calculate correlation
correlation = comparison_df['GPT4o_Confidence'].corr(comparison_df['Human_Agreement_Rate'])
z = np.polyfit(comparison_df['GPT4o_Confidence'], comparison_df['Human_Agreement_Rate'], 1)
p = np.poly1d(z)
r2 = r2_score(comparison_df['Human_Agreement_Rate'], p(comparison_df['GPT4o_Confidence']))
_, p_value = stats.pearsonr(comparison_df['GPT4o_Confidence'], comparison_df['Human_Agreement_Rate'])

# Add regression line
x_line = np.linspace(0.55, 0.95, 100)
plt.plot(x_line, p(x_line), '--', color='#457B9D', linewidth=3, alpha=0.8)

# Add statistics box
stats_text = f'Pearson r = {correlation:.3f}\nR² = {r2:.3f}\np-value = {p_value:.3f}'
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='gray', alpha=0.95),
         fontsize=13, verticalalignment='top', weight='bold')

# Styling
plt.xlabel('GPT-4o Confidence Score (with GIFs & Traces)', fontsize=18, weight='bold')
plt.ylabel('Human Pairwise Agreement Rate', fontsize=18, weight='bold')
plt.title('Human Agreement vs GPT-4o Confidence\n(Full Evaluation with Visual Evidence)', 
         fontsize=20, weight='bold', pad=25)
plt.xlim(0.55, 0.95)
plt.ylim(0.38, 1.02)
plt.grid(True, alpha=0.35, linestyle='--')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=agree_color, markersize=14, 
           label='GPT-4o agrees with baseline', markeredgecolor='white', markeredgewidth=2),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=disagree_color, markersize=14, 
           label='GPT-4o disagrees with baseline', markeredgecolor='white', markeredgewidth=2),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, 
           label='Circle = Agent 1 baseline', alpha=0.6),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=12, 
           label='Square = Agent 2 baseline', alpha=0.6),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=12, 
           label='Triangle = Tie baseline', alpha=0.6),
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=12, frameon=True, 
          fancybox=True, shadow=True, framealpha=0.95)

plt.tight_layout()
plt.savefig('final_agreement_vs_confidence.png', dpi=300, bbox_inches='tight')
plt.close()

# Create vote distribution comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

sources = ['Baseline', 'Human V2', 'GPT-4o']
for i, (source, ax) in enumerate(zip(sources, axes)):
    if source == 'Baseline':
        votes = comparison_df['Baseline'].value_counts()
    elif source == 'Human V2':
        votes = comparison_df['Human_V2'].value_counts()
    else:
        votes = comparison_df['GPT4o'].value_counts()
    
    # Ensure all categories
    for cat in ['Agent 1', 'Agent 2', 'Tie']:
        if cat not in votes:
            votes[cat] = 0
    votes = votes.reindex(['Agent 1', 'Agent 2', 'Tie'])
    
    colors = ['#2E86AB', '#E63946', '#F77F00']
    bars = ax.bar(votes.index, votes.values, color=colors, alpha=0.85, edgecolor='white', linewidth=2.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}\n({height/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax.set_title(f'{source}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Questions', fontsize=14, weight='bold')
    ax.set_ylim(0, 20)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Vote Distribution Comparison', fontsize=20, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig('final_vote_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Create agreement matrix heatmap
agreement_matrix = pd.DataFrame(index=['Baseline', 'Human V2', 'GPT-4o'],
                               columns=['Baseline', 'Human V2', 'GPT-4o'])

# Calculate pairwise agreement percentages
agreement_matrix.loc['Baseline', 'Human V2'] = agreement_matrix.loc['Human V2', 'Baseline'] = baseline_human / total * 100
agreement_matrix.loc['Baseline', 'GPT-4o'] = agreement_matrix.loc['GPT-4o', 'Baseline'] = baseline_gpt4o / total * 100
agreement_matrix.loc['Human V2', 'GPT-4o'] = agreement_matrix.loc['GPT-4o', 'Human V2'] = human_gpt4o / total * 100

# Fill diagonal
for source in agreement_matrix.index:
    agreement_matrix.loc[source, source] = 100

# Convert to numeric
agreement_matrix = agreement_matrix.astype(float)

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(agreement_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
            vmin=0, vmax=100, square=True, cbar_kws={'label': 'Agreement %'},
            annot_kws={'size': 16, 'weight': 'bold'}, linewidths=2, linecolor='white')
plt.title('Pairwise Agreement Matrix (%)', fontsize=20, weight='bold', pad=25)
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=14, weight='bold')
plt.yticks(fontsize=14, weight='bold', rotation=0)
plt.tight_layout()
plt.savefig('final_agreement_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nFigures saved:")
print("- final_agreement_vs_confidence.png")
print("- final_vote_distribution.png")
print("- final_agreement_matrix.png")
print("- final_comprehensive_comparison.csv")

# Summary of key findings
print("\n=== KEY FINDINGS ===")
print(f"1. GPT-4o confidence: mean={comparison_df['GPT4o_Confidence'].mean():.3f}, std={comparison_df['GPT4o_Confidence'].std():.3f}")
print(f"2. Correlation between GPT-4o confidence and human agreement: r = {correlation:.3f} (p = {p_value:.3f})")
print(f"3. GPT-4o evaluated all {len(comparison_df)} questions with both GIFs and full vLLM traces")

# Questions where GPT-4o disagrees with baseline
disagreements = comparison_df[~comparison_df['Baseline_GPT4o_Agree']]
print(f"\n4. Questions where GPT-4o disagreed with baseline ({len(disagreements)}):")
for _, row in disagreements.iterrows():
    print(f"   {row['Question']}: Baseline={row['Baseline']}, GPT-4o={row['GPT4o']} (conf: {row['GPT4o_Confidence']:.2f})")

# Questions with perfect agreement
perfect_agreement = comparison_df[comparison_df['All_Agree']]
print(f"\n5. Questions with perfect three-way agreement ({len(perfect_agreement)}):")
for _, row in perfect_agreement.iterrows():
    print(f"   {row['Question']}: All chose {row['Baseline']}")

print("\nAnalysis complete!")