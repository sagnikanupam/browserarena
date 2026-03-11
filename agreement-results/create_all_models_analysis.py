#!/usr/bin/env python3
"""
Create comprehensive analysis with GPT-4o and Gemini evaluations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, cohen_kappa_score
import krippendorff

print("=== COMPREHENSIVE ANALYSIS: GPT-4o vs GEMINI ===")

# Read all data
baseline_df = pd.read_csv('baseline.csv')
human_v2_df = pd.read_csv('BrowserArenaAgreementv2_July_18_2025_11.01.csv')
gpt4o_df = pd.read_csv('gpt4o_proper_evaluation_final.csv')
gemini_df = pd.read_csv('gemini_evaluation_final.csv')
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
    gemini_row = gemini_df[gemini_df['question'] == q]
    
    if not baseline_row.empty and not gemini_row.empty:
        comparison_data.append({
            'Question': q,
            'Task': row['task'][:100] + '...' if len(row['task']) > 100 else row['task'],
            'Baseline': baseline_row.iloc[0]['winner'],
            'Human_V2': human_v2_majority.get(q, 'N/A'),
            'GPT4o': row['gpt4o_preference'],
            'Gemini': gemini_row.iloc[0]['gemini_preference'],
            'GPT4o_Confidence': row['gpt4o_confidence'],
            'Gemini_Confidence': gemini_row.iloc[0]['gemini_confidence'],
            'Human_Agreement_Rate': agreement_df[agreement_df['question'] == q]['pairwise_agreement'].iloc[0] if q in agreement_df['question'].values else 0,
            'All_Agree': (baseline_row.iloc[0]['winner'] == human_v2_majority.get(q) == row['gpt4o_preference'] == gemini_row.iloc[0]['gemini_preference']),
            'Baseline_GPT4o_Agree': (baseline_row.iloc[0]['winner'] == row['gpt4o_preference']),
            'Baseline_Gemini_Agree': (baseline_row.iloc[0]['winner'] == gemini_row.iloc[0]['gemini_preference']),
            'GPT4o_Gemini_Agree': (row['gpt4o_preference'] == gemini_row.iloc[0]['gemini_preference']),
            'Human_GPT4o_Agree': (human_v2_majority.get(q) == row['gpt4o_preference']),
            'Human_Gemini_Agree': (human_v2_majority.get(q) == gemini_row.iloc[0]['gemini_preference'])
        })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('all_models_comparison.csv', index=False)

# Calculate agreement statistics
print("\n=== AGREEMENT STATISTICS ===")
total = len(comparison_df)
all_agree = comparison_df['All_Agree'].sum()
baseline_gpt4o = comparison_df['Baseline_GPT4o_Agree'].sum()
baseline_gemini = comparison_df['Baseline_Gemini_Agree'].sum()
gpt4o_gemini = comparison_df['GPT4o_Gemini_Agree'].sum()
human_gpt4o = comparison_df['Human_GPT4o_Agree'].sum()
human_gemini = comparison_df['Human_Gemini_Agree'].sum()
baseline_human = sum((comparison_df['Baseline'] == comparison_df['Human_V2']))

print(f"Four-way agreement: {all_agree}/{total} ({all_agree/total*100:.1f}%)")
print(f"\nPairwise agreements:")
print(f"Baseline vs Human V2: {baseline_human}/{total} ({baseline_human/total*100:.1f}%)")
print(f"Baseline vs GPT-4o: {baseline_gpt4o}/{total} ({baseline_gpt4o/total*100:.1f}%)")
print(f"Baseline vs Gemini: {baseline_gemini}/{total} ({baseline_gemini/total*100:.1f}%)")
print(f"Human V2 vs GPT-4o: {human_gpt4o}/{total} ({human_gpt4o/total*100:.1f}%)")
print(f"Human V2 vs Gemini: {human_gemini}/{total} ({human_gemini/total*100:.1f}%)")
print(f"GPT-4o vs Gemini: {gpt4o_gemini}/{total} ({gpt4o_gemini/total*100:.1f}%)")

# Calculate Cohen's Kappa
labels_baseline = comparison_df['Baseline'].tolist()
labels_human = comparison_df['Human_V2'].tolist()
labels_gpt4o = comparison_df['GPT4o'].tolist()
labels_gemini = comparison_df['Gemini'].tolist()

print("\n=== COHEN'S KAPPA ===")
print(f"Baseline vs GPT-4o: {cohen_kappa_score(labels_baseline, labels_gpt4o):.3f}")
print(f"Baseline vs Gemini: {cohen_kappa_score(labels_baseline, labels_gemini):.3f}")
print(f"Human V2 vs GPT-4o: {cohen_kappa_score(labels_human, labels_gpt4o):.3f}")
print(f"Human V2 vs Gemini: {cohen_kappa_score(labels_human, labels_gemini):.3f}")
print(f"GPT-4o vs Gemini: {cohen_kappa_score(labels_gpt4o, labels_gemini):.3f}")

# Create scatter plot with both models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Define colors and markers
agree_color = '#2E86AB'
disagree_color = '#E63946'
marker_styles = {
    'Agent 1': {'marker': 'o', 'size': 150},
    'Agent 2': {'marker': 's', 'size': 150},
    'Tie': {'marker': '^', 'size': 180}
}

# Plot GPT-4o
for _, row in comparison_df.iterrows():
    baseline_winner = row['Baseline']
    agrees = row['Baseline_GPT4o_Agree']
    
    ax1.scatter(row['GPT4o_Confidence'], row['Human_Agreement_Rate'],
                s=marker_styles.get(baseline_winner, {'size': 150})['size'],
                marker=marker_styles.get(baseline_winner, {'marker': 'o'})['marker'],
                color=agree_color if agrees else disagree_color,
                alpha=0.8, edgecolors='white', linewidth=2)
    
    color = 'black' if agrees else disagree_color
    weight = 'normal' if agrees else 'bold'
    ax1.annotate(row['Question'], 
                (row['GPT4o_Confidence'] + 0.005, row['Human_Agreement_Rate'] + 0.005),
                fontsize=10, color=color, weight=weight, alpha=0.9)

# Calculate correlation for GPT-4o
correlation_gpt4o = comparison_df['GPT4o_Confidence'].corr(comparison_df['Human_Agreement_Rate'])
z_gpt4o = np.polyfit(comparison_df['GPT4o_Confidence'], comparison_df['Human_Agreement_Rate'], 1)
p_gpt4o = np.poly1d(z_gpt4o)
r2_gpt4o = r2_score(comparison_df['Human_Agreement_Rate'], p_gpt4o(comparison_df['GPT4o_Confidence']))
_, p_value_gpt4o = stats.pearsonr(comparison_df['GPT4o_Confidence'], comparison_df['Human_Agreement_Rate'])

x_line = np.linspace(0.55, 0.95, 100)
ax1.plot(x_line, p_gpt4o(x_line), '--', color='#457B9D', linewidth=2.5, alpha=0.8)

stats_text = f'Pearson r = {correlation_gpt4o:.3f}\\nR² = {r2_gpt4o:.3f}\\np-value = {p_value_gpt4o:.3f}'
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9),
         fontsize=12, verticalalignment='top')

ax1.set_xlabel('GPT-4o Confidence Score', fontsize=16, weight='bold')
ax1.set_ylabel('Human Pairwise Agreement Rate', fontsize=16, weight='bold')
ax1.set_title('Human Agreement vs GPT-4o Confidence\\n(with GIFs & Traces)', fontsize=18, weight='bold', pad=20)
ax1.set_xlim(0.55, 0.95)
ax1.set_ylim(0.38, 1.02)
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot Gemini
for _, row in comparison_df.iterrows():
    baseline_winner = row['Baseline']
    agrees = row['Baseline_Gemini_Agree']
    
    ax2.scatter(row['Gemini_Confidence'], row['Human_Agreement_Rate'],
                s=marker_styles.get(baseline_winner, {'size': 150})['size'],
                marker=marker_styles.get(baseline_winner, {'marker': 'o'})['marker'],
                color=agree_color if agrees else disagree_color,
                alpha=0.8, edgecolors='white', linewidth=2)
    
    color = 'black' if agrees else disagree_color
    weight = 'normal' if agrees else 'bold'
    ax2.annotate(row['Question'], 
                (row['Gemini_Confidence'] + 0.005, row['Human_Agreement_Rate'] + 0.005),
                fontsize=10, color=color, weight=weight, alpha=0.9)

# Calculate correlation for Gemini
correlation_gemini = comparison_df['Gemini_Confidence'].corr(comparison_df['Human_Agreement_Rate'])
z_gemini = np.polyfit(comparison_df['Gemini_Confidence'], comparison_df['Human_Agreement_Rate'], 1)
p_gemini = np.poly1d(z_gemini)
r2_gemini = r2_score(comparison_df['Human_Agreement_Rate'], p_gemini(comparison_df['Gemini_Confidence']))
_, p_value_gemini = stats.pearsonr(comparison_df['Gemini_Confidence'], comparison_df['Human_Agreement_Rate'])

x_line2 = np.linspace(0.45, 1.05, 100)
ax2.plot(x_line2, p_gemini(x_line2), '--', color='#457B9D', linewidth=2.5, alpha=0.8)

stats_text = f'Pearson r = {correlation_gemini:.3f}\\nR² = {r2_gemini:.3f}\\np-value = {p_value_gemini:.3f}'
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9),
         fontsize=12, verticalalignment='top')

ax2.set_xlabel('Gemini Confidence Score', fontsize=16, weight='bold')
ax2.set_ylabel('Human Pairwise Agreement Rate', fontsize=16, weight='bold')
ax2.set_title('Human Agreement vs Gemini Confidence\\n(Text-Only)', fontsize=18, weight='bold', pad=20)
ax2.set_xlim(0.45, 1.05)
ax2.set_ylim(0.38, 1.02)
ax2.grid(True, alpha=0.3, linestyle='--')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=agree_color, markersize=12, 
           label='Model agrees with baseline', markeredgecolor='white', markeredgewidth=2),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=disagree_color, markersize=12, 
           label='Model disagrees with baseline', markeredgecolor='white', markeredgewidth=2),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, 
           label='Circle = Agent 1 baseline', alpha=0.5),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, 
           label='Square = Agent 2 baseline', alpha=0.5),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, 
           label='Triangle = Tie baseline', alpha=0.5),
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=11, frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('all_models_confidence_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create vote distribution comparison
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

sources = ['Baseline', 'Human V2', 'GPT-4o', 'Gemini']
for i, (source, ax) in enumerate(zip(sources, axes)):
    if source == 'Baseline':
        votes = comparison_df['Baseline'].value_counts()
    elif source == 'Human V2':
        votes = comparison_df['Human_V2'].value_counts()
    elif source == 'GPT-4o':
        votes = comparison_df['GPT4o'].value_counts()
    else:
        votes = comparison_df['Gemini'].value_counts()
    
    # Ensure all categories
    for cat in ['Agent 1', 'Agent 2', 'Tie']:
        if cat not in votes:
            votes[cat] = 0
    votes = votes.reindex(['Agent 1', 'Agent 2', 'Tie'])
    
    colors = ['#2E86AB', '#E63946', '#F77F00']
    bars = ax.bar(votes.index, votes.values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}\\n({height/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_title(f'{source}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Questions', fontsize=12)
    ax.set_ylim(0, 20)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Vote Distribution Comparison Across All Evaluators', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('all_models_vote_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Create agreement matrix heatmap
agreement_matrix = pd.DataFrame(index=['Baseline', 'Human V2', 'GPT-4o', 'Gemini'],
                               columns=['Baseline', 'Human V2', 'GPT-4o', 'Gemini'])

# Calculate pairwise agreement percentages
agreement_matrix.loc['Baseline', 'Human V2'] = agreement_matrix.loc['Human V2', 'Baseline'] = baseline_human / total * 100
agreement_matrix.loc['Baseline', 'GPT-4o'] = agreement_matrix.loc['GPT-4o', 'Baseline'] = baseline_gpt4o / total * 100
agreement_matrix.loc['Baseline', 'Gemini'] = agreement_matrix.loc['Gemini', 'Baseline'] = baseline_gemini / total * 100
agreement_matrix.loc['Human V2', 'GPT-4o'] = agreement_matrix.loc['GPT-4o', 'Human V2'] = human_gpt4o / total * 100
agreement_matrix.loc['Human V2', 'Gemini'] = agreement_matrix.loc['Gemini', 'Human V2'] = human_gemini / total * 100
agreement_matrix.loc['GPT-4o', 'Gemini'] = agreement_matrix.loc['Gemini', 'GPT-4o'] = gpt4o_gemini / total * 100

# Fill diagonal
for source in agreement_matrix.index:
    agreement_matrix.loc[source, source] = 100

# Convert to numeric
agreement_matrix = agreement_matrix.astype(float)

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(agreement_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
            vmin=0, vmax=100, square=True, cbar_kws={'label': 'Agreement %'},
            annot_kws={'size': 14, 'weight': 'bold'})
plt.title('Pairwise Agreement Matrix (%)', fontsize=18, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('all_models_agreement_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nFigures saved:")
print("- all_models_confidence_comparison.png")
print("- all_models_vote_distribution.png")
print("- all_models_agreement_matrix.png")
print("- all_models_comparison.csv")

# Summary of key findings
print("\n=== KEY FINDINGS ===")
print(f"1. GPT-4o: mean confidence={comparison_df['GPT4o_Confidence'].mean():.3f}, std={comparison_df['GPT4o_Confidence'].std():.3f} (with GIFs)")
print(f"2. Gemini: mean confidence={comparison_df['Gemini_Confidence'].mean():.3f}, std={comparison_df['Gemini_Confidence'].std():.3f} (text-only)")
print(f"3. GPT-4o vs Gemini agreement: {gpt4o_gemini/total*100:.1f}%")
print(f"4. Correlation with human agreement:")
print(f"   - GPT-4o: r = {correlation_gpt4o:.3f} (p = {p_value_gpt4o:.3f})")
print(f"   - Gemini: r = {correlation_gemini:.3f} (p = {p_value_gemini:.3f})")

# Questions where models disagree
disagreements = comparison_df[~comparison_df['GPT4o_Gemini_Agree']]
print(f"\n5. Questions where GPT-4o and Gemini disagree ({len(disagreements)}):")
for _, row in disagreements.iterrows():
    print(f"   {row['Question']}: GPT-4o={row['GPT4o']} (conf:{row['GPT4o_Confidence']:.2f}), Gemini={row['Gemini']} (conf:{row['Gemini_Confidence']:.2f})")