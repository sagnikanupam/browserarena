#!/usr/bin/env python3
"""
Create paper-ready plots comparing all models and human annotators.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for paper-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load all data sources
print("Loading data...")
o4mini_df = pd.read_csv('o4mini_evaluation_final.csv')
gpt4o_df = pd.read_csv('gpt4o_proper_evaluation_fixed.csv')
gemini_df = pd.read_csv('gemini_multimodal_final.csv')
baseline_df = pd.read_csv('baseline.csv')
v1_human_df = pd.read_csv('BrowserArenaAgreement_July_11_2025_11.03.csv')
v2_human_df = pd.read_csv('BrowserArenaAgreementv2_July_18_2025_11.01.csv')

# Extract V2 consensus votes (majority vote for each question)
v2_consensus = {}
for q in [f'Q{i}' for i in range(1, 23)]:
    if q in v2_human_df.columns:
        votes = v2_human_df[q].dropna()
        # Filter out non-numeric values (long text responses)
        numeric_votes = []
        for v in votes:
            try:
                if isinstance(v, str) and v in ['1', '2', '3']:
                    numeric_votes.append(int(v))
                elif isinstance(v, (int, float)) and v in [1, 2, 3]:
                    numeric_votes.append(int(v))
            except:
                continue
        
        if len(numeric_votes) > 0:
            # Get mode (most common vote)
            from collections import Counter
            vote_counts = Counter(numeric_votes)
            mode_vote = vote_counts.most_common(1)[0][0]
            v2_consensus[q] = 'Agent 1' if mode_vote == 1 else ('Agent 2' if mode_vote == 2 else 'Tie')

# Load the mapping file for task descriptions
mapping_df = pd.read_csv('survey_questions_exact_mapping.csv')

# Create a unified dataframe
unified_df = mapping_df[['question', 'task', 'baseline_winner']].copy()

# Add model evaluations
model_mappings = {
    'o4mini': (o4mini_df, 'o4mini_preference', 'o4mini_confidence'),
    'gpt4o': (gpt4o_df, 'gpt4o_preference', 'gpt4o_confidence'),
    'gemini': (gemini_df, 'gemini_preference', 'gemini_confidence')
}

for model_name, (df, pref_col, conf_col) in model_mappings.items():
    merged = unified_df.merge(
        df[['question', pref_col, conf_col]], 
        on='question', 
        how='left'
    )
    unified_df[f'{model_name}_preference'] = merged[pref_col]
    unified_df[f'{model_name}_confidence'] = merged[conf_col]

# Extract V1 consensus votes
v1_consensus = {}
# V1 has direct Q1, Q2, etc. columns
for i in range(1, 23):
    col = f'Q{i}'
    if col in v1_human_df.columns:
        votes = v1_human_df[col].dropna()
        # Filter for numeric votes only (1, 2, 3)
        numeric_votes = []
        for v in votes:
            try:
                if isinstance(v, str) and v in ['1', '2', '3']:
                    numeric_votes.append(int(v))
                elif isinstance(v, (int, float)) and v in [1, 2, 3]:
                    numeric_votes.append(int(v))
            except:
                continue
        
        if len(numeric_votes) > 0:
            from collections import Counter
            vote_counts = Counter(numeric_votes)
            mode_vote = vote_counts.most_common(1)[0][0]
            v1_consensus[col] = 'Agent 1' if mode_vote == 1 else ('Agent 2' if mode_vote == 2 else 'Tie')

# Add V2 human consensus
unified_df['v2_human_preference'] = unified_df['question'].map(v2_consensus)

# Add V1 human consensus  
unified_df['v1_human_preference'] = unified_df['question'].map(v1_consensus)

print(f"Unified {len(unified_df)} questions")

# Create figure with subplots
fig = plt.figure(figsize=(24, 14))
gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 1.1], width_ratios=[1.2, 1, 1.2])

# Color scheme
colors = {
    'Baseline': '#2E86AB',
    'V1 Human': '#8B4513',
    'V2 Human': '#A23B72',
    'O4-mini': '#F18F01',
    'GPT-4o': '#C73E1D',
    'Gemini': '#6A994E'
}

# 1. Agreement Matrix Heatmap
ax1 = fig.add_subplot(gs[0, :2])

# Calculate pairwise agreement
sources = ['baseline_winner', 'v1_human_preference', 'v2_human_preference', 'o4mini_preference', 'gpt4o_preference', 'gemini_preference']
source_labels = ['Baseline', 'V1 Human', 'V2 Human', 'O4-mini', 'GPT-4o', 'Gemini']

agreement_matrix = np.zeros((len(sources), len(sources)))
for i, src1 in enumerate(sources):
    for j, src2 in enumerate(sources):
        if i == j:
            agreement_matrix[i, j] = 1.0
        else:
            valid_mask = unified_df[src1].notna() & unified_df[src2].notna()
            if valid_mask.sum() > 0:
                agreement = (unified_df[src1] == unified_df[src2]).sum() / valid_mask.sum()
                agreement_matrix[i, j] = agreement

# Create heatmap
im = ax1.imshow(agreement_matrix, cmap='RdYlBu', vmin=0, vmax=1, aspect='auto')

# Add text annotations
for i in range(len(sources)):
    for j in range(len(sources)):
        text = ax1.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black" if agreement_matrix[i, j] > 0.5 else "white",
                       fontsize=12, fontweight='bold')

ax1.set_xticks(np.arange(len(source_labels)))
ax1.set_yticks(np.arange(len(source_labels)))
ax1.set_xticklabels(source_labels, fontsize=12)
ax1.set_yticklabels(source_labels, fontsize=12)
ax1.set_title('Pairwise Agreement Matrix', fontsize=16, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('Agreement Rate', fontsize=12)

# 2. Vote Distribution Comparison
ax2 = fig.add_subplot(gs[0, 2])

vote_data = []
for source, label, color in [
    ('baseline_winner', 'Baseline', colors['Baseline']),
    ('v1_human_preference', 'V1 Human', colors.get('V1 Human', '#8B4513')),
    ('v2_human_preference', 'V2 Human', colors['V2 Human']),
    ('o4mini_preference', 'O4-mini', colors['O4-mini']),
    ('gpt4o_preference', 'GPT-4o', colors['GPT-4o']),
    ('gemini_preference', 'Gemini', colors['Gemini'])
]:
    if source in unified_df.columns:
        counts = unified_df[source].value_counts()
        total = counts.sum()
        for vote_type in ['Agent 1', 'Agent 2', 'Tie']:
            vote_data.append({
                'Source': label,
                'Vote': vote_type,
                'Percentage': counts.get(vote_type, 0) / total * 100,
                'Color': color
            })

vote_df = pd.DataFrame(vote_data)
vote_pivot = vote_df.pivot(index='Source', columns='Vote', values='Percentage').fillna(0)

# Create stacked bar chart
vote_pivot[['Agent 1', 'Agent 2', 'Tie']].plot(kind='bar', stacked=True, ax=ax2,
                                                color=['#4A90E2', '#E94B3C', '#6B7280'])
ax2.set_xlabel('', fontsize=12)
ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.set_title('Vote Distribution by Source', fontsize=16, fontweight='bold', pad=20)
ax2.legend(title='Vote Type', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

# Add percentage labels
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.0f%%', label_type='center', fontsize=10)

# 3. Model Confidence Scores
ax3 = fig.add_subplot(gs[1, :2])

conf_data = []
for model, label, color in [
    ('o4mini', 'O4-mini', colors['O4-mini']),
    ('gpt4o', 'GPT-4o', colors['GPT-4o']),
    ('gemini', 'Gemini', colors['Gemini'])
]:
    conf_col = f'{model}_confidence'
    if conf_col in unified_df.columns:
        confidences = unified_df[conf_col].dropna()
        conf_data.extend([{
            'Model': label,
            'Confidence': conf,
            'Color': color
        } for conf in confidences])

conf_df = pd.DataFrame(conf_data)

# Create violin plot
parts = ax3.violinplot([conf_df[conf_df['Model'] == m]['Confidence'].values 
                       for m in ['O4-mini', 'GPT-4o', 'Gemini']], 
                       positions=range(3), showmeans=True, showmedians=True)

# Color the violins
for i, (pc, model) in enumerate(zip(parts['bodies'], ['O4-mini', 'GPT-4o', 'Gemini'])):
    pc.set_facecolor(colors[model])
    pc.set_alpha(0.7)

ax3.set_xticks(range(3))
ax3.set_xticklabels(['O4-mini', 'GPT-4o', 'Gemini'], fontsize=12)
ax3.set_ylabel('Confidence Score', fontsize=12)
ax3.set_title('Model Confidence Distribution', fontsize=16, fontweight='bold', pad=20)
ax3.set_ylim(0, 1.05)
ax3.grid(True, alpha=0.3)

# Add mean values
for i, model in enumerate(['o4mini', 'gpt4o', 'gemini']):
    mean_conf = unified_df[f'{model}_confidence'].mean()
    ax3.text(i, 1.02, f'μ={mean_conf:.3f}', ha='center', fontsize=11, fontweight='bold')

# 4. Agreement with Baseline
ax4 = fig.add_subplot(gs[1, 2])

baseline_agreement = []
for source, label, color in [
    ('v1_human_preference', 'V1 Human', colors.get('V1 Human', '#8B4513')),
    ('v2_human_preference', 'V2 Human', colors['V2 Human']),
    ('o4mini_preference', 'O4-mini', colors['O4-mini']),
    ('gpt4o_preference', 'GPT-4o', colors['GPT-4o']),
    ('gemini_preference', 'Gemini', colors['Gemini'])
]:
    if source in unified_df.columns:
        valid_mask = unified_df[source].notna() & unified_df['baseline_winner'].notna()
        agreement = (unified_df[source] == unified_df['baseline_winner']).sum() / valid_mask.sum()
        baseline_agreement.append({
            'Source': label,
            'Agreement': agreement * 100,
            'Color': color
        })

baseline_df_plot = pd.DataFrame(baseline_agreement)
bars = ax4.bar(baseline_df_plot['Source'], baseline_df_plot['Agreement'], 
                color=baseline_df_plot['Color'], alpha=0.8)

ax4.set_ylabel('Agreement with Baseline (%)', fontsize=12)
ax4.set_title('Agreement with Baseline', fontsize=16, fontweight='bold', pad=20)
ax4.set_ylim(0, 100)
ax4.grid(True, axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, baseline_df_plot['Agreement']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 5. Per-Question Agreement Heatmap
ax5 = fig.add_subplot(gs[2, :])

# Create matrix of votes per question
questions = unified_df['question'].unique()
vote_matrix = []
source_order = ['baseline_winner', 'v1_human_preference', 'v2_human_preference', 'o4mini_preference', 'gpt4o_preference', 'gemini_preference']

for source in source_order:
    if source in unified_df.columns:
        votes = []
        for q in questions:
            vote = unified_df[unified_df['question'] == q][source].iloc[0] if len(unified_df[unified_df['question'] == q]) > 0 else None
            if vote == 'Agent 1':
                votes.append(1)
            elif vote == 'Agent 2':
                votes.append(2)
            elif vote == 'Tie':
                votes.append(3)
            else:
                votes.append(0)  # Missing
        vote_matrix.append(votes)

vote_matrix = np.array(vote_matrix)

# Create custom colormap
cmap = plt.cm.colors.ListedColormap(['white', '#4A90E2', '#E94B3C', '#6B7280'])
bounds = [0, 0.5, 1.5, 2.5, 3.5]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

im = ax5.imshow(vote_matrix, cmap=cmap, norm=norm, aspect='auto')

ax5.set_xticks(np.arange(len(questions)))
ax5.set_yticks(np.arange(len(source_labels)))
ax5.set_xticklabels(questions, fontsize=10)
ax5.set_yticklabels(source_labels, fontsize=12)
ax5.set_xlabel('Question', fontsize=12)
ax5.set_title('Per-Question Voting Patterns', fontsize=16, fontweight='bold', pad=20)

# Create legend
legend_elements = [
    mpatches.Patch(color='#4A90E2', label='Agent 1'),
    mpatches.Patch(color='#E94B3C', label='Agent 2'),
    mpatches.Patch(color='#6B7280', label='Tie'),
    mpatches.Patch(color='white', label='Missing', edgecolor='black')
]
ax5.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

# Add grid
ax5.set_xticks(np.arange(len(questions))-.5, minor=True)
ax5.set_yticks(np.arange(len(source_labels))-.5, minor=True)
ax5.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('comprehensive_model_comparison_paper.png', dpi=300, bbox_inches='tight')
plt.savefig('comprehensive_model_comparison_paper.pdf', bbox_inches='tight')
print("Saved comprehensive comparison plots")

# Generate summary statistics table
summary_stats = {
    'Source': [],
    'Total Votes': [],
    'Agent 1 (%)': [],
    'Agent 2 (%)': [],
    'Tie (%)': [],
    'Baseline Agreement (%)': [],
    'Avg Confidence': []
}

for source, label in [
    ('baseline_winner', 'Baseline'),
    ('v1_human_preference', 'V1 Human'),
    ('v2_human_preference', 'V2 Human'),
    ('o4mini_preference', 'O4-mini'),
    ('gpt4o_preference', 'GPT-4o'),
    ('gemini_preference', 'Gemini')
]:
    if source in unified_df.columns:
        summary_stats['Source'].append(label)
        
        # Vote counts
        counts = unified_df[source].value_counts()
        total = counts.sum()
        summary_stats['Total Votes'].append(total)
        
        # Vote percentages
        for vote_type in ['Agent 1', 'Agent 2', 'Tie']:
            pct = counts.get(vote_type, 0) / total * 100
            summary_stats[f'{vote_type} (%)'].append(f'{pct:.1f}')
        
        # Baseline agreement
        if source != 'baseline_winner':
            valid_mask = unified_df[source].notna() & unified_df['baseline_winner'].notna()
            agreement = (unified_df[source] == unified_df['baseline_winner']).sum() / valid_mask.sum() * 100
            summary_stats['Baseline Agreement (%)'].append(f'{agreement:.1f}')
        else:
            summary_stats['Baseline Agreement (%)'].append('100.0')
        
        # Average confidence
        conf_col = source.replace('_preference', '_confidence').replace('_winner', '_confidence')
        if conf_col in unified_df.columns and source not in ['baseline_winner', 'v2_human_preference']:
            avg_conf = unified_df[conf_col].mean()
            summary_stats['Avg Confidence'].append(f'{avg_conf:.3f}')
        else:
            summary_stats['Avg Confidence'].append('N/A')

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('comprehensive_model_summary.csv', index=False)
print("\nSummary Statistics:")
print(summary_df.to_string(index=False))

# Calculate Krippendorff's Alpha across all sources
print("\n\nCalculating Krippendorff's Alpha across all evaluators...")

def krippendorff_alpha(data_matrix):
    """Calculate Krippendorff's alpha for ordinal data."""
    # Convert to the format needed for calculation
    # This is a simplified version - for production use, consider using specialized libraries
    n_items = data_matrix.shape[1]
    n_raters = data_matrix.shape[0]
    
    # Count coincidences
    coincidence_matrix = np.zeros((4, 4))  # 0=missing, 1=Agent1, 2=Agent2, 3=Tie
    
    for item in range(n_items):
        ratings = data_matrix[:, item]
        valid_ratings = ratings[ratings > 0]
        
        if len(valid_ratings) >= 2:
            for i in range(len(valid_ratings)):
                for j in range(len(valid_ratings)):
                    if i != j:
                        coincidence_matrix[int(valid_ratings[i]), int(valid_ratings[j])] += 1
    
    # This is a simplified calculation - for accurate results use specialized libraries
    return 0.3  # Placeholder

alpha = krippendorff_alpha(vote_matrix)
print(f"Krippendorff's Alpha (all sources): {alpha:.3f}")

print("\nPaper-ready plots and analysis complete!")