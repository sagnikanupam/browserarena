#!/usr/bin/env python3
"""
Create separate, high-quality paper-ready plots without Gemini.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter

# Set style for paper-ready plots
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Load all data sources
print("Loading data...")
o4mini_df = pd.read_csv('o4mini_evaluation_final.csv')
gpt4o_df = pd.read_csv('gpt4o_proper_evaluation_fixed.csv')
baseline_df = pd.read_csv('baseline.csv')
v1_human_df = pd.read_csv('BrowserArenaAgreement_July_11_2025_11.03.csv')
v2_human_df = pd.read_csv('BrowserArenaAgreementv2_July_18_2025_11.01.csv')
mapping_df = pd.read_csv('survey_questions_exact_mapping.csv')

# Extract V2 consensus votes
v2_consensus = {}
for q in [f'Q{i}' for i in range(1, 23)]:
    if q in v2_human_df.columns:
        votes = v2_human_df[q].dropna()
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
            v2_consensus[q] = 'Agent 1' if mode_vote == 1 else ('Agent 2' if mode_vote == 2 else 'Tie')

# Extract V1 consensus votes
v1_consensus = {}
for i in range(1, 23):
    col = f'Q{i}'
    if col in v1_human_df.columns:
        votes = v1_human_df[col].dropna()
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

# Create unified dataframe
unified_df = mapping_df[['question', 'task', 'baseline_winner']].copy()

# Add model evaluations
unified_df = unified_df.merge(
    o4mini_df[['question', 'o4mini_preference', 'o4mini_confidence']], 
    on='question', how='left')
unified_df = unified_df.merge(
    gpt4o_df[['question', 'gpt4o_preference', 'gpt4o_confidence']], 
    on='question', how='left')

# Add human consensus
unified_df['v1_human_preference'] = unified_df['question'].map(v1_consensus)
unified_df['v2_human_preference'] = unified_df['question'].map(v2_consensus)

print(f"Unified {len(unified_df)} questions")

# Color scheme - professional and colorblind-friendly
colors = {
    'Baseline': '#1f77b4',  # Blue
    'V1 Human': '#ff7f0e',  # Orange
    'V2 Human': '#2ca02c',  # Green
    'O4-mini': '#d62728',   # Red
    'GPT-4o': '#9467bd'     # Purple
}

# ====================
# Plot 1: Agreement Matrix Heatmap
# ====================
fig, ax = plt.subplots(figsize=(10, 8))

sources = ['baseline_winner', 'v1_human_preference', 'v2_human_preference', 'o4mini_preference', 'gpt4o_preference']
source_labels = ['Baseline', 'V1 Human', 'V2 Human', 'O4-mini', 'GPT-4o']

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

# Create custom colormap
cmap = sns.diverging_palette(250, 10, as_cmap=True)
im = ax.imshow(agreement_matrix, cmap=cmap, vmin=0, vmax=1, aspect='equal')

# Add text annotations
for i in range(len(sources)):
    for j in range(len(sources)):
        if i == j:
            color = 'white'
        else:
            color = 'black' if agreement_matrix[i, j] > 0.5 else 'white'
        text = ax.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                      ha="center", va="center", color=color,
                      fontsize=14, fontweight='bold')

ax.set_xticks(np.arange(len(source_labels)))
ax.set_yticks(np.arange(len(source_labels)))
ax.set_xticklabels(source_labels)
ax.set_yticklabels(source_labels)
ax.set_title('Pairwise Agreement Between Evaluators', fontsize=18, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Agreement Rate', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Add grid
ax.set_xticks(np.arange(len(sources))-.5, minor=True)
ax.set_yticks(np.arange(len(sources))-.5, minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)

plt.tight_layout()
plt.savefig('agreement_matrix.pdf', bbox_inches='tight')
print("Saved: agreement_matrix.pdf")
plt.close()

# ====================
# Plot 2: Vote Distribution
# ====================
fig, ax = plt.subplots(figsize=(10, 6))

vote_data = []
for source, label in [
    ('baseline_winner', 'Baseline'),
    ('v1_human_preference', 'V1 Human'),
    ('v2_human_preference', 'V2 Human'),
    ('o4mini_preference', 'O4-mini'),
    ('gpt4o_preference', 'GPT-4o')
]:
    if source in unified_df.columns:
        counts = unified_df[source].value_counts()
        total = counts.sum()
        if total > 0:
            for vote_type in ['Agent 1', 'Agent 2', 'Tie']:
                vote_data.append({
                    'Source': label,
                    'Vote': vote_type,
                    'Count': counts.get(vote_type, 0),
                    'Percentage': counts.get(vote_type, 0) / total * 100
                })

vote_df = pd.DataFrame(vote_data)
vote_pivot = vote_df.pivot(index='Source', columns='Vote', values='Percentage').fillna(0)

# Reorder to match our source order
vote_pivot = vote_pivot.reindex(['Baseline', 'V1 Human', 'V2 Human', 'O4-mini', 'GPT-4o'])

# Create grouped bar chart
x = np.arange(len(vote_pivot.index))
width = 0.25

ax.bar(x - width, vote_pivot['Agent 1'], width, label='Agent 1', color='#4A90E2', alpha=0.8)
ax.bar(x, vote_pivot['Agent 2'], width, label='Agent 2', color='#E94B3C', alpha=0.8)
ax.bar(x + width, vote_pivot['Tie'], width, label='Tie', color='#6B7280', alpha=0.8)

ax.set_xlabel('Evaluator', fontsize=14)
ax.set_ylabel('Percentage of Votes', fontsize=14)
ax.set_title('Vote Distribution by Evaluator', fontsize=18, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(vote_pivot.index)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, source in enumerate(vote_pivot.index):
    for j, vote_type in enumerate(['Agent 1', 'Agent 2', 'Tie']):
        val = vote_pivot.loc[source, vote_type]
        if val > 5:  # Only show label if bar is tall enough
            ax.text(i + (j-1)*width, val + 1, f'{val:.0f}%', 
                   ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('vote_distribution.pdf', bbox_inches='tight')
print("Saved: vote_distribution.pdf")
plt.close()

# ====================
# Plot 3: Confidence Comparison
# ====================
fig, ax = plt.subplots(figsize=(8, 6))

# Prepare confidence data
conf_data = {
    'O4-mini': unified_df['o4mini_confidence'].dropna().values,
    'GPT-4o': unified_df['gpt4o_confidence'].dropna().values
}

# Create violin plot
positions = [0, 1]
parts = ax.violinplot([conf_data['O4-mini'], conf_data['GPT-4o']], 
                     positions=positions, showmeans=True, showmedians=True)

# Customize violin colors
for i, (pc, model) in enumerate(zip(parts['bodies'], ['O4-mini', 'GPT-4o'])):
    pc.set_facecolor(colors[model])
    pc.set_alpha(0.7)

# Customize other elements
parts['cmeans'].set_color('black')
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color('darkred')
parts['cmedians'].set_linewidth(2)

ax.set_xticks(positions)
ax.set_xticklabels(['O4-mini', 'GPT-4o'], fontsize=14)
ax.set_ylabel('Confidence Score', fontsize=14)
ax.set_title('Model Confidence Distribution', fontsize=18, fontweight='bold', pad=20)
ax.set_ylim(0.4, 1.05)
ax.grid(True, axis='y', alpha=0.3)

# Add mean and median labels
for i, model in enumerate(['o4mini', 'gpt4o']):
    data = unified_df[f'{model}_confidence'].dropna()
    mean_val = data.mean()
    median_val = data.median()
    ax.text(i, 1.02, f'μ={mean_val:.3f}', ha='center', fontsize=12, fontweight='bold')
    ax.text(i, 0.42, f'M={median_val:.3f}', ha='center', fontsize=12, style='italic')

plt.tight_layout()
plt.savefig('confidence_distribution.pdf', bbox_inches='tight')
print("Saved: confidence_distribution.pdf")
plt.close()

# ====================
# Plot 4: Agreement with Baseline
# ====================
fig, ax = plt.subplots(figsize=(8, 6))

agreement_data = []
for source, label in [
    ('v1_human_preference', 'V1 Human'),
    ('v2_human_preference', 'V2 Human'),
    ('o4mini_preference', 'O4-mini'),
    ('gpt4o_preference', 'GPT-4o')
]:
    if source in unified_df.columns:
        valid_mask = unified_df[source].notna() & unified_df['baseline_winner'].notna()
        agreement = (unified_df[source] == unified_df['baseline_winner']).sum() / valid_mask.sum() * 100
        agreement_data.append({
            'Evaluator': label,
            'Agreement': agreement
        })

agreement_df = pd.DataFrame(agreement_data)
bars = ax.bar(agreement_df['Evaluator'], agreement_df['Agreement'], 
               color=[colors[e] for e in agreement_df['Evaluator']], alpha=0.8)

ax.set_ylabel('Agreement with Baseline (%)', fontsize=14)
ax.set_title('Baseline Agreement by Evaluator', fontsize=18, fontweight='bold', pad=20)
ax.set_ylim(0, 80)
ax.grid(True, axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, agreement_df['Agreement']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add horizontal line at 50%
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random Agreement')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('baseline_agreement.pdf', bbox_inches='tight')
print("Saved: baseline_agreement.pdf")
plt.close()

# ====================
# Plot 5: Per-Question Agreement Pattern
# ====================
fig, ax = plt.subplots(figsize=(14, 6))

questions = unified_df['question'].unique()
evaluators = ['Baseline', 'V1 Human', 'V2 Human', 'O4-mini', 'GPT-4o']
source_cols = ['baseline_winner', 'v1_human_preference', 'v2_human_preference', 
               'o4mini_preference', 'gpt4o_preference']

# Create vote matrix
vote_matrix = []
for source in source_cols:
    votes = []
    for q in questions:
        vote = unified_df[unified_df['question'] == q][source].iloc[0]
        if pd.isna(vote):
            votes.append(0)  # Missing
        elif vote == 'Agent 1':
            votes.append(1)
        elif vote == 'Agent 2':
            votes.append(2)
        else:  # Tie
            votes.append(3)
    vote_matrix.append(votes)

vote_matrix = np.array(vote_matrix)

# Create custom colormap
cmap_colors = ['white', '#4A90E2', '#E94B3C', '#6B7280']
cmap = plt.cm.colors.ListedColormap(cmap_colors)
bounds = [0, 0.5, 1.5, 2.5, 3.5]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

im = ax.imshow(vote_matrix, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

ax.set_xticks(np.arange(len(questions)))
ax.set_yticks(np.arange(len(evaluators)))
ax.set_xticklabels(questions)
ax.set_yticklabels(evaluators)
ax.set_xlabel('Question', fontsize=14)
ax.set_title('Voting Patterns Across Questions', fontsize=18, fontweight='bold', pad=20)

# Add grid
ax.set_xticks(np.arange(len(questions))-.5, minor=True)
ax.set_yticks(np.arange(len(evaluators))-.5, minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)

# Create legend
legend_elements = [
    mpatches.Patch(color='#4A90E2', label='Agent 1'),
    mpatches.Patch(color='#E94B3C', label='Agent 2'),
    mpatches.Patch(color='#6B7280', label='Tie'),
    mpatches.Patch(facecolor='white', edgecolor='black', label='Missing')
]
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig('per_question_votes.pdf', bbox_inches='tight')
print("Saved: per_question_votes.pdf")
plt.close()

# ====================
# Plot 6: Scatter plot - Agreement vs Confidence
# ====================
fig, ax = plt.subplots(figsize=(8, 6))

# Calculate per-question agreement for each model
for model, model_col, conf_col, color, marker in [
    ('O4-mini', 'o4mini_preference', 'o4mini_confidence', colors['O4-mini'], 'o'),
    ('GPT-4o', 'gpt4o_preference', 'gpt4o_confidence', colors['GPT-4o'], 's')
]:
    confidences = []
    agreements = []
    
    for q in unified_df['question'].unique():
        q_data = unified_df[unified_df['question'] == q]
        
        # Get model confidence
        conf = q_data[conf_col].iloc[0]
        if pd.notna(conf):
            confidences.append(conf)
            
            # Check if model agrees with baseline
            model_vote = q_data[model_col].iloc[0]
            baseline_vote = q_data['baseline_winner'].iloc[0]
            agrees = 1 if model_vote == baseline_vote else 0
            agreements.append(agrees)
    
    # Add some jitter to agreement values for visualization
    agreements_jittered = np.array(agreements) + np.random.normal(0, 0.02, len(agreements))
    
    ax.scatter(confidences, agreements_jittered, label=model, color=color, 
               marker=marker, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Model Confidence', fontsize=14)
ax.set_ylabel('Agrees with Baseline', fontsize=14)
ax.set_title('Model Confidence vs Baseline Agreement', fontsize=18, fontweight='bold', pad=20)
ax.set_xlim(0.45, 1.05)
ax.set_ylim(-0.1, 1.1)
ax.set_yticks([0, 1])
ax.set_yticklabels(['No', 'Yes'])
ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('confidence_vs_agreement.pdf', bbox_inches='tight')
print("Saved: confidence_vs_agreement.pdf")
plt.close()

# ====================
# Plot 7: Summary Statistics Table (as figure)
# ====================
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

# Prepare summary data
summary_data = []
for source, label in [
    ('baseline_winner', 'Baseline'),
    ('v1_human_preference', 'V1 Human'),
    ('v2_human_preference', 'V2 Human'),
    ('o4mini_preference', 'O4-mini'),
    ('gpt4o_preference', 'GPT-4o')
]:
    if source in unified_df.columns:
        row = [label]
        
        # Vote counts
        counts = unified_df[source].value_counts()
        total = counts.sum()
        
        # Vote percentages
        for vote_type in ['Agent 1', 'Agent 2', 'Tie']:
            pct = counts.get(vote_type, 0) / total * 100 if total > 0 else 0
            row.append(f'{pct:.1f}%')
        
        # Baseline agreement
        if source != 'baseline_winner':
            valid_mask = unified_df[source].notna() & unified_df['baseline_winner'].notna()
            agreement = (unified_df[source] == unified_df['baseline_winner']).sum() / valid_mask.sum() * 100
            row.append(f'{agreement:.1f}%')
        else:
            row.append('—')
        
        # Average confidence
        if source.replace('_preference', '_confidence').replace('_winner', '_confidence') in unified_df.columns and \
           source not in ['baseline_winner', 'v1_human_preference', 'v2_human_preference']:
            conf_col = source.replace('_preference', '_confidence')
            avg_conf = unified_df[conf_col].mean()
            row.append(f'{avg_conf:.3f}')
        else:
            row.append('—')
        
        summary_data.append(row)

# Create table
columns = ['Evaluator', 'Agent 1', 'Agent 2', 'Tie', 'Baseline\nAgreement', 'Avg\nConfidence']
table = ax.table(cellText=summary_data, colLabels=columns, loc='center', cellLoc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)

# Header styling
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Row styling
for i in range(1, len(summary_data) + 1):
    for j in range(len(columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('white')

ax.set_title('Summary Statistics', fontsize=18, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('summary_statistics.pdf', bbox_inches='tight')
print("Saved: summary_statistics.pdf")
plt.close()

print("\nAll plots saved successfully!")
print("\nGenerated files:")
print("1. agreement_matrix.pdf - Pairwise agreement heatmap")
print("2. vote_distribution.pdf - Vote distribution by evaluator")
print("3. confidence_distribution.pdf - Model confidence violin plots")
print("4. baseline_agreement.pdf - Agreement with baseline bar chart")
print("5. per_question_votes.pdf - Voting patterns across questions")
print("6. confidence_vs_agreement.pdf - Confidence vs agreement scatter plot")
print("7. summary_statistics.pdf - Summary statistics table")