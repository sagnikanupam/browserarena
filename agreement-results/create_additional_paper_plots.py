#!/usr/bin/env python3
"""
Create additional paper-ready plots for comprehensive analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial']
})

# Load data
print("Loading data...")
o4mini_df = pd.read_csv('o4mini_evaluation_final.csv')
gpt4o_df = pd.read_csv('gpt4o_proper_evaluation_fixed.csv')
v2_human_df = pd.read_csv('BrowserArenaAgreementv2_July_18_2025_11.01.csv')
mapping_df = pd.read_csv('survey_questions_exact_mapping.csv')

# Create unified dataframe
unified_df = mapping_df[['question', 'task', 'baseline_winner']].copy()
unified_df = unified_df.merge(o4mini_df[['question', 'o4mini_preference', 'o4mini_confidence']], 
                              on='question', how='left')
unified_df = unified_df.merge(gpt4o_df[['question', 'gpt4o_preference', 'gpt4o_confidence']], 
                              on='question', how='left')

# ====================
# Plot 1: Inter-annotator Agreement for V2 Human Data
# ====================
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate per-question agreement rates for V2
agreement_rates = []
question_labels = []

for q in [f'Q{i}' for i in range(1, 23)]:
    if q in v2_human_df.columns:
        votes = v2_human_df[q].dropna()
        # Filter numeric votes only
        numeric_votes = []
        for v in votes:
            try:
                if isinstance(v, str) and v in ['1', '2', '3']:
                    numeric_votes.append(int(v))
                elif isinstance(v, (int, float)) and v in [1, 2, 3]:
                    numeric_votes.append(int(v))
            except:
                continue
        
        if len(numeric_votes) > 10:  # Need sufficient votes
            # Calculate pairwise agreement
            total_pairs = 0
            agreeing_pairs = 0
            
            for i in range(len(numeric_votes)):
                for j in range(i+1, len(numeric_votes)):
                    total_pairs += 1
                    if numeric_votes[i] == numeric_votes[j]:
                        agreeing_pairs += 1
            
            if total_pairs > 0:
                agreement_rate = agreeing_pairs / total_pairs
                agreement_rates.append(agreement_rate * 100)
                question_labels.append(q)

# Create bar plot
bars = ax.bar(range(len(agreement_rates)), agreement_rates, color='#2ca02c', alpha=0.7)
ax.set_xlabel('Question', fontsize=14)
ax.set_ylabel('Pairwise Agreement Rate (%)', fontsize=14)
ax.set_title('Human Inter-Annotator Agreement by Question (V2)', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(question_labels)))
ax.set_xticklabels(question_labels, rotation=45)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Add mean line
mean_agreement = np.mean(agreement_rates)
ax.axhline(y=mean_agreement, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {mean_agreement:.1f}%')
ax.legend()

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, agreement_rates)):
    if val > 5:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('human_inter_annotator_agreement.pdf', bbox_inches='tight')
print("Saved: human_inter_annotator_agreement.pdf")
plt.close()

# ====================
# Plot 2: Model vs Human Agreement Comparison
# ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: O4-mini vs Humans
for q in unified_df['question'].unique():
    q_data = unified_df[unified_df['question'] == q]
    
    # Get V2 human agreement rate
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
        
        if len(numeric_votes) > 10:
            # Get majority vote
            from collections import Counter
            vote_counts = Counter(numeric_votes)
            total_votes = sum(vote_counts.values())
            majority_pct = max(vote_counts.values()) / total_votes * 100
            
            # Get model confidence
            o4_conf = q_data['o4mini_confidence'].iloc[0]
            if pd.notna(o4_conf):
                ax1.scatter(majority_pct, o4_conf * 100, alpha=0.6, s=80, 
                           color='#d62728', edgecolors='black', linewidth=0.5)

ax1.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='y = x')
ax1.set_xlabel('Human Majority Agreement (%)', fontsize=14)
ax1.set_ylabel('O4-mini Confidence (%)', fontsize=14)
ax1.set_title('O4-mini Confidence vs Human Agreement', fontsize=14, fontweight='bold')
ax1.set_xlim(30, 100)
ax1.set_ylim(40, 100)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Right plot: GPT-4o vs Humans
for q in unified_df['question'].unique():
    q_data = unified_df[unified_df['question'] == q]
    
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
        
        if len(numeric_votes) > 10:
            from collections import Counter
            vote_counts = Counter(numeric_votes)
            total_votes = sum(vote_counts.values())
            majority_pct = max(vote_counts.values()) / total_votes * 100
            
            gpt_conf = q_data['gpt4o_confidence'].iloc[0]
            if pd.notna(gpt_conf):
                ax2.scatter(majority_pct, gpt_conf * 100, alpha=0.6, s=80,
                           color='#9467bd', edgecolors='black', linewidth=0.5)

ax2.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='y = x')
ax2.set_xlabel('Human Majority Agreement (%)', fontsize=14)
ax2.set_ylabel('GPT-4o Confidence (%)', fontsize=14)
ax2.set_title('GPT-4o Confidence vs Human Agreement', fontsize=14, fontweight='bold')
ax2.set_xlim(30, 100)
ax2.set_ylim(40, 100)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('model_confidence_vs_human_agreement.pdf', bbox_inches='tight')
print("Saved: model_confidence_vs_human_agreement.pdf")
plt.close()

# ====================
# Plot 3: Disagreement Analysis
# ====================
fig, ax = plt.subplots(figsize=(12, 6))

# Find disagreements between models and baseline
disagreements = []
for idx, row in unified_df.iterrows():
    q = row['question']
    baseline = row['baseline_winner']
    
    disagreement_types = []
    if pd.notna(row['o4mini_preference']) and row['o4mini_preference'] != baseline:
        disagreement_types.append('O4-mini')
    if pd.notna(row['gpt4o_preference']) and row['gpt4o_preference'] != baseline:
        disagreement_types.append('GPT-4o')
    
    if disagreement_types:
        disagreements.append({
            'Question': q,
            'Baseline': baseline,
            'O4-mini': row['o4mini_preference'],
            'GPT-4o': row['gpt4o_preference'],
            'Disagreeing': ', '.join(disagreement_types),
            'Task': row['task'][:50] + '...' if len(row['task']) > 50 else row['task']
        })

# Create disagreement matrix
disagree_df = pd.DataFrame(disagreements)
if len(disagree_df) > 0:
    # Count types of disagreements
    disagree_counts = {
        'Both Agree': 0,
        'Only O4-mini Disagrees': 0,
        'Only GPT-4o Disagrees': 0,
        'Both Disagree': 0
    }
    
    for _, row in unified_df.iterrows():
        o4_agrees = row['o4mini_preference'] == row['baseline_winner']
        gpt_agrees = row['gpt4o_preference'] == row['baseline_winner']
        
        if o4_agrees and gpt_agrees:
            disagree_counts['Both Agree'] += 1
        elif not o4_agrees and gpt_agrees:
            disagree_counts['Only O4-mini Disagrees'] += 1
        elif o4_agrees and not gpt_agrees:
            disagree_counts['Only GPT-4o Disagrees'] += 1
        else:
            disagree_counts['Both Disagree'] += 1
    
    # Create pie chart
    colors_pie = ['#2ca02c', '#ff7f0e', '#d62728', '#17becf']
    wedges, texts, autotexts = ax.pie(disagree_counts.values(), 
                                      labels=disagree_counts.keys(),
                                      colors=colors_pie,
                                      autopct='%1.0f%%',
                                      startangle=90)
    
    # Enhance text
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    ax.set_title('Model Agreement Patterns with Baseline', fontsize=16, fontweight='bold', pad=20)
    
    # Add counts to labels
    new_labels = []
    for label, count in disagree_counts.items():
        new_labels.append(f'{label}\n(n={count})')
    
    for text, new_label in zip(texts, new_labels):
        text.set_text(new_label)

plt.tight_layout()
plt.savefig('disagreement_patterns.pdf', bbox_inches='tight')
print("Saved: disagreement_patterns.pdf")
plt.close()

# ====================
# Plot 4: Tie Analysis
# ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Tie rates by evaluator
tie_rates = {}
for source, label in [
    ('baseline_winner', 'Baseline'),
    ('o4mini_preference', 'O4-mini'),
    ('gpt4o_preference', 'GPT-4o')
]:
    if source in unified_df.columns:
        ties = (unified_df[source] == 'Tie').sum()
        total = unified_df[source].notna().sum()
        tie_rates[label] = ties / total * 100 if total > 0 else 0

# Add V2 human tie rate
v2_ties = 0
v2_total = 0
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
            if mode_vote == 3:  # Tie
                v2_ties += 1
            v2_total += 1

if v2_total > 0:
    tie_rates['V2 Human'] = v2_ties / v2_total * 100

# Create bar plot
evaluators = list(tie_rates.keys())
rates = list(tie_rates.values())
colors_bar = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']

bars = ax1.bar(evaluators, rates, color=colors_bar[:len(evaluators)], alpha=0.8)
ax1.set_ylabel('Tie Rate (%)', fontsize=14)
ax1.set_title('Tie Rates by Evaluator', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 50)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar, rate in zip(bars, rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Right: Questions with highest tie rates
question_tie_data = []
for q in unified_df['question'].unique():
    q_data = unified_df[unified_df['question'] == q]
    tie_count = 0
    total_count = 0
    
    # Count ties across all evaluators
    for col in ['baseline_winner', 'o4mini_preference', 'gpt4o_preference']:
        if col in q_data.columns:
            vote = q_data[col].iloc[0]
            if pd.notna(vote):
                total_count += 1
                if vote == 'Tie':
                    tie_count += 1
    
    if total_count > 0:
        question_tie_data.append({
            'Question': q,
            'Tie_Rate': tie_count / total_count * 100,
            'Task': q_data['task'].iloc[0][:30] + '...'
        })

# Sort by tie rate and show top 10
tie_df = pd.DataFrame(question_tie_data).sort_values('Tie_Rate', ascending=False).head(10)

bars2 = ax2.barh(tie_df['Question'], tie_df['Tie_Rate'], color='#6B7280', alpha=0.8)
ax2.set_xlabel('Tie Rate Across Evaluators (%)', fontsize=14)
ax2.set_ylabel('Question', fontsize=14)
ax2.set_title('Questions with Highest Tie Rates', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 100)
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for bar, rate in zip(bars2, tie_df['Tie_Rate']):
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'{rate:.0f}%', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('tie_analysis.pdf', bbox_inches='tight')
print("Saved: tie_analysis.pdf")
plt.close()

# ====================
# Plot 5: Confidence Distribution by Vote Type
# ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# O4-mini confidence by vote type
vote_types = ['Agent 1', 'Agent 2', 'Tie']
o4_conf_by_vote = {vt: [] for vt in vote_types}

for idx, row in unified_df.iterrows():
    if pd.notna(row['o4mini_preference']) and pd.notna(row['o4mini_confidence']):
        o4_conf_by_vote[row['o4mini_preference']].append(row['o4mini_confidence'])

# Create box plot
o4_data = [o4_conf_by_vote[vt] for vt in vote_types if o4_conf_by_vote[vt]]
o4_labels = [vt for vt in vote_types if o4_conf_by_vote[vt]]

bp1 = ax1.boxplot(o4_data, labels=o4_labels, patch_artist=True, notch=True)
for patch, color in zip(bp1['boxes'], ['#4A90E2', '#E94B3C', '#6B7280']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_ylabel('Confidence Score', fontsize=14)
ax1.set_title('O4-mini Confidence by Vote Type', fontsize=14, fontweight='bold')
ax1.set_ylim(0.4, 1.0)
ax1.grid(axis='y', alpha=0.3)

# GPT-4o confidence by vote type
gpt_conf_by_vote = {vt: [] for vt in vote_types}

for idx, row in unified_df.iterrows():
    if pd.notna(row['gpt4o_preference']) and pd.notna(row['gpt4o_confidence']):
        gpt_conf_by_vote[row['gpt4o_preference']].append(row['gpt4o_confidence'])

# Create box plot
gpt_data = [gpt_conf_by_vote[vt] for vt in vote_types if gpt_conf_by_vote[vt]]
gpt_labels = [vt for vt in vote_types if gpt_conf_by_vote[vt]]

bp2 = ax2.boxplot(gpt_data, labels=gpt_labels, patch_artist=True, notch=True)
for patch, color in zip(bp2['boxes'], ['#4A90E2', '#E94B3C', '#6B7280']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_ylabel('Confidence Score', fontsize=14)
ax2.set_title('GPT-4o Confidence by Vote Type', fontsize=14, fontweight='bold')
ax2.set_ylim(0.4, 1.0)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('confidence_by_vote_type.pdf', bbox_inches='tight')
print("Saved: confidence_by_vote_type.pdf")
plt.close()

print("\nAll additional plots saved successfully!")
print("\nGenerated additional files:")
print("1. human_inter_annotator_agreement.pdf - Human agreement by question")
print("2. model_confidence_vs_human_agreement.pdf - Model confidence vs human consensus")
print("3. disagreement_patterns.pdf - Agreement patterns with baseline")
print("4. tie_analysis.pdf - Analysis of tie votes")
print("5. confidence_by_vote_type.pdf - Model confidence by vote type")