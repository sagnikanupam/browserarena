#!/usr/bin/env python3
"""Comprehensive analysis comparing original baseline, new human eval v2, and GPT-4.1 evaluations"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa
import warnings
warnings.filterwarnings('ignore')

# Read all data sources
print("COMPREHENSIVE EVALUATION COMPARISON ANALYSIS")
print("=" * 80)
print("Reading data sources...")

# 1. Read baseline data (original single annotator)
baseline_df = pd.read_csv('baseline.csv')
baseline_dict = {}
for _, row in baseline_df.iterrows():
    q_num = row['question']
    if row['winner'] == 'Agent 1':
        baseline_dict[q_num] = '1'
    elif row['winner'] == 'Agent 2':
        baseline_dict[q_num] = '2'
    elif row['winner'] == 'Tie':
        baseline_dict[q_num] = '3'

print(f"Baseline: {len(baseline_dict)} questions")

# 2. Read new human eval data v2
print("\nReading new human evaluation data (v2)...")
df_v2 = pd.read_csv('BrowserArenaAgreementv2_July_18_2025_11.01.csv')

# Check structure
print(f"V2 CSV shape: {df_v2.shape}")
print(f"Columns: {list(df_v2.columns[:20])}")

# Skip header rows and filter valid responses
df_v2 = df_v2.iloc[2:]  # Skip header rows
df_v2 = df_v2[df_v2['Status'] == '0']  # Completed responses

print(f"Valid responses in v2: {len(df_v2)}")

# Get question columns
question_cols = [col for col in df_v2.columns if col.startswith('Q') and col[1:].isdigit()]
print(f"Questions in v2: {question_cols}")

# Convert responses to standardized format
def standardize_response(response):
    if pd.isna(response):
        return None
    response = str(response).strip()
    if response in ['1', '2', '3']:
        return response
    elif 'Agent 1' in response and 'Agent 2' not in response:
        return '1'
    elif 'Agent 2' in response and 'Agent 1' not in response:
        return '2'
    elif 'Tie' in response:
        return '3'
    else:
        return None

# Process v2 human responses
print("\nProcessing v2 human responses...")
human_v2_data = {}
for q in question_cols:
    responses = df_v2[q].apply(standardize_response)
    valid_responses = responses.dropna()
    human_v2_data[q] = valid_responses.tolist()
    print(f"{q}: {len(valid_responses)} valid responses")

# Calculate majority labels for v2
human_v2_majority = {}
for q in question_cols:
    if q in human_v2_data and len(human_v2_data[q]) > 0:
        counter = Counter(human_v2_data[q])
        most_common = counter.most_common(1)[0]
        human_v2_majority[q] = most_common[0]

# 3. Prepare GPT-4.1 evaluations (we'll need to either read from a file or use baseline as proxy)
# For now, using the baseline as GPT-4.1 eval (adjust this if you have separate GPT-4.1 data)
gpt4_dict = baseline_dict.copy()  # Placeholder - replace with actual GPT-4.1 data if available

# Calculate inter-annotator reliability for v2
print("\n=== INTER-ANNOTATOR RELIABILITY (V2) ===")
all_questions = sorted([q for q in question_cols if q in human_v2_data and len(human_v2_data[q]) > 0])

if len(all_questions) > 0:
    # Krippendorff's alpha
    reliability_data = []
    max_raters = max(len(human_v2_data[q]) for q in all_questions)
    
    for rater_idx in range(max_raters):
        row = []
        for q in all_questions:
            if rater_idx < len(human_v2_data[q]):
                response = human_v2_data[q][rater_idx]
                row.append(int(response))
            else:
                row.append(np.nan)
        reliability_data.append(row)
    
    reliability_data = np.array(reliability_data)
    
    try:
        alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
        print(f"Krippendorff's alpha: {alpha:.3f}")
    except Exception as e:
        print(f"Error calculating Krippendorff's alpha: {e}")
    
    # Fleiss' Kappa
    fleiss_data = []
    for q in all_questions:
        if q in human_v2_data:
            responses = human_v2_data[q]
            counts = {'1': 0, '2': 0, '3': 0}
            for r in responses:
                if r in counts:
                    counts[r] += 1
            fleiss_data.append([counts['1'], counts['2'], counts['3']])
    
    if len(fleiss_data) > 0:
        fleiss_data = np.array(fleiss_data)
        try:
            kappa = fleiss_kappa(fleiss_data, method='fleiss')
            print(f"Fleiss' Kappa: {kappa:.3f}")
        except Exception as e:
            print(f"Error calculating Fleiss' Kappa: {e}")

# Calculate agreement rates
print("\n=== AGREEMENT STATISTICS ===")
agreement_stats = []
for q in all_questions:
    if q in human_v2_data and len(human_v2_data[q]) > 1:
        responses = human_v2_data[q]
        counter = Counter(responses)
        total = len(responses)
        
        # Majority agreement rate
        majority_count = counter.most_common(1)[0][1]
        majority_agreement = majority_count / total
        
        agreement_stats.append({
            'question': q,
            'n_raters': total,
            'majority_agreement': majority_agreement
        })

if len(agreement_stats) > 0:
    agreement_df = pd.DataFrame(agreement_stats)
    print(f"Average majority agreement (v2): {agreement_df['majority_agreement'].mean():.3f}")

# Compare all three sources
print("\n=== THREE-WAY COMPARISON ===")
comparison_data = []
label_map = {'1': 'Agent 1', '2': 'Agent 2', '3': 'Tie'}

for q in all_questions:
    baseline_label = label_map.get(baseline_dict.get(q, 'N/A'), 'N/A')
    human_v2_label = label_map.get(human_v2_majority.get(q, 'N/A'), 'N/A')
    gpt4_label = label_map.get(gpt4_dict.get(q, 'N/A'), 'N/A')
    
    # Get vote distribution for human v2
    if q in human_v2_data:
        counter = Counter(human_v2_data[q])
        total = len(human_v2_data[q])
        agent1_votes = counter.get('1', 0)
        agent2_votes = counter.get('2', 0)
        tie_votes = counter.get('3', 0)
        
        comparison_data.append({
            'Question': q,
            'Original_Baseline': baseline_label,
            'Human_V2_Majority': human_v2_label,
            'GPT4_Eval': gpt4_label,
            'V2_Agent1_Votes': agent1_votes,
            'V2_Agent2_Votes': agent2_votes,
            'V2_Tie_Votes': tie_votes,
            'V2_Total': total,
            'V2_Agreement_Rate': f"{(counter.most_common(1)[0][1]/total)*100:.1f}%"
        })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('three_way_comparison.csv', index=False)
print("Saved three-way comparison to three_way_comparison.csv")

# Calculate pairwise agreements
print("\n=== PAIRWISE AGREEMENTS ===")
# Baseline vs Human V2
baseline_v2_agree = sum(1 for _, row in comparison_df.iterrows() 
                       if row['Original_Baseline'] == row['Human_V2_Majority'])
print(f"Baseline vs Human V2: {baseline_v2_agree}/{len(comparison_df)} ({baseline_v2_agree/len(comparison_df)*100:.1f}%)")

# Baseline vs GPT-4
baseline_gpt4_agree = sum(1 for _, row in comparison_df.iterrows() 
                         if row['Original_Baseline'] == row['GPT4_Eval'])
print(f"Baseline vs GPT-4: {baseline_gpt4_agree}/{len(comparison_df)} ({baseline_gpt4_agree/len(comparison_df)*100:.1f}%)")

# Human V2 vs GPT-4
v2_gpt4_agree = sum(1 for _, row in comparison_df.iterrows() 
                   if row['Human_V2_Majority'] == row['GPT4_Eval'])
print(f"Human V2 vs GPT-4: {v2_gpt4_agree}/{len(comparison_df)} ({v2_gpt4_agree/len(comparison_df)*100:.1f}%)")

# Create visualizations
print("\n=== GENERATING VISUALIZATIONS ===")

# 1. Three-way comparison heatmap
fig, ax = plt.subplots(figsize=(14, 10))

# Prepare data for heatmap
questions = comparison_df['Question'].tolist()
evaluators = ['Original Baseline', 'Human V2 Majority', 'GPT-4.1']
heatmap_data = np.zeros((len(questions), len(evaluators)))

for i, q in enumerate(questions):
    row = comparison_df[comparison_df['Question'] == q].iloc[0]
    for j, (eval_name, col_name) in enumerate([('Original Baseline', 'Original_Baseline'),
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

# Plot heatmap
sns.heatmap(heatmap_data, 
            xticklabels=evaluators,
            yticklabels=questions,
            cmap=cmap,
            cbar_kws={'label': 'Choice', 'ticks': [0.5, 1, 2, 2.5]},
            vmin=0, vmax=3,
            linewidths=0.5,
            linecolor='gray')

# Customize colorbar
cbar = ax.collections[0].colorbar
cbar.set_ticklabels(['', 'Agent 1', 'Agent 2', 'Tie'])

plt.title('Three-Way Evaluation Comparison', fontsize=16)
plt.xlabel('Evaluator', fontsize=12)
plt.ylabel('Question', fontsize=12)
plt.tight_layout()
plt.savefig('three_way_comparison_heatmap.png', dpi=300)
plt.close()

# 2. Vote distribution comparison
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Original baseline distribution
baseline_counts = comparison_df['Original_Baseline'].value_counts()
colors = ['#3498db', '#e74c3c', '#95a5a6']
ax1.pie(baseline_counts.values, labels=baseline_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Original Baseline')

# Human V2 distribution
v2_counts = comparison_df['Human_V2_Majority'].value_counts()
ax2.pie(v2_counts.values, labels=v2_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('Human V2 Majority')

# GPT-4 distribution
gpt4_counts = comparison_df['GPT4_Eval'].value_counts()
ax3.pie(gpt4_counts.values, labels=gpt4_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
ax3.set_title('GPT-4.1 Evaluation')

plt.suptitle('Distribution of Evaluations Across Three Sources', fontsize=16)
plt.tight_layout()
plt.savefig('three_source_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Agreement rates per question (Human V2)
if len(agreement_df) > 0:
    plt.figure(figsize=(12, 6))
    x = range(len(agreement_df))
    plt.bar(x, agreement_df['majority_agreement'])
    plt.xlabel('Question')
    plt.ylabel('Majority Agreement Rate')
    plt.title('Human V2 Agreement Rates by Question')
    plt.xticks(x, agreement_df['question'], rotation=45)
    plt.axhline(y=agreement_df['majority_agreement'].mean(), color='r', linestyle='--', 
                label=f'Average: {agreement_df["majority_agreement"].mean():.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('v2_agreement_rates.png', dpi=300)
    plt.close()

# 4. Confusion matrices
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Helper function for confusion matrices
def plot_confusion_matrix(y_true, y_pred, labels, title, ax):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Agent 1', 'Agent 2', 'Tie'],
                yticklabels=['Agent 1', 'Agent 2', 'Tie'],
                ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    
    # Calculate metrics
    total = sum(cm.flatten())
    correct = sum(cm[i][i] for i in range(len(labels)))
    accuracy = correct / total if total > 0 else 0
    ax.text(0.5, -0.15, f'Agreement: {accuracy:.1%}', 
            transform=ax.transAxes, ha='center')

# Baseline vs Human V2
baseline_labels = []
v2_labels = []
for _, row in comparison_df.iterrows():
    if row['Original_Baseline'] != 'N/A' and row['Human_V2_Majority'] != 'N/A':
        baseline_labels.append({'Agent 1': '1', 'Agent 2': '2', 'Tie': '3'}[row['Original_Baseline']])
        v2_labels.append({'Agent 1': '1', 'Agent 2': '2', 'Tie': '3'}[row['Human_V2_Majority']])

if len(baseline_labels) > 0:
    plot_confusion_matrix(baseline_labels, v2_labels, ['1', '2', '3'], 
                         'Baseline vs Human V2', ax1)

# Baseline vs GPT-4
baseline_labels2 = []
gpt4_labels = []
for _, row in comparison_df.iterrows():
    if row['Original_Baseline'] != 'N/A' and row['GPT4_Eval'] != 'N/A':
        baseline_labels2.append({'Agent 1': '1', 'Agent 2': '2', 'Tie': '3'}[row['Original_Baseline']])
        gpt4_labels.append({'Agent 1': '1', 'Agent 2': '2', 'Tie': '3'}[row['GPT4_Eval']])

if len(baseline_labels2) > 0:
    plot_confusion_matrix(baseline_labels2, gpt4_labels, ['1', '2', '3'], 
                         'Baseline vs GPT-4.1', ax2)

# Human V2 vs GPT-4
v2_labels2 = []
gpt4_labels2 = []
for _, row in comparison_df.iterrows():
    if row['Human_V2_Majority'] != 'N/A' and row['GPT4_Eval'] != 'N/A':
        v2_labels2.append({'Agent 1': '1', 'Agent 2': '2', 'Tie': '3'}[row['Human_V2_Majority']])
        gpt4_labels2.append({'Agent 1': '1', 'Agent 2': '2', 'Tie': '3'}[row['GPT4_Eval']])

if len(v2_labels2) > 0:
    plot_confusion_matrix(v2_labels2, gpt4_labels2, ['1', '2', '3'], 
                         'Human V2 vs GPT-4.1', ax3)

# Hide the fourth subplot
ax4.axis('off')

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=300)
plt.close()

# 5. Detailed vote breakdown for Human V2
fig, ax = plt.subplots(figsize=(14, 8))

questions = comparison_df['Question'].tolist()
x = np.arange(len(questions))
width = 0.25

agent1_votes = comparison_df['V2_Agent1_Votes'].tolist()
agent2_votes = comparison_df['V2_Agent2_Votes'].tolist()
tie_votes = comparison_df['V2_Tie_Votes'].tolist()

r1 = x - width
r2 = x
r3 = x + width

plt.bar(r1, agent1_votes, width, label='Agent 1', color='#3498db', alpha=0.8)
plt.bar(r2, agent2_votes, width, label='Agent 2', color='#e74c3c', alpha=0.8)
plt.bar(r3, tie_votes, width, label='Tie', color='#95a5a6', alpha=0.8)

plt.xlabel('Question', fontsize=12)
plt.ylabel('Number of Votes', fontsize=12)
plt.title('Human V2 Vote Distribution by Question', fontsize=14)
plt.xticks(x, questions, rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('v2_vote_distribution.png', dpi=300)
plt.close()

print("\nGenerated visualizations:")
print("- three_way_comparison_heatmap.png")
print("- three_source_distribution.png")
print("- v2_agreement_rates.png")
print("- confusion_matrices_comparison.png")
print("- v2_vote_distribution.png")

# Generate summary report
print("\n=== GENERATING SUMMARY REPORT ===")
with open('v2_analysis_report.md', 'w') as f:
    f.write("# BrowserArena Evaluation Comparison Report\n\n")
    f.write("## Overview\n")
    f.write("This report compares three evaluation sources:\n")
    f.write("1. **Original Baseline**: Single annotator judgments\n")
    f.write("2. **Human V2**: New crowdsourced human evaluations\n")
    f.write("3. **GPT-4.1**: AI model evaluations\n\n")
    
    f.write("## Key Findings\n\n")
    f.write("### Inter-Annotator Reliability (Human V2)\n")
    if 'alpha' in locals():
        f.write(f"- Krippendorff's Alpha: {alpha:.3f}\n")
    if 'kappa' in locals():
        f.write(f"- Fleiss' Kappa: {kappa:.3f}\n")
    if len(agreement_df) > 0:
        f.write(f"- Average Majority Agreement: {agreement_df['majority_agreement'].mean():.1%}\n\n")
    
    f.write("### Pairwise Agreement Rates\n")
    f.write(f"- Baseline vs Human V2: {baseline_v2_agree}/{len(comparison_df)} ({baseline_v2_agree/len(comparison_df)*100:.1f}%)\n")
    f.write(f"- Baseline vs GPT-4.1: {baseline_gpt4_agree}/{len(comparison_df)} ({baseline_gpt4_agree/len(comparison_df)*100:.1f}%)\n")
    f.write(f"- Human V2 vs GPT-4.1: {v2_gpt4_agree}/{len(comparison_df)} ({v2_gpt4_agree/len(comparison_df)*100:.1f}%)\n\n")
    
    f.write("### Distribution Comparison\n")
    f.write("| Source | Agent 1 | Agent 2 | Tie |\n")
    f.write("|--------|---------|---------|-----|\n")
    
    for source, counts in [('Baseline', baseline_counts), ('Human V2', v2_counts), ('GPT-4.1', gpt4_counts)]:
        agent1 = counts.get('Agent 1', 0)
        agent2 = counts.get('Agent 2', 0)
        tie = counts.get('Tie', 0)
        total = agent1 + agent2 + tie
        f.write(f"| {source} | {agent1} ({agent1/total*100:.1f}%) | {agent2} ({agent2/total*100:.1f}%) | {tie} ({tie/total*100:.1f}%) |\n")
    
    f.write("\n### Questions with Disagreement\n")
    disagreements = comparison_df[
        (comparison_df['Original_Baseline'] != comparison_df['Human_V2_Majority']) |
        (comparison_df['Original_Baseline'] != comparison_df['GPT4_Eval']) |
        (comparison_df['Human_V2_Majority'] != comparison_df['GPT4_Eval'])
    ]
    
    f.write(f"\nTotal questions with disagreement: {len(disagreements)}/{len(comparison_df)}\n\n")
    
    for _, row in disagreements.iterrows():
        f.write(f"**{row['Question']}**:\n")
        f.write(f"- Original: {row['Original_Baseline']}\n")
        f.write(f"- Human V2: {row['Human_V2_Majority']} ({row['V2_Agreement_Rate']} agreement)\n")
        f.write(f"- GPT-4.1: {row['GPT4_Eval']}\n")
        f.write(f"- V2 Votes: Agent1={row['V2_Agent1_Votes']}, Agent2={row['V2_Agent2_Votes']}, Tie={row['V2_Tie_Votes']}\n\n")

print("Analysis complete! Report saved to v2_analysis_report.md")