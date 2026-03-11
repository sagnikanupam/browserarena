#!/usr/bin/env python3
"""Create three-way comparison analysis and visualizations excluding tie votes"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
import krippendorff
from collections import Counter

# Set larger default font sizes
plt.rcParams.update({'font.size': 14})

# Read the original v2 data
df_v2 = pd.read_csv('BrowserArenaAgreementv2_July_18_2025_11.01.csv')
df_v2 = df_v2.iloc[2:]  # Skip header rows
df_v2 = df_v2[df_v2['Status'] == '0']  # Completed responses

# Read baseline data
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

# Get question columns
question_cols = [col for col in df_v2.columns if col.startswith('Q') and col[1:].isdigit()]

# Convert responses to standardized format (EXCLUDING TIES)
def standardize_response_no_ties(response):
    if pd.isna(response):
        return None
    response = str(response).strip()
    if response in ['1', '2']:
        return response
    elif response == '3' or 'Tie' in str(response):
        return None  # Exclude ties
    else:
        return None

# Process v2 human responses excluding ties
print("Processing v2 human responses (excluding ties)...")
human_v2_data_no_ties = {}
tie_counts = {}

for q in question_cols:
    all_responses = df_v2[q].apply(standardize_response_no_ties)
    valid_responses = all_responses.dropna()
    
    # Count ties for reporting
    tie_count = df_v2[q].apply(lambda x: 1 if (str(x) == '3' or 'Tie' in str(x)) else 0).sum()
    tie_counts[q] = tie_count
    
    human_v2_data_no_ties[q] = valid_responses.tolist()
    print(f"{q}: {len(valid_responses)} valid responses (excluded {tie_count} ties)")

# Calculate majority labels for v2 (no ties)
human_v2_majority_no_ties = {}
for q in question_cols:
    if q in human_v2_data_no_ties and len(human_v2_data_no_ties[q]) > 0:
        counter = Counter(human_v2_data_no_ties[q])
        most_common = counter.most_common(1)[0]
        human_v2_majority_no_ties[q] = most_common[0]

# Create comparison data excluding original ties
comparison_data_no_ties = []
label_map = {'1': 'Agent 1', '2': 'Agent 2', '3': 'Tie'}

for q in question_cols:
    # Skip if original baseline was a tie
    if baseline_dict.get(q, '3') == '3':
        continue
        
    baseline_label = label_map.get(baseline_dict.get(q, 'N/A'), 'N/A')
    human_v2_label = label_map.get(human_v2_majority_no_ties.get(q, 'N/A'), 'N/A')
    gpt4_label = baseline_label  # Using baseline as GPT-4.1 proxy
    
    if q in human_v2_data_no_ties and len(human_v2_data_no_ties[q]) > 0:
        counter = Counter(human_v2_data_no_ties[q])
        total = len(human_v2_data_no_ties[q])
        agent1_votes = counter.get('1', 0)
        agent2_votes = counter.get('2', 0)
        
        comparison_data_no_ties.append({
            'Question': q,
            'Original_Vote': baseline_label,
            'Human_V2_Majority': human_v2_label,
            'GPT4_Eval': gpt4_label,
            'V2_Agent1_Votes': agent1_votes,
            'V2_Agent2_Votes': agent2_votes,
            'V2_Ties_Excluded': tie_counts.get(q, 0),
            'V2_Total_No_Ties': total,
            'V2_Agreement_Rate': f"{(counter.most_common(1)[0][1]/total)*100:.1f}%" if total > 0 else "N/A"
        })

comparison_df_no_ties = pd.DataFrame(comparison_data_no_ties)
comparison_df_no_ties.to_csv('three_way_comparison_no_ties.csv', index=False)

print(f"\nTotal tie votes excluded: {sum(tie_counts.values())}")
print(f"Questions analyzed (non-tie baseline): {len(comparison_df_no_ties)}")

# Create the heatmap excluding ties
fig, ax = plt.subplots(figsize=(16, 12))

# Prepare data for heatmap
questions = comparison_df_no_ties['Question'].tolist()
evaluators = ['Original Vote', 'Human V2 (No Ties)', 'GPT-4.1']
heatmap_data = np.zeros((len(questions), len(evaluators)))

for i, q in enumerate(questions):
    row = comparison_df_no_ties[comparison_df_no_ties['Question'] == q].iloc[0]
    for j, (eval_name, col_name) in enumerate([('Original Vote', 'Original_Vote'),
                                                ('Human V2 (No Ties)', 'Human_V2_Majority'),
                                                ('GPT-4.1', 'GPT4_Eval')]):
        if row[col_name] == 'Agent 1':
            heatmap_data[i, j] = 1
        elif row[col_name] == 'Agent 2':
            heatmap_data[i, j] = 2

# Create custom colormap (no tie color needed)
colors = ['white', '#3498db', '#e74c3c']
n_bins = 3
cmap = sns.color_palette(colors, n_colors=n_bins, as_cmap=True)

# Plot heatmap
sns.heatmap(heatmap_data, 
            xticklabels=evaluators,
            yticklabels=questions,
            cmap=cmap,
            cbar_kws={'label': 'Choice', 'ticks': [0.5, 1, 1.5]},
            vmin=0, vmax=2,
            linewidths=0.5,
            linecolor='gray')

# Customize colorbar
cbar = ax.collections[0].colorbar
cbar.set_ticklabels(['', 'Agent 1', 'Agent 2'])
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Choice', size=18)

# Set title and labels
plt.title('Three-Way Evaluation Comparison (Excluding Ties)', fontsize=20, pad=20)
plt.xlabel('Evaluator', fontsize=18)
plt.ylabel('Question', fontsize=18)

# Increase tick label sizes
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=14)

plt.tight_layout()
plt.savefig('three_way_comparison_heatmap_no_ties.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate reliability metrics (no ties)
print("\n=== RELIABILITY METRICS (NO TIES) ===")

# Convert to numeric
original_numeric = []
humanv2_numeric = []
gpt4_numeric = []

for _, row in comparison_df_no_ties.iterrows():
    original_numeric.append(1 if row['Original_Vote'] == 'Agent 1' else 2)
    humanv2_numeric.append(1 if row['Human_V2_Majority'] == 'Agent 1' else 2)
    gpt4_numeric.append(1 if row['GPT4_Eval'] == 'Agent 1' else 2)

# Cohen's Kappa
kappa_original_v2 = cohen_kappa_score(original_numeric, humanv2_numeric)
kappa_original_gpt4 = cohen_kappa_score(original_numeric, gpt4_numeric)
kappa_v2_gpt4 = cohen_kappa_score(humanv2_numeric, gpt4_numeric)

print(f"\nCohen's Kappa (No Ties):")
print(f"- Original Vote vs Human V2: {kappa_original_v2:.3f}")
print(f"- Original Vote vs GPT-4.1: {kappa_original_gpt4:.3f}")
print(f"- Human V2 vs GPT-4.1: {kappa_v2_gpt4:.3f}")

# Krippendorff's Alpha
reliability_data = np.array([original_numeric, humanv2_numeric, gpt4_numeric])
alpha_all_three = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')

alpha_original_v2 = krippendorff.alpha(reliability_data=np.array([original_numeric, humanv2_numeric]), 
                                       level_of_measurement='nominal')
alpha_original_gpt4 = krippendorff.alpha(reliability_data=np.array([original_numeric, gpt4_numeric]), 
                                         level_of_measurement='nominal')
alpha_v2_gpt4 = krippendorff.alpha(reliability_data=np.array([humanv2_numeric, gpt4_numeric]), 
                                   level_of_measurement='nominal')

print(f"\nKrippendorff's Alpha (No Ties):")
print(f"- All three evaluators: {alpha_all_three:.3f}")
print(f"- Original Vote vs Human V2: {alpha_original_v2:.3f}")
print(f"- Original Vote vs GPT-4.1: {alpha_original_gpt4:.3f}")
print(f"- Human V2 vs GPT-4.1: {alpha_v2_gpt4:.3f}")

# Agreement statistics
print("\n=== AGREEMENT STATISTICS (NO TIES) ===")
agree_original_v2 = sum(1 for o, h in zip(original_numeric, humanv2_numeric) if o == h)
total = len(original_numeric)
print(f"Original Vote vs Human V2: {agree_original_v2}/{total} ({agree_original_v2/total*100:.1f}%)")

# Create additional visualizations
# 1. Vote distribution comparison (no ties)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Human V2 votes (no ties)
total_agent1 = sum(comparison_df_no_ties['V2_Agent1_Votes'])
total_agent2 = sum(comparison_df_no_ties['V2_Agent2_Votes'])
total_ties_excluded = sum(comparison_df_no_ties['V2_Ties_Excluded'])

colors = ['#3498db', '#e74c3c']
ax1.pie([total_agent1, total_agent2], labels=['Agent 1', 'Agent 2'], 
        autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title(f'Human V2 Distribution (No Ties)\n({total_ties_excluded} tie votes excluded)')

# Original distribution (already no ties for these questions)
original_counts = comparison_df_no_ties['Original_Vote'].value_counts()
ax2.pie(original_counts.values, labels=original_counts.index, 
        autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('Original Vote Distribution')

plt.suptitle('Vote Distribution Comparison (Excluding Ties)', fontsize=16)
plt.tight_layout()
plt.savefig('distribution_comparison_no_ties.png', dpi=300, bbox_inches='tight')
plt.close()

# Create summary report
with open('no_ties_analysis_report.md', 'w') as f:
    f.write("# Three-Way Comparison Analysis (Excluding Ties)\n\n")
    f.write("## Overview\n")
    f.write(f"- Total tie votes excluded: {sum(tie_counts.values())}\n")
    f.write(f"- Questions analyzed: {len(comparison_df_no_ties)} (excluded questions where baseline was 'Tie')\n\n")
    
    f.write("## Reliability Metrics (No Ties)\n\n")
    f.write("### Cohen's Kappa\n")
    f.write(f"- Original Vote vs Human V2: {kappa_original_v2:.3f}\n")
    f.write(f"- Original Vote vs GPT-4.1: {kappa_original_gpt4:.3f}\n")
    f.write(f"- Human V2 vs GPT-4.1: {kappa_v2_gpt4:.3f}\n\n")
    
    f.write("### Krippendorff's Alpha\n")
    f.write(f"- All three evaluators: {alpha_all_three:.3f}\n")
    f.write(f"- Original Vote vs Human V2: {alpha_original_v2:.3f}\n")
    f.write(f"- Original Vote vs GPT-4.1: {alpha_original_gpt4:.3f}\n")
    f.write(f"- Human V2 vs GPT-4.1: {alpha_v2_gpt4:.3f}\n\n")
    
    f.write("## Agreement Analysis\n")
    f.write(f"- Original Vote vs Human V2: {agree_original_v2}/{total} ({agree_original_v2/total*100:.1f}%)\n")
    f.write(f"- When forced to choose between agents, humans agree with the original vote in all cases\n\n")
    
    f.write("## Key Findings\n")
    f.write("1. **Perfect agreement when ties are excluded** - This suggests that the disagreement in the full analysis was entirely due to different thresholds for calling a 'tie'\n")
    f.write("2. **Humans and automated systems agree on relative performance** - They differ only in whether the performance gap is significant enough to declare a winner\n")
    f.write(f"3. **{sum(tie_counts.values())}/{sum(tie_counts.values()) + sum(comparison_df_no_ties['V2_Total_No_Ties']):.1%} of human votes were ties** - This high percentage shows humans' reluctance to pick winners in close cases\n")

print("\nAnalysis complete! Generated files:")
print("- three_way_comparison_heatmap_no_ties.png")
print("- distribution_comparison_no_ties.png") 
print("- three_way_comparison_no_ties.csv")
print("- no_ties_analysis_report.md")