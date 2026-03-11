#!/usr/bin/env python3
"""Analysis excluding tie votes - only Agent 1 vs Agent 2"""

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

# Read the survey data
print("ANALYSIS EXCLUDING TIE VOTES")
print("=" * 80)
print("Reading survey data...")
df = pd.read_csv('BrowserArenaAgreement_July_11_2025_11.03.csv')

# Filter to valid responses
df = df.iloc[2:]  # Skip header rows
df = df[df['Status'] == '0']  # Status 0 seems to be completed responses

# Get question columns
question_cols = [col for col in df.columns if col.startswith('Q') and col[1:].isdigit()]
print(f"Found {len(question_cols)} questions: {question_cols}")
print(f"Found {len(df)} valid responses")

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

# Convert responses to standardized format (EXCLUDING TIES)
def standardize_response_no_ties(response):
    if pd.isna(response):
        return None
    response = str(response).strip()
    # Only accept Agent 1 or Agent 2 responses
    if response in ['1', '2']:
        return response
    elif 'Agent 1' in response and 'Agent 2' not in response and 'Tie' not in response:
        return '1'
    elif 'Agent 2' in response and 'Agent 1' not in response and 'Tie' not in response:
        return '2'
    else:
        return None  # Exclude ties and ambiguous responses

# Process all responses excluding ties
print("\nProcessing responses (excluding ties)...")
processed_data = {}
total_excluded = 0
for q in question_cols:
    all_responses = df[q].apply(lambda x: standardize_response_no_ties(x) if x != '3' and 'Tie' not in str(x) else None)
    valid_responses = all_responses.dropna()
    
    # Count excluded tie votes
    tie_count = df[q].apply(lambda x: 1 if (str(x) == '3' or 'Tie' in str(x)) else 0).sum()
    total_excluded += tie_count
    
    processed_data[q] = valid_responses.tolist()
    print(f"{q}: {len(valid_responses)} valid responses (excluded {tie_count} tie votes)")

print(f"\nTotal tie votes excluded: {total_excluded}")

# Calculate majority labels (no ties possible now)
print("\n=== MAJORITY LABELS (NO TIES) ===")
majority_labels = {}
for q in question_cols:
    if q in processed_data and len(processed_data[q]) > 0:
        counter = Counter(processed_data[q])
        total = sum(counter.values())
        
        if total > 0:
            most_common = counter.most_common(1)[0]
            majority_labels[q] = most_common[0]
            majority_pct = (most_common[1] / total) * 100
            
            label_map = {'1': 'Agent 1', '2': 'Agent 2'}
            print(f"{q}: {label_map.get(majority_labels[q], majority_labels[q])} ({majority_pct:.1f}% agreement)")
            
            # Show vote counts
            agent1_votes = counter.get('1', 0)
            agent2_votes = counter.get('2', 0)
            print(f"   Votes: Agent 1: {agent1_votes}, Agent 2: {agent2_votes}")

# Calculate agreement rates
print("\n=== AGREEMENT RATES (NO TIES) ===")
agreement_stats = []
for q in question_cols:
    if q in processed_data and len(processed_data[q]) > 1:
        responses = processed_data[q]
        counter = Counter(responses)
        total = len(responses)
        
        if total > 0:
            # Pairwise agreement rate
            pairs_agree = 0
            pairs_total = 0
            for i in range(len(responses)):
                for j in range(i+1, len(responses)):
                    pairs_total += 1
                    if responses[i] == responses[j]:
                        pairs_agree += 1
            
            pairwise_agreement = pairs_agree / pairs_total if pairs_total > 0 else 0
            
            # Majority agreement rate
            majority_count = counter.most_common(1)[0][1]
            majority_agreement = majority_count / total
            
            agreement_stats.append({
                'question': q,
                'n_raters': total,
                'pairwise_agreement': pairwise_agreement,
                'majority_agreement': majority_agreement
            })

if len(agreement_stats) > 0:
    agreement_df = pd.DataFrame(agreement_stats)
    print(f"Average pairwise agreement: {agreement_df['pairwise_agreement'].mean():.3f}")
    print(f"Average majority agreement: {agreement_df['majority_agreement'].mean():.3f}")

# Calculate Krippendorff's alpha (no ties)
print("\n=== KRIPPENDORFF'S ALPHA (NO TIES) ===")
all_questions = sorted([q for q in question_cols if q in processed_data and len(processed_data[q]) > 0])

if len(all_questions) > 0:
    reliability_data = []
    max_raters = max(len(processed_data[q]) for q in all_questions if len(processed_data[q]) > 0)
    
    for rater_idx in range(max_raters):
        row = []
        for q in all_questions:
            if rater_idx < len(processed_data[q]):
                response = processed_data[q][rater_idx]
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

# Confusion matrix: Original vs Majority (excluding ties)
print("\n=== CONFUSION MATRIX: ORIGINAL VS MAJORITY (NO TIES) ===")
original_labels = []
majority_pred = []

# Only include questions where original wasn't a tie
for q in all_questions:
    if q in baseline_dict and q in majority_labels and baseline_dict[q] != '3':
        original_labels.append(baseline_dict[q])
        majority_pred.append(majority_labels[q])

if len(original_labels) > 0:
    labels = ['1', '2']
    label_names = ['Agent 1', 'Agent 2']
    cm = confusion_matrix(original_labels, majority_pred, labels=labels)
    
    print("\nConfusion Matrix:")
    print("Original\\Majority", end="")
    for label_name in label_names:
        print(f"\t{label_name}", end="")
    print()
    
    for i, label_name in enumerate(label_names):
        print(f"{label_name}", end="")
        for j in range(len(labels)):
            print(f"\t{cm[i][j]}", end="")
        print()
    
    # Calculate agreement
    total = sum(cm.flatten())
    correct = sum(cm[i][i] for i in range(len(labels)))
    accuracy = correct / total if total > 0 else 0
    print(f"\nAgreement between original and majority: {accuracy:.1%} ({correct}/{total})")
    
    # Cohen's Kappa
    try:
        kappa = cohen_kappa_score(original_labels, majority_pred)
        print(f"Cohen's Kappa (original vs majority): {kappa:.3f}")
    except:
        pass

# Save results
print("\n=== SAVING RESULTS (NO TIES) ===")
results = {
    'question': [],
    'original_label': [],
    'majority_label_no_ties': [],
    'n_responses_no_ties': [],
    'agent_1_votes': [],
    'agent_2_votes': [],
    'ties_excluded': [],
    'majority_percentage': []
}

# Get tie counts from original data
original_df = pd.read_csv('agreement_analysis_results.csv')
tie_counts = {row['question']: row['tie_votes'] for _, row in original_df.iterrows()}

label_map = {'1': 'Agent 1', '2': 'Agent 2', '3': 'Tie'}

for q in all_questions:
    results['question'].append(q)
    results['original_label'].append(label_map.get(baseline_dict.get(q, 'N/A'), 'N/A'))
    
    if q in processed_data and len(processed_data[q]) > 0:
        counter = Counter(processed_data[q])
        total = len(processed_data[q])
        
        results['majority_label_no_ties'].append(label_map.get(majority_labels.get(q, 'N/A'), 'N/A'))
        results['n_responses_no_ties'].append(total)
        results['agent_1_votes'].append(counter.get('1', 0))
        results['agent_2_votes'].append(counter.get('2', 0))
        results['ties_excluded'].append(tie_counts.get(q, 0))
        
        if q in majority_labels:
            majority_count = counter[majority_labels[q]]
            results['majority_percentage'].append(f"{(majority_count/total)*100:.1f}%")
        else:
            results['majority_percentage'].append('N/A')
    else:
        results['majority_label_no_ties'].append('N/A')
        results['n_responses_no_ties'].append(0)
        results['agent_1_votes'].append(0)
        results['agent_2_votes'].append(0)
        results['ties_excluded'].append(tie_counts.get(q, 0))
        results['majority_percentage'].append('N/A')

results_df = pd.DataFrame(results)
results_df.to_csv('agreement_analysis_no_ties.csv', index=False)
print("Saved detailed results to agreement_analysis_no_ties.csv")

# Visualizations
print("\n=== GENERATING VISUALIZATIONS (NO TIES) ===")

# 1. Vote distribution bar chart
if len(results_df) > 0 and results_df['n_responses_no_ties'].sum() > 0:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    questions = results_df['question']
    x = np.arange(len(questions))
    width = 0.35
    
    agent1_votes = results_df['agent_1_votes']
    agent2_votes = results_df['agent_2_votes']
    
    bars1 = ax.bar(x - width/2, agent1_votes, width, label='Agent 1', color='#3498db')
    bars2 = ax.bar(x + width/2, agent2_votes, width, label='Agent 2', color='#e74c3c')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
    
    ax.set_xlabel('Question')
    ax.set_ylabel('Number of Votes (Ties Excluded)')
    ax.set_title('Vote Distribution by Question (No Ties)')
    ax.set_xticks(x)
    ax.set_xticklabels(questions, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vote_distribution_no_ties.png', dpi=300)
    plt.close()

# 2. Comparison chart: with ties vs without ties
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Original data with ties
original_df = pd.read_csv('agreement_analysis_results.csv')
total_with_ties = original_df[['agent_1_votes', 'agent_2_votes', 'tie_votes']].sum()

# Data without ties
total_no_ties = results_df[['agent_1_votes', 'agent_2_votes']].sum()

# Pie chart with ties
colors = ['#3498db', '#e74c3c', '#95a5a6']
ax1.pie(total_with_ties.values, labels=['Agent 1', 'Agent 2', 'Tie'], 
        autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Overall Vote Distribution (Including Ties)')

# Pie chart without ties
ax2.pie(total_no_ties.values, labels=['Agent 1', 'Agent 2'], 
        autopct='%1.1f%%', colors=colors[:2], startangle=90)
ax2.set_title('Overall Vote Distribution (Excluding Ties)')

plt.tight_layout()
plt.savefig('distribution_comparison_no_ties.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis complete! Generated files:")
print("- agreement_analysis_no_ties.csv")
print("- vote_distribution_no_ties.png")
print("- distribution_comparison_no_ties.png")

# Summary statistics
print("\n=== SUMMARY STATISTICS (NO TIES) ===")
if len(results_df) > 0:
    total_votes = results_df['n_responses_no_ties'].sum()
    total_excluded = results_df['ties_excluded'].sum()
    
    print(f"Total votes analyzed (excluding ties): {total_votes}")
    print(f"Total tie votes excluded: {total_excluded}")
    print(f"Percentage of votes that were ties: {total_excluded/(total_votes+total_excluded)*100:.1f}%")
    
    # Questions where majority changed after excluding ties
    print("\nQuestions where excluding ties changed the outcome:")
    for i, row in results_df.iterrows():
        q = row['question']
        if q in original_df['question'].values:
            orig_row = original_df[original_df['question'] == q].iloc[0]
            if orig_row['majority_label'] != row['majority_label_no_ties'] and row['n_responses_no_ties'] > 0:
                print(f"  {q}: {orig_row['majority_label']} → {row['majority_label_no_ties']}")