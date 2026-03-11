#!/usr/bin/env python3
"""Comprehensive analysis of BrowserArena survey data"""

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
print("Reading survey data...")
df = pd.read_csv('BrowserArenaAgreement_July_11_2025_11.03.csv')

# Filter out rows that don't have actual responses (header rows, etc.)
# Skip first two rows (headers) and only keep rows with numeric responses
df = df.iloc[2:]  # Skip header rows
df = df[df['Status'] == '0']  # Status 0 seems to be completed responses

# Get question columns (Q1, Q2, etc.)
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

print(f"\nBaseline labels: {baseline_dict}")

# Convert responses to standardized format
def standardize_response(response):
    if pd.isna(response):
        return None
    response = str(response).strip()
    # Look for numeric responses (1, 2, 3)
    if response in ['1', '2', '3']:
        return response
    # Also check for "Agent 1", "Agent 2", "Tie" in the text
    elif 'Agent 1' in response and 'Agent 2' not in response:
        return '1'
    elif 'Agent 2' in response and 'Agent 1' not in response:
        return '2'
    elif 'Tie' in response:
        return '3'
    else:
        return None

# Process all responses
print("\nProcessing responses...")
processed_data = {}
for q in question_cols:
    responses = df[q].apply(standardize_response)
    valid_responses = responses.dropna()
    processed_data[q] = valid_responses.tolist()
    print(f"{q}: {len(valid_responses)} valid responses")

# Calculate majority labels
print("\n=== MAJORITY LABELS ===")
majority_labels = {}
for q in question_cols:
    if q in processed_data and len(processed_data[q]) > 0:
        counter = Counter(processed_data[q])
        total = sum(counter.values())
        
        # Get majority label
        if total > 0:
            most_common = counter.most_common(1)[0]
            majority_labels[q] = most_common[0]
            majority_pct = (most_common[1] / total) * 100
            
            label_map = {'1': 'Agent 1', '2': 'Agent 2', '3': 'Tie'}
            print(f"{q}: {label_map.get(majority_labels[q], majority_labels[q])} ({majority_pct:.1f}% agreement)")
            dist_str = {label_map.get(k, k): v for k, v in counter.items()}
            print(f"   Distribution: {dist_str}")

# Calculate agreement rates
print("\n=== AGREEMENT RATES ===")
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
else:
    agreement_df = pd.DataFrame()

# Calculate Krippendorff's alpha
print("\n=== KRIPPENDORFF'S ALPHA ===")
# Prepare data for Krippendorff
all_questions = sorted([q for q in question_cols if q in processed_data and len(processed_data[q]) > 0])

if len(all_questions) > 0:
    # Create reliability data matrix
    reliability_data = []
    max_raters = max(len(processed_data[q]) for q in all_questions)
    
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
    
    # Calculate Krippendorff's alpha
    try:
        alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
        print(f"Krippendorff's alpha: {alpha:.3f}")
    except Exception as e:
        print(f"Error calculating Krippendorff's alpha: {e}")

# Calculate Fleiss' Kappa
print("\n=== FLEISS' KAPPA ===")
# Prepare data for Fleiss' Kappa
fleiss_data = []
for q in all_questions:
    if q in processed_data:
        responses = processed_data[q]
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

# Confusion matrix: Original vs Majority
print("\n=== CONFUSION MATRIX: ORIGINAL VS MAJORITY ===")
original_labels = []
majority_pred = []

for q in all_questions:
    if q in baseline_dict and q in majority_labels:
        original_labels.append(baseline_dict[q])
        majority_pred.append(majority_labels[q])

if len(original_labels) > 0:
    # Calculate confusion matrix
    labels = ['1', '2', '3']
    label_names = ['Agent 1', 'Agent 2', 'Tie']
    cm = confusion_matrix(original_labels, majority_pred, labels=labels)
    
    # Print confusion matrix
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
    
    # Cohen's Kappa between original and majority
    try:
        kappa = cohen_kappa_score(original_labels, majority_pred)
        print(f"Cohen's Kappa (original vs majority): {kappa:.3f}")
    except:
        pass

# Save detailed results
print("\n=== SAVING RESULTS ===")
results = {
    'question': [],
    'original_label': [],
    'majority_label': [],
    'n_responses': [],
    'majority_percentage': [],
    'agent_1_votes': [],
    'agent_2_votes': [],
    'tie_votes': []
}

label_map = {'1': 'Agent 1', '2': 'Agent 2', '3': 'Tie'}

for q in all_questions:
    results['question'].append(q)
    results['original_label'].append(label_map.get(baseline_dict.get(q, 'N/A'), 'N/A'))
    results['majority_label'].append(label_map.get(majority_labels.get(q, 'N/A'), 'N/A'))
    
    if q in processed_data:
        counter = Counter(processed_data[q])
        total = len(processed_data[q])
        results['n_responses'].append(total)
        
        results['agent_1_votes'].append(counter.get('1', 0))
        results['agent_2_votes'].append(counter.get('2', 0))
        results['tie_votes'].append(counter.get('3', 0))
        
        if q in majority_labels:
            majority_count = counter[majority_labels[q]]
            results['majority_percentage'].append(f"{(majority_count/total)*100:.1f}%")
        else:
            results['majority_percentage'].append('N/A')
    else:
        results['n_responses'].append(0)
        results['agent_1_votes'].append(0)
        results['agent_2_votes'].append(0)
        results['tie_votes'].append(0)
        results['majority_percentage'].append('N/A')

results_df = pd.DataFrame(results)
results_df.to_csv('agreement_analysis_results.csv', index=False)
print("Saved detailed results to agreement_analysis_results.csv")

# Generate visualizations
print("\n=== GENERATING VISUALIZATIONS ===")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 1. Agreement rates per question
if len(agreement_df) > 0:
    plt.figure(figsize=(12, 6))
    x = range(len(agreement_df))
    plt.bar(x, agreement_df['majority_agreement'])
    plt.xlabel('Question')
    plt.ylabel('Majority Agreement Rate')
    plt.title('Agreement Rates by Question')
    plt.xticks(x, agreement_df['question'], rotation=45)
    plt.axhline(y=agreement_df['majority_agreement'].mean(), color='r', linestyle='--', label=f'Average: {agreement_df["majority_agreement"].mean():.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('agreement_rates_by_question.png', dpi=300)
    plt.close()

# 2. Distribution of responses
plt.figure(figsize=(10, 8))
response_counts = {'Agent 1': 0, 'Agent 2': 0, 'Tie': 0}
for q in all_questions:
    if q in processed_data:
        counter = Counter(processed_data[q])
        response_counts['Agent 1'] += counter.get('1', 0)
        response_counts['Agent 2'] += counter.get('2', 0)
        response_counts['Tie'] += counter.get('3', 0)

colors = ['#3498db', '#e74c3c', '#95a5a6']
plt.pie(response_counts.values(), labels=response_counts.keys(), autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Overall Distribution of Responses')
plt.savefig('response_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Heatmap of confusion matrix
if len(original_labels) > 0:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix: Original vs Majority Labels')
    plt.ylabel('Original Label')
    plt.xlabel('Majority Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_heatmap.png', dpi=300)
    plt.close()

# 4. Vote distribution per question
if len(results_df) > 0:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    questions = results_df['question']
    x = np.arange(len(questions))
    width = 0.25
    
    ax.bar(x - width, results_df['agent_1_votes'], width, label='Agent 1', color='#3498db')
    ax.bar(x, results_df['agent_2_votes'], width, label='Agent 2', color='#e74c3c')
    ax.bar(x + width, results_df['tie_votes'], width, label='Tie', color='#95a5a6')
    
    ax.set_xlabel('Question')
    ax.set_ylabel('Number of Votes')
    ax.set_title('Vote Distribution by Question')
    ax.set_xticks(x)
    ax.set_xticklabels(questions, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('vote_distribution_by_question.png', dpi=300)
    plt.close()

print("\nAnalysis complete! Generated files:")
print("- agreement_analysis_results.csv")
print("- agreement_rates_by_question.png")
print("- response_distribution.png")
print("- confusion_matrix_heatmap.png")
print("- vote_distribution_by_question.png")

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")
if len(results_df) > 0:
    total_votes = results_df['n_responses'].sum()
    total_questions = len(results_df)
    avg_responses_per_q = results_df['n_responses'].mean()
    
    print(f"Total questions analyzed: {total_questions}")
    print(f"Total votes collected: {total_votes}")
    print(f"Average responses per question: {avg_responses_per_q:.1f}")
    
    # Agreement with baseline
    agreements = (results_df['original_label'] == results_df['majority_label']).sum()
    print(f"\nQuestions where majority agrees with baseline: {agreements}/{total_questions} ({agreements/total_questions*100:.1f}%)")
    
    # Questions with disagreement
    disagreements = results_df[results_df['original_label'] != results_df['majority_label']]
    if len(disagreements) > 0:
        print("\nQuestions with disagreement:")
        for _, row in disagreements.iterrows():
            print(f"  {row['question']}: Original={row['original_label']}, Majority={row['majority_label']} ({row['majority_percentage']} agreement)")