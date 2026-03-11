#!/usr/bin/env python3
"""Comprehensive analysis of BrowserArena survey data"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import warnings
warnings.filterwarnings('ignore')

# Read the survey data
print("Reading survey data...")
df = pd.read_csv('BrowserArenaAgreement_July_11_2025_11.03.csv')

# Get question columns (Q1, Q2, etc.)
question_cols = [col for col in df.columns if col.startswith('Q') and col[1:].isdigit()]
print(f"Found {len(question_cols)} questions: {question_cols}")

# Read baseline data
baseline_df = pd.read_csv('baseline.csv')
baseline_dict = {}
for _, row in baseline_df.iterrows():
    q_num = row['question']
    if row['winner'] == 'Agent 1':
        baseline_dict[q_num] = 'A'
    elif row['winner'] == 'Agent 2':
        baseline_dict[q_num] = 'B'
    elif row['winner'] == 'Tie':
        baseline_dict[q_num] = 'Tie'

print(f"\nBaseline labels: {baseline_dict}")

# Convert responses to standardized format
def standardize_response(response):
    if pd.isna(response):
        return None
    response = str(response).strip().upper()
    if 'AGENT 1' in response or response == 'A':
        return 'A'
    elif 'AGENT 2' in response or response == 'B':
        return 'B'
    elif 'TIE' in response:
        return 'Tie'
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
            
            print(f"{q}: {majority_labels[q]} ({majority_pct:.1f}% agreement)")
            print(f"   Distribution: {dict(counter)}")

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

agreement_df = pd.DataFrame(agreement_stats)
print(f"Average pairwise agreement: {agreement_df['pairwise_agreement'].mean():.3f}")
print(f"Average majority agreement: {agreement_df['majority_agreement'].mean():.3f}")

# Calculate Krippendorff's alpha
print("\n=== KRIPPENDORFF'S ALPHA ===")
# Prepare data for Krippendorff
# Create a matrix where rows are items (questions) and columns are raters
all_questions = sorted([q for q in question_cols if q in processed_data])
max_raters = max(len(processed_data[q]) for q in all_questions)

# Create reliability data matrix
reliability_data = []
for q_idx, q in enumerate(all_questions):
    row = []
    for rater_idx in range(max_raters):
        if rater_idx < len(processed_data[q]):
            response = processed_data[q][rater_idx]
            # Convert to numeric: A=1, B=2, Tie=3
            if response == 'A':
                row.append(1)
            elif response == 'B':
                row.append(2)
            elif response == 'Tie':
                row.append(3)
        else:
            row.append(np.nan)
    reliability_data.append(row)

reliability_data = np.array(reliability_data).T  # Transpose so raters are rows

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
        counts = {'A': 0, 'B': 0, 'Tie': 0}
        for r in responses:
            if r in counts:
                counts[r] += 1
        fleiss_data.append([counts['A'], counts['B'], counts['Tie']])

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
    labels = ['A', 'B', 'Tie']
    cm = confusion_matrix(original_labels, majority_pred, labels=labels)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("Original\\Majority", end="")
    for label in labels:
        print(f"\t{label}", end="")
    print()
    
    for i, label in enumerate(labels):
        print(f"{label}", end="")
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
    'distribution': []
}

for q in all_questions:
    results['question'].append(q)
    results['original_label'].append(baseline_dict.get(q, 'N/A'))
    results['majority_label'].append(majority_labels.get(q, 'N/A'))
    
    if q in processed_data:
        counter = Counter(processed_data[q])
        total = len(processed_data[q])
        results['n_responses'].append(total)
        
        if q in majority_labels:
            majority_count = counter[majority_labels[q]]
            results['majority_percentage'].append(f"{(majority_count/total)*100:.1f}%")
        else:
            results['majority_percentage'].append('N/A')
        
        results['distribution'].append(str(dict(counter)))
    else:
        results['n_responses'].append(0)
        results['majority_percentage'].append('N/A')
        results['distribution'].append('{}')

results_df = pd.DataFrame(results)
results_df.to_csv('agreement_analysis_results.csv', index=False)
print("Saved detailed results to agreement_analysis_results.csv")

# Generate visualizations
print("\n=== GENERATING VISUALIZATIONS ===")

# 1. Agreement rates per question
plt.figure(figsize=(12, 6))
if len(agreement_df) > 0:
    x = range(len(agreement_df))
    plt.bar(x, agreement_df['majority_agreement'])
    plt.xlabel('Question')
    plt.ylabel('Majority Agreement Rate')
    plt.title('Agreement Rates by Question')
    plt.xticks(x, agreement_df['question'], rotation=45)
    plt.axhline(y=agreement_df['majority_agreement'].mean(), color='r', linestyle='--', label='Average')
    plt.legend()
    plt.tight_layout()
    plt.savefig('agreement_rates_by_question.png', dpi=300)
    plt.close()

# 2. Distribution of responses
plt.figure(figsize=(10, 8))
response_counts = {'A': 0, 'B': 0, 'Tie': 0}
for q in all_questions:
    if q in processed_data:
        for r in processed_data[q]:
            if r in response_counts:
                response_counts[r] += 1

plt.pie(response_counts.values(), labels=response_counts.keys(), autopct='%1.1f%%')
plt.title('Overall Distribution of Responses')
plt.savefig('response_distribution.png', dpi=300)
plt.close()

# 3. Heatmap of confusion matrix
if len(original_labels) > 0:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix: Original vs Majority Labels')
    plt.ylabel('Original Label')
    plt.xlabel('Majority Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_heatmap.png', dpi=300)
    plt.close()

print("\nAnalysis complete! Generated files:")
print("- agreement_analysis_results.csv")
print("- agreement_rates_by_question.png")
print("- response_distribution.png")
print("- confusion_matrix_heatmap.png")