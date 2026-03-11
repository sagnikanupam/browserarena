#!/usr/bin/env python3
"""Calculate inter-annotator agreement for V2 survey data"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Read the V2 survey data
print("Reading V2 survey data...")
df = pd.read_csv('BrowserArenaAgreementv2_July_18_2025_11.01.csv')

# Filter out non-response rows
df = df.iloc[2:]  # Skip header rows
df = df[df['Status'] == '0']  # Completed responses only

# Get question columns
question_cols = [col for col in df.columns if col.startswith('Q') and col[1:].isdigit()]
print(f"Found {len(question_cols)} questions: {question_cols}")
print(f"Found {len(df)} valid responses")

# Convert responses to standardized format
def standardize_response(response):
    if pd.isna(response):
        return None
    response = str(response).strip()
    if response in ['1', '2', '3']:
        return int(response)
    return None

# Process all responses
processed_data = {}
for q in question_cols:
    responses = df[q].apply(standardize_response)
    valid_responses = responses.dropna().tolist()
    if len(valid_responses) > 0:
        processed_data[q] = valid_responses

print(f"\nProcessed {len(processed_data)} questions with valid responses")

# 1. Per-question inter-annotator agreement
print("\n=== PER-QUESTION INTER-ANNOTATOR AGREEMENT ===")
print("-" * 80)
print(f"{'Question':<10} {'N Raters':<10} {'Majority':<15} {'Pairwise':<12} {'Fleiss K':<12} {'Kripp α':<12}")
print("-" * 80)

per_question_results = []

for q in sorted(processed_data.keys()):
    responses = processed_data[q]
    n_raters = len(responses)
    
    # Calculate pairwise agreement
    pairs_agree = 0
    pairs_total = 0
    for i in range(len(responses)):
        for j in range(i+1, len(responses)):
            pairs_total += 1
            if responses[i] == responses[j]:
                pairs_agree += 1
    
    pairwise_agreement = pairs_agree / pairs_total if pairs_total > 0 else 0
    
    # Calculate majority agreement
    counter = Counter(responses)
    majority_count = counter.most_common(1)[0][1]
    majority_agreement = majority_count / n_raters
    
    # Calculate Fleiss' Kappa for this question
    # Create table: rows = items (just 1 for single question), cols = categories
    fleiss_table = np.zeros((1, 3))  # 3 categories: 1, 2, 3
    for r in responses:
        fleiss_table[0, r-1] += 1
    
    try:
        fleiss_k = fleiss_kappa(fleiss_table, method='fleiss')
    except:
        fleiss_k = np.nan
    
    # Calculate Krippendorff's alpha for this question
    # Create reliability data matrix for single question
    reliability_data = np.array([responses])
    try:
        alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
    except:
        alpha = np.nan
    
    per_question_results.append({
        'question': q,
        'n_raters': n_raters,
        'majority_agreement': majority_agreement,
        'pairwise_agreement': pairwise_agreement,
        'fleiss_kappa': fleiss_k,
        'krippendorff_alpha': alpha
    })
    
    # Get majority label
    label_map = {1: 'Agent 1', 2: 'Agent 2', 3: 'Tie'}
    majority_label = label_map[counter.most_common(1)[0][0]]
    
    print(f"{q:<10} {n_raters:<10} {majority_label:<15} {pairwise_agreement:<12.3f} {fleiss_k:<12.3f} {alpha:<12.3f}")

# 2. Overall inter-annotator agreement across all questions
print("\n\n=== OVERALL INTER-ANNOTATOR AGREEMENT ===")

# Prepare data for overall Krippendorff's alpha
# Create matrix: rows = raters, cols = questions
max_raters = max(len(processed_data[q]) for q in processed_data)
reliability_matrix = []

for rater_idx in range(max_raters):
    rater_row = []
    for q in sorted(processed_data.keys()):
        if rater_idx < len(processed_data[q]):
            rater_row.append(processed_data[q][rater_idx])
        else:
            rater_row.append(np.nan)
    reliability_matrix.append(rater_row)

reliability_matrix = np.array(reliability_matrix)

# Calculate overall Krippendorff's alpha
overall_alpha = krippendorff.alpha(reliability_data=reliability_matrix, level_of_measurement='nominal')
print(f"\nOverall Krippendorff's Alpha: {overall_alpha:.3f}")

# Prepare data for overall Fleiss' Kappa
fleiss_data = []
for q in sorted(processed_data.keys()):
    responses = processed_data[q]
    counts = [0, 0, 0]  # For categories 1, 2, 3
    for r in responses:
        counts[r-1] += 1
    fleiss_data.append(counts)

fleiss_data = np.array(fleiss_data)
overall_fleiss = fleiss_kappa(fleiss_data, method='fleiss')
print(f"Overall Fleiss' Kappa: {overall_fleiss:.3f}")

# Calculate average pairwise agreement
avg_pairwise = np.mean([r['pairwise_agreement'] for r in per_question_results])
avg_majority = np.mean([r['majority_agreement'] for r in per_question_results])
print(f"\nAverage Pairwise Agreement: {avg_pairwise:.3f}")
print(f"Average Majority Agreement: {avg_majority:.3f}")

# Save detailed results
results_df = pd.DataFrame(per_question_results)
results_df.to_csv('v2_per_question_agreement.csv', index=False)
print("\nDetailed results saved to v2_per_question_agreement.csv")

# 3. Visualizations
print("\n=== GENERATING VISUALIZATIONS ===")

# Plot 1: Per-question agreement rates
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

questions = results_df['question'].tolist()
x_pos = np.arange(len(questions))

# Pairwise agreement
ax1.bar(x_pos, results_df['pairwise_agreement'], alpha=0.7, label='Pairwise Agreement')
ax1.axhline(y=avg_pairwise, color='r', linestyle='--', label=f'Average ({avg_pairwise:.3f})')
ax1.set_xlabel('Question')
ax1.set_ylabel('Agreement Rate')
ax1.set_title('Per-Question Pairwise Agreement Rates')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(questions, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Fleiss' Kappa
ax2.bar(x_pos, results_df['fleiss_kappa'], alpha=0.7, color='green', label="Fleiss' Kappa")
ax2.axhline(y=overall_fleiss, color='r', linestyle='--', label=f'Overall ({overall_fleiss:.3f})')
ax2.set_xlabel('Question')
ax2.set_ylabel("Fleiss' Kappa")
ax2.set_title("Per-Question Fleiss' Kappa Values")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(questions, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v2_per_question_agreement.png', dpi=300)
plt.close()

# Plot 2: Agreement metrics comparison
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Pairwise\nAgreement', 'Majority\nAgreement', "Fleiss'\nKappa", "Krippendorff's\nAlpha"]
values = [avg_pairwise, avg_majority, overall_fleiss, overall_alpha]
colors = ['blue', 'green', 'orange', 'red']

bars = ax.bar(metrics, values, color=colors, alpha=0.7)
ax.set_ylabel('Agreement Score')
ax.set_title('Overall Inter-Annotator Agreement Metrics (V2 Survey)')
ax.set_ylim(0, 1)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{value:.3f}', ha='center', va='bottom')

ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('v2_overall_agreement_metrics.png', dpi=300)
plt.close()

print("\nGenerated visualizations:")
print("- v2_per_question_agreement.png")
print("- v2_overall_agreement_metrics.png")

# Summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Questions with highest pairwise agreement:")
top_3 = results_df.nlargest(3, 'pairwise_agreement')[['question', 'pairwise_agreement']]
for _, row in top_3.iterrows():
    print(f"  {row['question']}: {row['pairwise_agreement']:.3f}")

print(f"\nQuestions with lowest pairwise agreement:")
bottom_3 = results_df.nsmallest(3, 'pairwise_agreement')[['question', 'pairwise_agreement']]
for _, row in bottom_3.iterrows():
    print(f"  {row['question']}: {row['pairwise_agreement']:.3f}")

print(f"\nAgreement interpretation:")
print(f"- Overall Krippendorff's Alpha ({overall_alpha:.3f}): {'Poor' if overall_alpha < 0.667 else 'Tentative' if overall_alpha < 0.8 else 'Good'} agreement")
print(f"- Overall Fleiss' Kappa ({overall_fleiss:.3f}): {'Slight' if overall_fleiss < 0.21 else 'Fair' if overall_fleiss < 0.41 else 'Moderate' if overall_fleiss < 0.61 else 'Substantial'} agreement")