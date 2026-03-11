#!/usr/bin/env python3
"""Correctly calculate reliability metrics for no-ties analysis using individual responses"""

import pandas as pd
import numpy as np
import krippendorff
from sklearn.metrics import cohen_kappa_score

# Read the v2 data
df_v2 = pd.read_csv('BrowserArenaAgreementv2_July_18_2025_11.01.csv')
df_v2 = df_v2.iloc[2:]  # Skip header rows
df_v2 = df_v2[df_v2['Status'] == '0']  # Completed responses

# Read baseline
baseline_df = pd.read_csv('baseline.csv')
baseline_dict = {}
for _, row in baseline_df.iterrows():
    q_num = row['question']
    if row['winner'] == 'Agent 1':
        baseline_dict[q_num] = 1
    elif row['winner'] == 'Agent 2':
        baseline_dict[q_num] = 2
    elif row['winner'] == 'Tie':
        baseline_dict[q_num] = 3

# Get question columns
question_cols = [col for col in df_v2.columns if col.startswith('Q') and col[1:].isdigit()]

# Process responses excluding ties
def get_numeric_response_no_ties(response):
    if pd.isna(response):
        return np.nan
    response = str(response).strip()
    if response == '1':
        return 1
    elif response == '2':
        return 2
    elif response == '3' or 'Tie' in str(response):
        return np.nan  # Exclude ties
    else:
        return np.nan

print("CORRECTED RELIABILITY METRICS (EXCLUDING TIES)")
print("=" * 60)

# 1. Calculate Krippendorff's alpha for human annotators (no ties)
print("\n1. Human V2 Inter-Annotator Reliability (No Ties):")

# Create reliability data matrix
all_questions = sorted([q for q in question_cols if baseline_dict.get(q, 3) != 3])  # Exclude baseline ties
reliability_data = []

for annotator_idx in range(len(df_v2)):
    annotator_responses = []
    for q in all_questions:
        response = get_numeric_response_no_ties(df_v2.iloc[annotator_idx][q])
        annotator_responses.append(response)
    reliability_data.append(annotator_responses)

reliability_data = np.array(reliability_data)

# Calculate Krippendorff's alpha
alpha_humans = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
print(f"Krippendorff's Alpha (Human V2 annotators, no ties): {alpha_humans:.3f}")

# 2. Calculate agreement between baseline and individual human responses
print("\n2. Baseline vs Individual Human Responses (No Ties):")

baseline_responses = []
human_responses = []

for q in all_questions:
    baseline_val = baseline_dict.get(q)
    if baseline_val and baseline_val != 3:  # Exclude baseline ties
        for annotator_idx in range(len(df_v2)):
            human_val = get_numeric_response_no_ties(df_v2.iloc[annotator_idx][q])
            if not np.isnan(human_val):
                baseline_responses.append(baseline_val)
                human_responses.append(int(human_val))

# Calculate metrics
if len(baseline_responses) > 0:
    agreement_rate = sum(1 for b, h in zip(baseline_responses, human_responses) if b == h) / len(baseline_responses)
    print(f"Raw agreement rate: {agreement_rate:.3f} ({sum(1 for b, h in zip(baseline_responses, human_responses) if b == h)}/{len(baseline_responses)})")
    
    kappa = cohen_kappa_score(baseline_responses, human_responses)
    print(f"Cohen's Kappa (Baseline vs Human individuals): {kappa:.3f}")
    
    # Krippendorff's alpha between baseline and humans
    alpha_baseline_humans = krippendorff.alpha(
        reliability_data=np.array([baseline_responses, human_responses]), 
        level_of_measurement='nominal'
    )
    print(f"Krippendorff's Alpha (Baseline vs Human individuals): {alpha_baseline_humans:.3f}")

# 3. Per-question agreement analysis
print("\n3. Per-Question Analysis (No Ties):")
print(f"{'Question':<10} {'Baseline':<10} {'Agent1':<10} {'Agent2':<10} {'Total':<10} {'% Agree':<10}")
print("-" * 60)

for q in all_questions:
    baseline_val = baseline_dict.get(q)
    if baseline_val and baseline_val != 3:
        agent1_count = 0
        agent2_count = 0
        
        for annotator_idx in range(len(df_v2)):
            response = get_numeric_response_no_ties(df_v2.iloc[annotator_idx][q])
            if response == 1:
                agent1_count += 1
            elif response == 2:
                agent2_count += 1
        
        total = agent1_count + agent2_count
        if total > 0:
            agree_count = agent1_count if baseline_val == 1 else agent2_count
            agree_pct = (agree_count / total) * 100
            baseline_label = "Agent 1" if baseline_val == 1 else "Agent 2"
            
            print(f"{q:<10} {baseline_label:<10} {agent1_count:<10} {agent2_count:<10} {total:<10} {agree_pct:<10.1f}%")

# Summary
print("\n" + "=" * 60)
print("SUMMARY:")
print(f"- Krippendorff's Alpha (Human annotators only): {alpha_humans:.3f}")
print(f"- Cohen's Kappa (Baseline vs Humans): {kappa:.3f}")
print(f"- Raw agreement (Baseline vs Humans): {agreement_rate:.3f}")
print("\nNote: These metrics are calculated on individual responses, not majority votes.")
print("Perfect agreement at the majority level does NOT mean perfect individual agreement.")