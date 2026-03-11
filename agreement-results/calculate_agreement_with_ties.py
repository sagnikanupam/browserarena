#!/usr/bin/env python3
"""Calculate agreement rates INCLUDING ties for comparison"""

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

# Process responses INCLUDING ties
def get_numeric_response_with_ties(response):
    if pd.isna(response):
        return np.nan
    response = str(response).strip()
    if response == '1':
        return 1
    elif response == '2':
        return 2
    elif response == '3' or 'Tie' in str(response):
        return 3
    else:
        return np.nan

print("AGREEMENT RATES COMPARISON: WITH TIES vs WITHOUT TIES")
print("=" * 80)

# 1. Calculate WITH TIES
print("\n1. INCLUDING TIES:")
print("-" * 40)

# Human V2 inter-annotator reliability (with ties)
all_questions = sorted(question_cols)
reliability_data_with_ties = []

for annotator_idx in range(len(df_v2)):
    annotator_responses = []
    for q in all_questions:
        response = get_numeric_response_with_ties(df_v2.iloc[annotator_idx][q])
        annotator_responses.append(response)
    reliability_data_with_ties.append(annotator_responses)

reliability_data_with_ties = np.array(reliability_data_with_ties)
alpha_humans_with_ties = krippendorff.alpha(reliability_data=reliability_data_with_ties, level_of_measurement='nominal')

# Baseline vs individual human responses (with ties)
baseline_responses_with = []
human_responses_with = []

for q in all_questions:
    baseline_val = baseline_dict.get(q)
    if baseline_val:
        for annotator_idx in range(len(df_v2)):
            human_val = get_numeric_response_with_ties(df_v2.iloc[annotator_idx][q])
            if not np.isnan(human_val):
                baseline_responses_with.append(baseline_val)
                human_responses_with.append(int(human_val))

agreement_rate_with = sum(1 for b, h in zip(baseline_responses_with, human_responses_with) if b == h) / len(baseline_responses_with)
kappa_with = cohen_kappa_score(baseline_responses_with, human_responses_with)
alpha_baseline_humans_with = krippendorff.alpha(
    reliability_data=np.array([baseline_responses_with, human_responses_with]), 
    level_of_measurement='nominal'
)

print(f"Human V2 Inter-Annotator Agreement:")
print(f"  - Krippendorff's Alpha: {alpha_humans_with_ties:.3f}")
print(f"\nBaseline vs Human V2 (Individual Responses):")
print(f"  - Raw Agreement Rate: {agreement_rate_with:.3f} ({sum(1 for b, h in zip(baseline_responses_with, human_responses_with) if b == h)}/{len(baseline_responses_with)})")
print(f"  - Cohen's Kappa: {kappa_with:.3f}")
print(f"  - Krippendorff's Alpha: {alpha_baseline_humans_with:.3f}")

# 2. Calculate WITHOUT TIES (for comparison)
print("\n2. EXCLUDING TIES:")
print("-" * 40)

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

# Only non-tie baseline questions
all_questions_no_tie_baseline = sorted([q for q in question_cols if baseline_dict.get(q, 3) != 3])
reliability_data_no_ties = []

for annotator_idx in range(len(df_v2)):
    annotator_responses = []
    for q in all_questions_no_tie_baseline:
        response = get_numeric_response_no_ties(df_v2.iloc[annotator_idx][q])
        annotator_responses.append(response)
    reliability_data_no_ties.append(annotator_responses)

reliability_data_no_ties = np.array(reliability_data_no_ties)
alpha_humans_no_ties = krippendorff.alpha(reliability_data=reliability_data_no_ties, level_of_measurement='nominal')

# Baseline vs individual human responses (no ties)
baseline_responses_no = []
human_responses_no = []

for q in all_questions_no_tie_baseline:
    baseline_val = baseline_dict.get(q)
    if baseline_val and baseline_val != 3:
        for annotator_idx in range(len(df_v2)):
            human_val = get_numeric_response_no_ties(df_v2.iloc[annotator_idx][q])
            if not np.isnan(human_val):
                baseline_responses_no.append(baseline_val)
                human_responses_no.append(int(human_val))

agreement_rate_no = sum(1 for b, h in zip(baseline_responses_no, human_responses_no) if b == h) / len(baseline_responses_no)
kappa_no = cohen_kappa_score(baseline_responses_no, human_responses_no)
alpha_baseline_humans_no = krippendorff.alpha(
    reliability_data=np.array([baseline_responses_no, human_responses_no]), 
    level_of_measurement='nominal'
)

print(f"Human V2 Inter-Annotator Agreement:")
print(f"  - Krippendorff's Alpha: {alpha_humans_no_ties:.3f}")
print(f"\nBaseline vs Human V2 (Individual Responses):")
print(f"  - Raw Agreement Rate: {agreement_rate_no:.3f} ({sum(1 for b, h in zip(baseline_responses_no, human_responses_no) if b == h)}/{len(baseline_responses_no)})")
print(f"  - Cohen's Kappa: {kappa_no:.3f}")
print(f"  - Krippendorff's Alpha: {alpha_baseline_humans_no:.3f}")

# 3. Summary comparison
print("\n3. SUMMARY COMPARISON:")
print("-" * 40)
print(f"{'Metric':<40} {'With Ties':<15} {'Without Ties':<15} {'Change':<15}")
print("-" * 80)
print(f"{'Human Inter-Annotator Alpha':<40} {alpha_humans_with_ties:<15.3f} {alpha_humans_no_ties:<15.3f} {alpha_humans_no_ties - alpha_humans_with_ties:+15.3f}")
print(f"{'Baseline vs Human Agreement Rate':<40} {agreement_rate_with:<15.3f} {agreement_rate_no:<15.3f} {agreement_rate_no - agreement_rate_with:+15.3f}")
print(f"{'Baseline vs Human Kappa':<40} {kappa_with:<15.3f} {kappa_no:<15.3f} {kappa_no - kappa_with:+15.3f}")
print(f"{'Baseline vs Human Alpha':<40} {alpha_baseline_humans_with:<15.3f} {alpha_baseline_humans_no:<15.3f} {alpha_baseline_humans_no - alpha_baseline_humans_with:+15.3f}")

# 4. Detailed breakdown
print("\n4. VOTE DISTRIBUTION:")
print("-" * 40)

total_votes_with_ties = len(baseline_responses_with)
total_votes_no_ties = len(baseline_responses_no)
total_tie_votes = total_votes_with_ties - total_votes_no_ties

print(f"Total votes (with ties): {total_votes_with_ties}")
print(f"Total votes (no ties): {total_votes_no_ties}")
print(f"Tie votes excluded: {total_tie_votes} ({total_tie_votes/total_votes_with_ties*100:.1f}%)")

# Count agreements by type
print("\n5. AGREEMENT BREAKDOWN (WITH TIES):")
print("-" * 40)

agree_agent1 = sum(1 for b, h in zip(baseline_responses_with, human_responses_with) if b == 1 and h == 1)
agree_agent2 = sum(1 for b, h in zip(baseline_responses_with, human_responses_with) if b == 2 and h == 2)
agree_tie = sum(1 for b, h in zip(baseline_responses_with, human_responses_with) if b == 3 and h == 3)
total_agree = agree_agent1 + agree_agent2 + agree_tie

baseline_agent1_total = sum(1 for b in baseline_responses_with if b == 1)
baseline_agent2_total = sum(1 for b in baseline_responses_with if b == 2)
baseline_tie_total = sum(1 for b in baseline_responses_with if b == 3)

print(f"When baseline chose Agent 1: {agree_agent1}/{baseline_agent1_total} agreed ({agree_agent1/baseline_agent1_total*100:.1f}%)")
print(f"When baseline chose Agent 2: {agree_agent2}/{baseline_agent2_total} agreed ({agree_agent2/baseline_agent2_total*100:.1f}%)")
print(f"When baseline chose Tie: {agree_tie}/{baseline_tie_total} agreed ({agree_tie/baseline_tie_total*100:.1f}%)")
print(f"\nTotal agreements: {total_agree}/{total_votes_with_ties} ({total_agree/total_votes_with_ties*100:.1f}%)")