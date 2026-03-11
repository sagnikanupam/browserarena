#!/usr/bin/env python3
"""
Verify GPT-4 results vs baseline - check if they're suspiciously similar
"""

import json
import csv
from pathlib import Path

# Load GPT-4 results
with open('gpt4_evaluation_results.json', 'r') as f:
    gpt4_data = json.load(f)

# Load baseline
baseline_votes = {}
question_map = {
    'Q1': 'interaction_01', 'Q2': 'interaction_02', 'Q3': 'interaction_03',
    'Q4': 'interaction_04', 'Q5': 'interaction_05', 'Q6': 'interaction_06',
    'Q7': 'interaction_07', 'Q8': 'interaction_08', 'Q9': 'interaction_09',
    'Q10': 'interaction_10', 'Q11': 'interaction_11', 'Q12': 'interaction_12',
    'Q13': 'interaction_13', 'Q14': 'interaction_14', 'Q15': 'interaction_15',
    'Q16': 'interaction_16', 'Q17': 'interaction_17', 'Q18': 'interaction_18',
    'Q19': 'interaction_19', 'Q20': 'interaction_20', 'Q21': 'interaction_21',
    'Q22': 'interaction_22', 'Q23': 'interaction_23', 'Q24': 'interaction_24',
    'Q25': 'interaction_25'
}

with open('agreement-results/baseline.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['question'] in question_map:
            int_id = question_map[row['question']]
            winner = row['winner']
            if winner == 'Agent 1':
                vote = 'Left'
            elif winner == 'Agent 2':
                vote = 'Right'
            else:
                vote = 'Tie'
            baseline_votes[int_id] = vote

print("DETAILED COMPARISON: GPT-4 vs BASELINE")
print("="*60)
print(f"{'Interaction':<20} {'GPT-4':<10} {'Baseline':<10} {'Match':<10}")
print("-"*60)

matches = 0
total = 0

for int_id in sorted(gpt4_data['evaluations'].keys()):
    gpt_vote = gpt4_data['evaluations'][int_id]
    base_vote = baseline_votes.get(int_id, 'N/A')
    
    if base_vote != 'N/A':
        total += 1
        match = gpt_vote == base_vote
        if match:
            matches += 1
        match_str = '✓' if match else '✗'
    else:
        match_str = '-'
    
    print(f"{int_id:<20} {gpt_vote:<10} {base_vote:<10} {match_str:<10}")

print("-"*60)
print(f"Agreement rate: {matches}/{total} = {matches/total*100:.1f}%")

# Show disagreements
print("\nDISAGREEMENTS:")
print("-"*60)
for int_id in sorted(gpt4_data['evaluations'].keys()):
    gpt_vote = gpt4_data['evaluations'][int_id]
    base_vote = baseline_votes.get(int_id, 'N/A')
    if base_vote != 'N/A' and gpt_vote != base_vote:
        print(f"{int_id}: GPT-4={gpt_vote}, Baseline={base_vote}")