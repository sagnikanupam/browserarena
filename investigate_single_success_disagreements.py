#!/usr/bin/env python3
"""
Investigate cases where only one agent succeeded but there's disagreement
"""

import json
import pandas as pd

def main():
    # Load results
    with open('gpt4o_all_tasks_evaluations/evaluation_results.json', 'r') as f:
        results = json.load(f)

    df = pd.DataFrame(results)
    valid_df = df[df['gpt4o_preference'] != 'error'].copy()

    # Normalize votes
    valid_df['normalized_human_vote'] = valid_df['original_vote'].str.lower().str.replace(r'.*left.*', 'left', regex=True).str.replace(r'.*right.*', 'right', regex=True)
    valid_df['normalized_gpt4o_vote'] = valid_df['gpt4o_preference'].str.lower()

    # Find single success cases
    right_only = valid_df[(valid_df['right_success'] == True) & (valid_df['left_success'] == False)]
    left_only = valid_df[(valid_df['left_success'] == True) & (valid_df['right_success'] == False)]

    print(f"Right only succeeded: {len(right_only)} cases")
    print(f"Left only succeeded: {len(left_only)} cases")

    # Check agreement
    right_agree = right_only['normalized_human_vote'] == right_only['normalized_gpt4o_vote']
    left_agree = left_only['normalized_human_vote'] == left_only['normalized_gpt4o_vote']

    print(f"\nRight only - Agreement: {right_agree.sum()}/{len(right_only)} ({right_agree.sum()/len(right_only)*100:.1f}%)")
    print(f"Left only - Agreement: {left_agree.sum()}/{len(left_only)} ({left_agree.sum()/len(left_only)*100:.1f}%)")

    print("\n=== RIGHT ONLY SUCCEEDED - DISAGREEMENT CASES ===")
    right_disagree = right_only[~right_agree]
    for _, row in right_disagree.head(5).iterrows():
        print(f"\nCase: {row['case_id']}")
        print(f"Human: '{row['original_vote']}' -> {row['normalized_human_vote']}")
        print(f"GPT-4o: {row['gpt4o_preference']} (confidence: {row['gpt4o_confidence']:.2f})")
        print(f"Reasoning: {row['gpt4o_reasoning'][:200]}...")

    print("\n=== LEFT ONLY SUCCEEDED - DISAGREEMENT CASES ===")
    left_disagree = left_only[~left_agree]
    for _, row in left_disagree.head(5).iterrows():
        print(f"\nCase: {row['case_id']}")
        print(f"Human: '{row['original_vote']}' -> {row['normalized_human_vote']}")
        print(f"GPT-4o: {row['gpt4o_preference']} (confidence: {row['gpt4o_confidence']:.2f})")
        print(f"Reasoning: {row['gpt4o_reasoning'][:200]}...")

    # Check if humans are correctly identifying the successful agent
    print("\n=== CHECKING IF HUMANS VOTE FOR SUCCESSFUL AGENT ===")
    
    print(f"\nRight only succeeded - Human votes:")
    right_human_votes = right_only['normalized_human_vote'].value_counts()
    print(right_human_votes)
    print(f"Humans correctly chose 'right': {right_human_votes.get('right', 0)}/{len(right_only)} ({right_human_votes.get('right', 0)/len(right_only)*100:.1f}%)")
    
    print(f"\nLeft only succeeded - Human votes:")
    left_human_votes = left_only['normalized_human_vote'].value_counts()
    print(left_human_votes)
    print(f"Humans correctly chose 'left': {left_human_votes.get('left', 0)}/{len(left_only)} ({left_human_votes.get('left', 0)/len(left_only)*100:.1f}%)")

if __name__ == "__main__":
    main()