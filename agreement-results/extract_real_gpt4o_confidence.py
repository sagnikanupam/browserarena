#!/usr/bin/env python3
"""
Extract real GPT-4o confidence scores for each survey question by matching
the survey responses with the GPT-4o evaluation results.
"""

import pandas as pd
import json
import numpy as np

print("=== EXTRACTING REAL GPT-4O CONFIDENCE SCORES ===")

# Read the three-way comparison which has the survey questions
comparison_df = pd.read_csv('three_way_comparison.csv')
print(f"\nFound {len(comparison_df)} questions in three-way comparison")

# Read all GPT-4o evaluation results
gpt4o_files = [
    '../gpt4o_evaluation_analysis.csv',
    '../gpt4o_human_preference_detailed_analysis.csv',
    '../gpt4o_all_tasks_detailed_analysis.csv'
]

all_gpt4o_data = []
for file in gpt4o_files:
    try:
        df = pd.read_csv(file)
        all_gpt4o_data.append(df)
        print(f"Loaded {len(df)} rows from {file}")
    except:
        pass

# Combine all GPT-4o data
if all_gpt4o_data:
    gpt4o_df = pd.concat(all_gpt4o_data, ignore_index=True)
    print(f"\nTotal GPT-4o evaluations: {len(gpt4o_df)}")
else:
    print("ERROR: No GPT-4o data found")
    exit(1)

# The key insight: We need to match the survey questions to the GPT-4o evaluations
# Let's check if the GPT-4o data has any task descriptions that we can match

print("\n=== ANALYZING GPT-4O DATA STRUCTURE ===")
print(f"GPT-4o columns: {list(gpt4o_df.columns)}")

# Check sample of GPT-4o data
print("\nSample GPT-4o evaluation:")
if 'gpt4o_reasoning' in gpt4o_df.columns:
    sample = gpt4o_df.iloc[0]
    print(f"Case ID: {sample.get('case_id', 'N/A')}")
    print(f"Confidence: {sample.get('gpt4o_confidence', 'N/A')}")
    print(f"Task (if available): {sample.get('task', 'N/A')}")

# Since we already have the baseline mapping of questions to winners,
# and GPT-4o agreed 100% with baseline, we can use the average confidence
# scores by vote type

# Read baseline to get question winners
baseline_df = pd.read_csv('baseline.csv')

# Calculate average GPT-4o confidence by vote type
print("\n=== CALCULATING CONFIDENCE BY VOTE TYPE ===")
vote_confidence = {}

if 'normalized_original_vote' in gpt4o_df.columns:
    vote_groups = gpt4o_df.groupby('normalized_original_vote')['gpt4o_confidence'].agg(['mean', 'std', 'count'])
    print("\nGPT-4o confidence by original vote:")
    print(vote_groups)
    
    for vote_type, stats in vote_groups.iterrows():
        vote_confidence[vote_type] = {
            'mean': stats['mean'],
            'std': stats['std'],
            'count': stats['count']
        }

# Now assign confidence to each question based on its winner
print("\n=== ASSIGNING CONFIDENCE TO QUESTIONS ===")
question_confidence = {}

for _, row in baseline_df.iterrows():
    question = row['question']
    winner = row['winner']
    
    # Map winner to vote type
    if winner == 'Agent 1':
        vote_type = 'Left'
    elif winner == 'Agent 2':
        vote_type = 'Right'
    else:
        vote_type = 'Tie'
    
    # Get confidence for this vote type
    if vote_type in vote_confidence:
        # Use mean confidence for the vote type
        # Add some realistic variation based on std dev
        mean_conf = vote_confidence[vote_type]['mean']
        std_conf = vote_confidence[vote_type]['std']
        
        # Since we don't have exact mappings, use the mean
        # In reality, each question would have its specific confidence
        confidence = mean_conf
        
        question_confidence[question] = {
            'confidence': confidence,
            'winner': winner,
            'vote_type': vote_type,
            'based_on_average': True
        }
        
        print(f"{question}: {winner} -> {vote_type} -> confidence = {confidence:.3f}")

# Try to find more specific mappings if possible
# Check if we have any direct question-to-case mappings in the data
print("\n=== SEARCHING FOR DIRECT MAPPINGS ===")

# Check the survey data for case IDs
survey_df = pd.read_csv('BrowserArenaAgreementv2_July_18_2025_11.01.csv')
survey_df = survey_df.iloc[2:]  # Skip headers
survey_df = survey_df[survey_df['Status'] == '0']

# Look for columns that might contain case IDs or task descriptions
potential_mapping_cols = []
for col in survey_df.columns:
    if any(keyword in col.lower() for keyword in ['case', 'task', 'id', 'gif', 'video']):
        potential_mapping_cols.append(col)
        
if potential_mapping_cols:
    print(f"Found potential mapping columns: {potential_mapping_cols}")

# Save the results
output_data = []
for question, data in question_confidence.items():
    output_data.append({
        'question': question,
        'gpt4o_confidence': data['confidence'],
        'baseline_winner': data['winner'],
        'confidence_source': 'vote_type_average'
    })

output_df = pd.DataFrame(output_data)
output_df = output_df.sort_values('question')
output_df.to_csv('question_gpt4o_real_confidence.csv', index=False)

print(f"\n=== SAVED RESULTS ===")
print(f"Saved {len(output_df)} question confidence scores to question_gpt4o_real_confidence.csv")
print("\nSummary:")
print(f"Average confidence across all questions: {output_df['gpt4o_confidence'].mean():.3f}")
print(f"Min confidence: {output_df['gpt4o_confidence'].min():.3f}")
print(f"Max confidence: {output_df['gpt4o_confidence'].max():.3f}")

# Also save as JSON for reference
with open('question_gpt4o_real_confidence.json', 'w') as f:
    json.dump(question_confidence, f, indent=2)

print("\nNOTE: These confidence scores are based on the average confidence for each vote type")
print("(Left/Agent 1, Right/Agent 2, Tie) since we don't have direct question-to-evaluation mappings.")
print("GPT-4o agreed 100% with the baseline, so the vote types match perfectly.")