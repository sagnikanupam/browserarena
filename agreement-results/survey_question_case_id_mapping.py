#!/usr/bin/env python3
"""
Extract mapping between survey questions (Q1, Q2, etc.) and case IDs used in GPT-4o evaluations.
"""

import json
import csv
import pandas as pd
from collections import defaultdict

# Load GPT-4o evaluation results
with open('/Users/davisbrown/browserarena/gpt4o_full_dataset_evaluations/evaluation_results.json', 'r') as f:
    gpt4o_results = json.load(f)

# Create a mapping from response_id to case_ids
response_to_cases = {}
for result in gpt4o_results:
    response_id = result['response_id']
    response_to_cases[response_id] = {
        'case_id': result['case_id'],
        'left_task': result['left_task'],
        'right_task': result['right_task']
    }

# Load survey data
survey_files = [
    '/Users/davisbrown/browserarena/agreement-results/BrowserArenaAgreement_July_11_2025_11.03.csv',
    '/Users/davisbrown/browserarena/agreement-results/BrowserArenaAgreementv2_July_18_2025_11.01.csv'
]

# Extract question numbers and map to response IDs
question_mapping = []

for survey_file in survey_files:
    try:
        df = pd.read_csv(survey_file, nrows=1000)  # Read limited rows to check structure
        
        # Find Q columns
        q_columns = [col for col in df.columns if col.startswith('Q') and col[1:].isdigit()]
        
        print(f"\nAnalyzing {survey_file}")
        print(f"Found {len(q_columns)} Q columns: {q_columns[:10]}...")
        
        # Check if ResponseId column exists
        if 'ResponseId' in df.columns:
            for idx, row in df.iterrows():
                response_id = row['ResponseId']
                if response_id in response_to_cases:
                    for q_col in q_columns:
                        question_mapping.append({
                            'survey_file': survey_file.split('/')[-1],
                            'question_number': q_col,
                            'response_id': response_id,
                            'case_id': response_to_cases[response_id]['case_id'],
                            'left_task': response_to_cases[response_id]['left_task'],
                            'right_task': response_to_cases[response_id]['right_task']
                        })
    except Exception as e:
        print(f"Error processing {survey_file}: {e}")

# Save the mapping
output_file = '/Users/davisbrown/browserarena/agreement-results/survey_question_to_case_id_mapping.json'
with open(output_file, 'w') as f:
    json.dump(question_mapping, f, indent=2)

# Create a summary
summary = defaultdict(list)
for mapping in question_mapping:
    key = f"{mapping['question_number']} -> {mapping['case_id']}"
    summary[mapping['question_number']].append({
        'response_id': mapping['response_id'],
        'case_id': mapping['case_id']
    })

# Print summary
print(f"\n\nSummary of Question to Case ID Mappings:")
print(f"Total mappings found: {len(question_mapping)}")
print(f"\nSample mappings:")
for q_num in sorted(list(summary.keys()))[:10]:
    if summary[q_num]:
        sample = summary[q_num][0]
        print(f"{q_num}: Response {sample['response_id']} -> Case {sample['case_id']}")

print(f"\nMapping saved to: {output_file}")

# Also check the faithfulness data which has clearer structure
print("\n\nChecking faithfulness_data.csv for direct mappings...")
faith_df = pd.read_csv('/Users/davisbrown/browserarena/agreement-results/data_updated/faithfulness_data.csv')

# Find columns with case IDs
case_id_pattern = r'\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2}_[a-zA-Z]{3}'
case_id_columns = []
for col in faith_df.columns:
    if faith_df[col].astype(str).str.match(case_id_pattern).any():
        case_id_columns.append(col)

print(f"Columns containing case IDs: {case_id_columns}")

# Extract direct mappings from faithfulness data
direct_mappings = []
for idx, row in faith_df.iterrows():
    response_id = row['ResponseId']
    if pd.notna(row.get('Q7')) and pd.notna(row.get('Q24')):
        # Check if these look like case IDs
        if '_' in str(row['Q7']) and '_' in str(row['Q24']):
            direct_mappings.append({
                'response_id': response_id,
                'task_question': row.get('Q2', ''),
                'Q7_left_case': str(row['Q7']),
                'Q24_right_case': str(row['Q24'])
            })

# Save direct mappings
direct_output = '/Users/davisbrown/browserarena/agreement-results/direct_survey_case_mappings.json'
with open(direct_output, 'w') as f:
    json.dump(direct_mappings, f, indent=2)

print(f"\nDirect mappings from faithfulness data: {len(direct_mappings)}")
print(f"Saved to: {direct_output}")