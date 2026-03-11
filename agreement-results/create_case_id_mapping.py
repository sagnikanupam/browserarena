#!/usr/bin/env python3
"""
Create a comprehensive mapping between survey questions and case IDs.
"""

import json
import csv
import pandas as pd

# Load the faithfulness data
df = pd.read_csv('/Users/davisbrown/browserarena/agreement-results/data_updated/faithfulness_data.csv')

# Get column names
print("Column positions:")
for i, col in enumerate(df.columns):
    if 'Q' in col:
        print(f"Column {i}: {col}")

# The case IDs are in specific positions based on our grep results
# Column 26 (0-indexed) is Q7 with left case ID
# Column 45 (0-indexed) should have right case ID

mappings = []

for idx, row in df.iterrows():
    response_id = row['ResponseId']
    task_question = row['Q2'] if pd.notna(row['Q2']) else ''
    
    # Get values from the correct columns
    # Q7 is the left case ID (column index 25)
    # Q25 is the right case ID (column index 43)
    left_case = str(row['Q7']) if pd.notna(row['Q7']) else ''
    right_case = str(row['Q25']) if pd.notna(row['Q25']) else ''
    
    # Check if these are valid case IDs (format: DD_MM_YYYY_HH_MM_SS_XXX)
    case_id_pattern = r'^\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2}_[a-zA-Z]{3}$'
    import re
    
    if re.match(case_id_pattern, left_case) and re.match(case_id_pattern, right_case):
        mapping = {
            'response_id': response_id,
            'task_question': task_question,
            'left_case_id': left_case,
            'right_case_id': right_case,
            'combined_case_id': f"{left_case}_{right_case}"
        }
        mappings.append(mapping)
        
        # For debugging, print first few
        if len(mappings) <= 5:
            print(f"\nFound mapping:")
            print(f"  Response ID: {response_id}")
            print(f"  Task: {task_question[:50]}...")
            print(f"  Left Case: {left_case}")
            print(f"  Right Case: {right_case}")

# Save the mappings
output_file = '/Users/davisbrown/browserarena/agreement-results/survey_case_id_complete_mapping.json'
with open(output_file, 'w') as f:
    json.dump(mappings, f, indent=2)

# Also create a CSV version
csv_output = '/Users/davisbrown/browserarena/agreement-results/survey_case_id_complete_mapping.csv'
if mappings:
    keys = mappings[0].keys()
    with open(csv_output, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(mappings)

print(f"\n\nTotal mappings found: {len(mappings)}")
print(f"Saved to:")
print(f"  JSON: {output_file}")
print(f"  CSV: {csv_output}")

# Create a lookup table for easy access
lookup = {}
for mapping in mappings:
    lookup[mapping['response_id']] = mapping
    lookup[mapping['combined_case_id']] = mapping

lookup_file = '/Users/davisbrown/browserarena/agreement-results/case_id_lookup.json'
with open(lookup_file, 'w') as f:
    json.dump(lookup, f, indent=2)

print(f"  Lookup table: {lookup_file}")

# Check if we have the specific case you mentioned
target_case = "10_05_2025_03_16_40_dQm_10_05_2025_03_16_40_fSK"
if target_case in lookup:
    print(f"\n\nFound the specific case you mentioned:")
    print(json.dumps(lookup[target_case], indent=2))