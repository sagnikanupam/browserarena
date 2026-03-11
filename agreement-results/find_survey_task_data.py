#!/usr/bin/env python3
"""
Find the actual task data and agent outputs for each survey question.
"""

import json
import pandas as pd
from pathlib import Path
import re

print("=== FINDING TASK DATA FOR SURVEY QUESTIONS ===")

# Read the baseline data
baseline_df = pd.read_csv('baseline.csv')
survey_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 
                   'Q11', 'Q12', 'Q13', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q22']

print(f"Looking for data for {len(survey_questions)} survey questions")

# Check various data sources
print("\n1. Checking for HTML snippets...")
snippets_dirs = [
    Path('../survey_html_snippets'),
    Path('../survey_html_snippets_truncated'),
    Path('survey_html_snippets'),
    Path('survey_html_snippets_truncated')
]

html_files_found = []
for snippets_dir in snippets_dirs:
    if snippets_dir.exists():
        files = list(snippets_dir.glob('*.html'))
        print(f"Found {len(files)} HTML files in {snippets_dir}")
        html_files_found.extend(files)
        
        # Show first few files
        for f in sorted(files)[:5]:
            print(f"  - {f.name}")

# Check for JSON interaction files
print("\n2. Checking for JSON interaction files...")
json_patterns = [
    '*interaction*.json',
    '*verified*.json',
    '*survey*.json'
]

json_files_found = []
for pattern in json_patterns:
    files = list(Path('..').glob(pattern))
    if files:
        print(f"Found {len(files)} files matching {pattern}")
        json_files_found.extend(files)
        for f in files[:3]:
            print(f"  - {f.name}")

# Check the data_updated directory
print("\n3. Checking data_updated directory...")
data_dir = Path('data_updated')
if data_dir.exists():
    csv_files = list(data_dir.glob('*.csv'))
    json_files = list(data_dir.glob('*.json'))
    print(f"Found {len(csv_files)} CSV files and {len(json_files)} JSON files")
    
    # Check faithfulness data specifically
    faith_file = data_dir / 'faithfulness_data.csv'
    if faith_file.exists():
        faith_df = pd.read_csv(faith_file)
        print(f"\nFaithfulness data: {faith_df.shape}")
        print("Columns with Q prefix:")
        q_cols = [col for col in faith_df.columns if col.startswith('Q')]
        print(f"  {q_cols[:10]}...")
        
        # Check what's in Q2 (task descriptions)
        if 'Q2' in faith_df.columns:
            print("\nSample tasks from Q2:")
            for i in range(min(3, len(faith_df))):
                task = str(faith_df.iloc[i]['Q2'])[:100]
                print(f"  Row {i}: {task}...")

# Check FastChat directory for prompts and outputs
print("\n4. Checking FastChat directory...")
fastchat_dir = Path('../FastChat')
if fastchat_dir.exists():
    prompts_dir = fastchat_dir / 'prompts_and_outputs'
    if prompts_dir.exists():
        json_files = list(prompts_dir.glob('*.json'))
        print(f"Found {len(json_files)} JSON files in prompts_and_outputs")
        
        # Sample one to see structure
        if json_files:
            with open(json_files[0], 'r') as f:
                sample = json.load(f)
            print(f"\nSample structure from {json_files[0].name}:")
            print(f"  Keys: {list(sample.keys())}")

# Look for any files that might map questions to tasks
print("\n5. Looking for mapping files...")
mapping_patterns = [
    '*map*.csv',
    '*map*.json',
    '*question*.csv',
    '*question*.json'
]

for pattern in mapping_patterns:
    files = list(Path('..').glob(pattern))
    if files:
        print(f"\nFound {len(files)} files matching {pattern}:")
        for f in files[:5]:
            print(f"  - {f}")

# Check if we have the actual survey response data with task info
print("\n6. Checking survey response data...")
survey_file = 'BrowserArenaAgreementv2_July_18_2025_11.01.csv'
if Path(survey_file).exists():
    survey_df = pd.read_csv(survey_file)
    print(f"Survey data shape: {survey_df.shape}")
    
    # Look for columns that might contain task info
    print("\nColumns that might contain task info:")
    for col in survey_df.columns:
        if any(keyword in col.lower() for keyword in ['task', 'prompt', 'description', 'gif', 'agent']):
            print(f"  - {col}")
    
    # Check if questions contain task descriptions
    survey_df_clean = survey_df.iloc[2:]  # Skip headers
    survey_df_clean = survey_df_clean[survey_df_clean['Status'] == '0']
    
    print(f"\nChecking content of question columns:")
    for q in ['Q1', 'Q2', 'Q3']:
        if q in survey_df_clean.columns:
            sample = survey_df_clean[q].dropna().iloc[0] if len(survey_df_clean[q].dropna()) > 0 else 'N/A'
            print(f"  {q}: {str(sample)[:100]}...")

print("\n=== SUMMARY ===")
print(f"HTML files found: {len(html_files_found)}")
print(f"JSON files found: {len(json_files_found)}")
print("\nTo proceed with exact GPT-4o evaluation, we need:")
print("1. The actual task descriptions for each survey question")
print("2. The agent outputs (text or trajectory data) for each question")
print("3. A mapping between question numbers (Q1, Q2, etc.) and the actual task data")

# Create a template mapping file that can be filled in
print("\nCreating template mapping file...")
template = []
for q in survey_questions:
    baseline_row = baseline_df[baseline_df['question'] == q]
    if not baseline_row.empty:
        template.append({
            'question': q,
            'task_description': '',  # To be filled
            'task_id': '',  # To be filled
            'agent1_output_file': '',  # To be filled
            'agent2_output_file': '',  # To be filled
            'baseline_winner': baseline_row.iloc[0]['winner'],
            'agent1_model': baseline_row.iloc[0]['agent1_model'],
            'agent2_model': baseline_row.iloc[0]['agent2_model']
        })

template_df = pd.DataFrame(template)
template_df.to_csv('survey_questions_mapping_template.csv', index=False)
print("Created survey_questions_mapping_template.csv - fill this in with actual task data")