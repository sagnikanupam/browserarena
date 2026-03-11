#!/usr/bin/env python3
"""
Create a comprehensive CSV file with all cases where both agents succeeded,
ready for GPT-4o evaluation.
"""

import json
import pandas as pd
from pathlib import Path
import re

def create_comprehensive_csv():
    """Create CSV with all necessary information for evaluation"""
    
    # Load the full dataset cases
    with open('full_dataset_cases.json', 'r') as f:
        data = json.load(f)
    
    # Filter for successful cases
    successful_cases = [c for c in data['all_cases'] if c['both_succeeded']]
    
    print(f"Processing {len(successful_cases)} cases where both agents succeeded...")
    
    # Load the original analyzed dataset for additional context
    df_original = pd.read_csv('agreement-results/data_updated/ProlificBrowserArenaGIFFeedbackForm_May 14, 2025_22.32_analyzed.csv')
    
    # Create records for CSV
    records = []
    
    for case in successful_cases:
        record = {
            'case_id': f"{case['left_task']}_{case['right_task']}",
            'response_id': case['response_id'],
            'left_task_id': case['left_task'],
            'right_task_id': case['right_task'],
            'original_vote': case['vote'],
            'left_agent_success': 'Yes',
            'right_agent_success': 'Yes',
            'both_agents_successful': 'Yes'
        }
        
        # Try to find additional information from original dataset
        original_row = df_original[df_original['ResponseId'] == case['response_id']]
        if not original_row.empty:
            row = original_row.iloc[0]
            
            # Extract task description from Q2 or similar
            task_desc = None
            for q in ['Q2', 'Q3', 'Q4']:
                if pd.notna(row.get(q)):
                    val = str(row[q])
                    if len(val) > 20 and 'goal' not in val.lower():
                        task_desc = val
                        break
            
            if task_desc:
                record['task_description'] = task_desc[:200] + '...' if len(task_desc) > 200 else task_desc
            
            # Add metadata
            record['duration_seconds'] = row.get('Duration (in seconds)', None)
            record['prolific_pid'] = row.get('PROLIFIC_PID', None)
            record['recorded_date'] = row.get('RecordedDate', None)
        
        # File paths
        record['left_gif_path'] = f"FastChat/gifs/{case['left_task']}.gif"
        record['right_gif_path'] = f"FastChat/gifs/{case['right_task']}.gif"
        record['left_prompt_output_path'] = f"FastChat/prompts_and_outputs/{case['left_task']}.json"
        record['right_prompt_output_path'] = f"FastChat/prompts_and_outputs/{case['right_task']}.json"
        
        # Placeholders for GPT-4o evaluation
        record['gpt4o_preference'] = ''
        record['gpt4o_confidence'] = ''
        record['gpt4o_reasoning'] = ''
        record['gpt4o_agrees_with_original'] = ''
        
        records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save to CSV
    output_file = 'full_dataset_both_agents_successful_for_evaluation.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nCreated CSV with {len(df)} cases: {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total cases where both agents succeeded: {len(df)}")
    
    vote_counts = df['original_vote'].value_counts()
    print("\nOriginal vote distribution:")
    for vote, count in vote_counts.items():
        print(f"  {vote}: {count} ({count/len(df)*100:.1f}%)")
    
    # Also create a sample file for testing
    sample_df = df.head(10)
    sample_file = 'sample_cases_for_evaluation.csv'
    sample_df.to_csv(sample_file, index=False)
    print(f"\nCreated sample file with 10 cases: {sample_file}")
    
    return df

def check_file_availability(df):
    """Check which GIF and JSON files actually exist"""
    
    print("\nChecking file availability...")
    
    gif_found = 0
    json_found = 0
    
    for _, row in df.iterrows():
        # Check GIF files
        if Path(row['left_gif_path']).exists() and Path(row['right_gif_path']).exists():
            gif_found += 1
        
        # Check JSON files
        if Path(row['left_prompt_output_path']).exists() and Path(row['right_prompt_output_path']).exists():
            json_found += 1
    
    print(f"GIF files found for: {gif_found}/{len(df)} cases ({gif_found/len(df)*100:.1f}%)")
    print(f"JSON files found for: {json_found}/{len(df)} cases ({json_found/len(df)*100:.1f}%)")

if __name__ == "__main__":
    df = create_comprehensive_csv()
    check_file_availability(df)