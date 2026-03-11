#!/usr/bin/env python3
"""
Create a final summary of the dataset prepared for GPT-4o evaluation
"""

import pandas as pd
import json

def create_summary():
    """Create summary of the evaluation dataset"""
    
    # Load the cleaned dataset
    df = pd.read_csv('full_dataset_both_successful_cleaned_for_gpt4o_eval.csv')
    
    print("BROWSERARENA EVALUATION DATASET SUMMARY")
    print("="*60)
    
    print(f"\nTotal cases where both agents succeeded: {len(df)}")
    
    # Vote distribution
    print("\nHuman vote distribution (normalized):")
    vote_counts = df['normalized_vote'].value_counts()
    for vote, count in vote_counts.items():
        print(f"  {vote}: {count} ({count/len(df)*100:.1f}%)")
    
    # Check file availability
    gif_exists = 0
    json_exists = 0
    both_exist = 0
    
    for _, row in df.iterrows():
        left_gif = Path(row['left_gif_path']).exists()
        right_gif = Path(row['right_gif_path']).exists()
        left_json = Path(row['left_prompt_output_path']).exists()
        right_json = Path(row['right_prompt_output_path']).exists()
        
        if left_gif and right_gif:
            gif_exists += 1
        if left_json and right_json:
            json_exists += 1
        if left_gif and right_gif and left_json and right_json:
            both_exist += 1
    
    print(f"\nFile availability:")
    print(f"  Cases with both GIF files: {gif_exists} ({gif_exists/len(df)*100:.1f}%)")
    print(f"  Cases with both JSON files: {json_exists} ({json_exists/len(df)*100:.1f}%)")
    print(f"  Cases with all files: {both_exist} ({both_exist/len(df)*100:.1f}%)")
    
    # Date distribution
    print("\nData collection dates:")
    # Extract dates from task IDs
    dates = []
    for _, row in df.iterrows():
        task_id = row['left_task_id']
        # Extract date from format: DD_MM_YYYY_HH_MM_SS_XXX
        parts = task_id.split('_')
        if len(parts) >= 3:
            date = f"{parts[2]}-{parts[1]}-{parts[0]}"  # YYYY-MM-DD
            dates.append(date)
    
    date_counts = pd.Series(dates).value_counts().sort_index()
    for date, count in date_counts.items():
        print(f"  {date}: {count} cases")
    
    # Create final summary CSV with key information
    summary_df = df[['case_id', 'left_task_id', 'right_task_id', 'normalized_vote', 'task_description']].copy()
    
    # Add file existence flags
    file_checks = []
    for _, row in df.iterrows():
        left_gif = Path(row['left_gif_path']).exists()
        right_gif = Path(row['right_gif_path']).exists()
        left_json = Path(row['left_prompt_output_path']).exists()
        right_json = Path(row['right_prompt_output_path']).exists()
        
        file_checks.append({
            'case_id': row['case_id'],
            'has_gif_files': left_gif and right_gif,
            'has_json_files': left_json and right_json,
            'ready_for_evaluation': left_gif and right_gif and left_json and right_json
        })
    
    file_df = pd.DataFrame(file_checks)
    summary_df = summary_df.merge(file_df, on='case_id')
    
    # Save final summary
    output_file = 'browserarena_both_successful_final_summary.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\nFinal summary saved to: {output_file}")
    
    # Also save cases ready for evaluation
    ready_df = summary_df[summary_df['ready_for_evaluation']]
    ready_file = 'browserarena_ready_for_gpt4o_evaluation.csv'
    ready_df.to_csv(ready_file, index=False)
    print(f"Cases ready for evaluation saved to: {ready_file} ({len(ready_df)} cases)")
    
    return summary_df

if __name__ == "__main__":
    from pathlib import Path
    df = create_summary()