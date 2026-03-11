#!/usr/bin/env python3
"""
Create a cleaned CSV with normalized vote labels for evaluation
"""

import pandas as pd
import re

def normalize_vote(vote):
    """Normalize vote labels to consistent format"""
    if pd.isna(vote):
        return 'Unknown'
    
    vote_str = str(vote).strip().lower()
    
    # Map various formats to standard labels
    if any(x in vote_str for x in ['left', 'a is better', 'model a', '(a)']):
        return 'Left'
    elif any(x in vote_str for x in ['right', 'b is better', 'model b', '(b)']):
        return 'Right'
    elif 'tie' in vote_str:
        return 'Tie'
    elif vote_str in ['a', 'model a']:
        return 'Left'
    elif vote_str in ['b', 'model b']:
        return 'Right'
    else:
        # Default based on content
        if 'left' in vote_str:
            return 'Left'
        elif 'right' in vote_str:
            return 'Right'
        else:
            return 'Unknown'

def create_cleaned_csv():
    """Create cleaned CSV with normalized votes"""
    
    # Load the original CSV
    df = pd.read_csv('full_dataset_both_agents_successful_for_evaluation.csv')
    
    print(f"Original dataset: {len(df)} cases")
    print("\nOriginal vote distribution:")
    print(df['original_vote'].value_counts())
    
    # Normalize votes
    df['normalized_vote'] = df['original_vote'].apply(normalize_vote)
    
    print("\nNormalized vote distribution:")
    print(df['normalized_vote'].value_counts())
    
    # Filter out unknown votes
    df_clean = df[df['normalized_vote'] != 'Unknown'].copy()
    
    print(f"\nAfter cleaning: {len(df_clean)} cases with valid votes")
    
    # Reorder columns for clarity
    column_order = [
        'case_id',
        'response_id',
        'left_task_id',
        'right_task_id',
        'normalized_vote',
        'original_vote',
        'task_description',
        'left_gif_path',
        'right_gif_path',
        'left_prompt_output_path',
        'right_prompt_output_path',
        'gpt4o_preference',
        'gpt4o_confidence',
        'gpt4o_reasoning',
        'gpt4o_agrees_with_original',
        'duration_seconds',
        'prolific_pid',
        'recorded_date'
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df_clean.columns]
    df_clean = df_clean[column_order]
    
    # Save cleaned CSV
    output_file = 'full_dataset_both_successful_cleaned_for_gpt4o_eval.csv'
    df_clean.to_csv(output_file, index=False)
    print(f"\nSaved cleaned dataset to: {output_file}")
    
    # Create sample for testing
    sample_df = df_clean.head(20)
    sample_file = 'sample_20_cases_for_gpt4o_testing.csv'
    sample_df.to_csv(sample_file, index=False)
    print(f"Created sample file: {sample_file}")
    
    # Summary statistics
    print("\nFinal Summary:")
    print(f"Total cases for evaluation: {len(df_clean)}")
    vote_counts = df_clean['normalized_vote'].value_counts()
    for vote, count in vote_counts.items():
        print(f"  {vote}: {count} ({count/len(df_clean)*100:.1f}%)")
    
    return df_clean

if __name__ == "__main__":
    df_clean = create_cleaned_csv()