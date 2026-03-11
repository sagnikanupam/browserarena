#!/usr/bin/env python3
"""
Create final CSV with VLM vs Human comparison results
"""

import pandas as pd
import json

def create_final_csv():
    """Create comprehensive CSV with all evaluation results"""
    
    # Load evaluation results
    with open('gpt4o_full_dataset_evaluations/evaluation_results.json', 'r') as f:
        all_results = json.load(f)
    
    # Process results
    processed_results = []
    for result in all_results:
        # Normalize votes for comparison
        human_vote = result['original_vote']
        gpt4_vote = result['gpt4o_preference']
        
        # Normalize human vote
        if pd.notna(human_vote):
            human_vote_str = str(human_vote).lower()
            if any(x in human_vote_str for x in ['left', 'a is better', 'model a']):
                human_vote_norm = 'Left'
            elif any(x in human_vote_str for x in ['right', 'b is better', 'model b']):
                human_vote_norm = 'Right'
            elif 'tie' in human_vote_str:
                human_vote_norm = 'Tie'
            else:
                human_vote_norm = 'Other'
        else:
            human_vote_norm = 'Unknown'
        
        # Normalize GPT-4 vote
        if gpt4_vote == 'error':
            gpt4_vote_norm = 'Error'
            agreement = 'N/A'
        elif gpt4_vote in ['left', 'Left']:
            gpt4_vote_norm = 'Left'
            agreement = 'Yes' if human_vote_norm == 'Left' else 'No'
        elif gpt4_vote in ['right', 'Right']:
            gpt4_vote_norm = 'Right'
            agreement = 'Yes' if human_vote_norm == 'Right' else 'No'
        elif gpt4_vote in ['tie', 'Tie']:
            gpt4_vote_norm = 'Tie'
            agreement = 'Yes' if human_vote_norm == 'Tie' else 'No'
        else:
            gpt4_vote_norm = 'Unknown'
            agreement = 'Unknown'
        
        processed_results.append({
            'case_id': result['case_id'],
            'left_task': result['left_task'],
            'right_task': result['right_task'],
            'human_vote_original': result['original_vote'],
            'human_vote_normalized': human_vote_norm,
            'gpt4o_vote': gpt4_vote_norm,
            'agreement': agreement,
            'gpt4o_confidence': result.get('gpt4o_confidence', 0),
            'gpt4o_reasoning': result.get('gpt4o_reasoning', ''),
            'evaluation_status': 'Success' if gpt4_vote != 'error' else 'Failed',
            'error_type': 'API Key Error' if 'API key' in result.get('gpt4o_reasoning', '') else 
                         'Missing Files' if 'Could not find GIF' in result.get('gpt4o_reasoning', '') else
                         'Other' if gpt4_vote == 'error' else 'None'
        })
    
    # Create DataFrame
    df = pd.DataFrame(processed_results)
    
    # Save full results
    df.to_csv('vlm_human_comparison_full_results.csv', index=False)
    
    # Create summary statistics
    valid_df = df[df['evaluation_status'] == 'Success']
    
    if len(valid_df) > 0:
        print("VLM vs HUMAN COMPARISON SUMMARY")
        print("="*50)
        print(f"Total cases: {len(df)}")
        print(f"Successfully evaluated: {len(valid_df)} ({len(valid_df)/len(df)*100:.1f}%)")
        print(f"Failed evaluations: {len(df) - len(valid_df)} ({(len(df) - len(valid_df))/len(df)*100:.1f}%)")
        
        print("\nAgreement Analysis (Valid Cases Only):")
        agreements = len(valid_df[valid_df['agreement'] == 'Yes'])
        print(f"Agreements: {agreements}/{len(valid_df)} ({agreements/len(valid_df)*100:.1f}%)")
        
        print("\nVote Distribution (Valid Cases):")
        print("\nHuman votes:")
        for vote, count in valid_df['human_vote_normalized'].value_counts().items():
            print(f"  {vote}: {count} ({count/len(valid_df)*100:.1f}%)")
        
        print("\nGPT-4o votes:")
        for vote, count in valid_df['gpt4o_vote'].value_counts().items():
            if vote != 'Error':
                print(f"  {vote}: {count} ({count/len(valid_df)*100:.1f}%)")
        
        # Save summary for valid cases only
        valid_df.to_csv('vlm_human_comparison_valid_only.csv', index=False)
        print(f"\nOutput files created:")
        print("  - vlm_human_comparison_full_results.csv (all {len(df)} cases)")
        print(f"  - vlm_human_comparison_valid_only.csv ({len(valid_df)} valid cases)")
    
    return df

if __name__ == "__main__":
    df = create_final_csv()