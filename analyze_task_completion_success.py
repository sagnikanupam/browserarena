#!/usr/bin/env python3
"""
Analyze task completion success by examining the trajectory outputs
to identify cases where both agents actually completed tasks correctly
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

def check_task_completion(task_id):
    """
    Check if a task was actually completed by examining the trajectory output
    """
    # Try to find the trajectory file
    trajectory_paths = [
        f"FastChat/prompts_and_outputs/{task_id}.json",
        f"FastChat/logs/{task_id}.txt"
    ]
    
    for path in trajectory_paths:
        if Path(path).exists():
            try:
                if path.endswith('.json'):
                    with open(path, 'r') as f:
                        data = json.load(f)
                        output = data.get('output', '')
                        
                        # Look for completion indicators
                        completion_indicators = [
                            'task completed',
                            'task complete',
                            'successfully completed',
                            'mission accomplished',
                            'goal achieved',
                            'task finished',
                            'done',
                            'success'
                        ]
                        
                        output_lower = output.lower()
                        for indicator in completion_indicators:
                            if indicator in output_lower:
                                return True, f"Found: '{indicator}'"
                                
                        # Check if the trajectory ended normally vs with errors
                        error_indicators = [
                            'error',
                            'failed',
                            'timeout',
                            'unable to',
                            'could not',
                            'parsing error'
                        ]
                        
                        has_errors = any(error in output_lower for error in error_indicators)
                        
                        # If no explicit success/failure, infer from trajectory structure
                        if 'step' in output_lower:
                            steps = output_lower.count('step')
                            if steps >= 3 and not has_errors:
                                return True, f"Completed {steps} steps without errors"
                        
                        return False, "No completion indicators found"
                        
                elif path.endswith('.txt'):
                    with open(path, 'r') as f:
                        content = f.read().lower()
                        if 'task completed' in content or 'success' in content:
                            return True, "Found completion in log"
                        
            except Exception as e:
                continue
    
    return None, "No trajectory file found"

def main():
    """Analyze cases where both agents actually completed tasks"""
    
    print("Loading evaluation results...")
    with open('gpt4o_all_tasks_evaluations/evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    valid_df = df[df['gpt4o_preference'] != 'error'].copy()
    
    print(f"Analyzing task completion for {len(valid_df)} cases...")
    
    # Check task completion for each case
    completion_results = []
    both_completed_cases = []
    
    for idx, row in valid_df.iterrows():
        left_task = row['left_task']
        right_task = row['right_task']
        
        left_completed, left_reason = check_task_completion(left_task)
        right_completed, right_reason = check_task_completion(right_task)
        
        completion_results.append({
            'case_id': row['case_id'],
            'left_task': left_task,
            'right_task': right_task,
            'left_completed': left_completed,
            'left_reason': left_reason,
            'right_completed': right_completed,
            'right_reason': right_reason,
            'both_completed': left_completed and right_completed,
            'original_vote': row['original_vote'],
            'gpt4o_preference': row['gpt4o_preference'],
            'gpt4o_confidence': row['gpt4o_confidence'],
            'gpt4o_reasoning': row['gpt4o_reasoning']
        })
        
        if left_completed and right_completed:
            both_completed_cases.append(row)
    
    # Save completion analysis
    completion_df = pd.DataFrame(completion_results)
    completion_df.to_csv('task_completion_analysis.csv', index=False)
    
    # Print summary
    total_cases = len(completion_results)
    left_completed_count = sum(1 for r in completion_results if r['left_completed'])
    right_completed_count = sum(1 for r in completion_results if r['right_completed'])  
    both_completed_count = sum(1 for r in completion_results if r['both_completed'])
    
    print(f"\n=== TASK COMPLETION ANALYSIS ===")
    print(f"Total cases analyzed: {total_cases}")
    print(f"Left agent completed: {left_completed_count} ({left_completed_count/total_cases*100:.1f}%)")
    print(f"Right agent completed: {right_completed_count} ({right_completed_count/total_cases*100:.1f}%)")
    print(f"Both agents completed: {both_completed_count} ({both_completed_count/total_cases*100:.1f}%)")
    
    if both_completed_count == 0:
        print("\n⚠️  WARNING: No cases found where both agents completed tasks!")
        print("This might be due to:")
        print("1. Missing trajectory files")
        print("2. Different completion indicators than expected")
        print("3. Tasks being marked as incomplete in the original data")
        
        # Show some examples of what we found
        print("\n=== SAMPLE COMPLETION REASONS ===")
        for result in completion_results[:10]:
            print(f"Case {result['case_id']}:")
            print(f"  Left: {result['left_completed']} - {result['left_reason']}")
            print(f"  Right: {result['right_completed']} - {result['right_reason']}")
            print()
    
    else:
        print(f"\n✅ Found {both_completed_count} cases where both agents completed tasks!")
        
        # Save the both-completed cases for further analysis
        both_completed_df = pd.DataFrame(both_completed_cases)
        both_completed_df.to_csv('both_agents_completed_cases.csv', index=False)
        
        # Quick analysis of these cases
        def normalize_vote(vote):
            if pd.isna(vote):
                return 'unknown'
            vote_str = str(vote).lower().strip()
            if any(pattern in vote_str for pattern in ['left', 'model a', 'a is better']):
                return 'left'
            if any(pattern in vote_str for pattern in ['right', 'model b', 'b is better']):
                return 'right'
            if any(pattern in vote_str for pattern in ['tie', 'equal', 'same']):
                return 'tie'
            return 'unknown'
        
        # Analyze the both-completed cases
        both_completed_df['normalized_human_vote'] = both_completed_df['original_vote'].apply(normalize_vote)
        both_completed_df['normalized_gpt4o_vote'] = both_completed_df['gpt4o_preference'].str.lower()
        
        clear_both_df = both_completed_df[both_completed_df['normalized_human_vote'].isin(['left', 'right', 'tie'])]
        
        if len(clear_both_df) > 0:
            agreements = clear_both_df['normalized_human_vote'] == clear_both_df['normalized_gpt4o_vote']
            agreement_rate = agreements.sum() / len(clear_both_df)
            
            print(f"\n=== BOTH AGENTS COMPLETED - AGREEMENT ANALYSIS ===")
            print(f"Cases with clear votes: {len(clear_both_df)}")
            print(f"Agreement rate: {agreements.sum()}/{len(clear_both_df)} ({agreement_rate*100:.1f}%)")
            
            # Vote distribution
            human_votes = Counter(clear_both_df['normalized_human_vote'])
            gpt4o_votes = Counter(clear_both_df['normalized_gpt4o_vote'])
            print(f"Human votes: {dict(human_votes)}")
            print(f"GPT-4o votes: {dict(gpt4o_votes)}")

if __name__ == "__main__":
    main()