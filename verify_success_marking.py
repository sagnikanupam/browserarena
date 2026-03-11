#!/usr/bin/env python3
"""
Verify the success marking by checking specific disagreement cases
"""

import pandas as pd
import json

def main():
    # Load the original analyzed CSV
    df_csv = pd.read_csv('agreement-results/data_updated/ProlificBrowserArenaGIFFeedbackForm_May 14, 2025_22.32_analyzed.csv')
    
    # Load our evaluation results  
    with open('gpt4o_all_tasks_evaluations/evaluation_results.json', 'r') as f:
        eval_results = json.load(f)
    
    # Load the mapping from full_dataset_cases.json
    with open('full_dataset_cases.json', 'r') as f:
        full_data = json.load(f)
    
    print("=== VERIFYING SUCCESS MARKING ===\n")
    
    # Create a mapping from response_id to task IDs
    response_to_tasks = {}
    for case in full_data['all_cases']:
        response_to_tasks[case['response_id']] = {
            'left_task': case['left_task'],
            'right_task': case['right_task'],
            'left_correct': case['left_correct'],
            'right_correct': case['right_correct']
        }
    
    # Check some specific disagreement cases
    disagreement_cases = [
        'R_5oaK9CRQgLCO0GM',  # Right only succeeded but human voted Left
        'R_1H3k1Z3Vz9Pn9Vc',  # Left only succeeded but human voted Right  
        'R_3Bb5xQSlFbCAyB5'   # Both failed but there was a vote
    ]
    
    for response_id in disagreement_cases:
        if response_id in response_to_tasks:
            case_info = response_to_tasks[response_id]
            
            # Find corresponding row in CSV
            csv_row = df_csv[df_csv['ResponseId'] == response_id]
            
            if not csv_row.empty:
                row = csv_row.iloc[0]
                
                print(f"Response ID: {response_id}")
                print(f"Left task: {case_info['left_task']}")
                print(f"Right task: {case_info['right_task']}")
                print(f"JSON success flags: Left={case_info['left_correct']}, Right={case_info['right_correct']}")
                print(f"CSV success flags: Left={row['CorrectLeftDataEntry']}, Right={row['CorrectRightDataEntry']}")
                print(f"CSV steps: Left valid={row['num_valid_steps_left']}/{row['max_steps_left']}, Right valid={row['num_valid_steps_right']}/{row['max_steps_right']}")
                print(f"Human vote: {row.get('Q52', 'Unknown')}")
                print()
    
    # Overall verification: check if JSON flags match CSV flags
    print("=== OVERALL VERIFICATION ===")
    matches = 0
    total = 0
    
    for case in full_data['all_cases']:
        response_id = case['response_id']
        csv_row = df_csv[df_csv['ResponseId'] == response_id]
        
        if not csv_row.empty:
            row = csv_row.iloc[0]
            total += 1
            
            csv_left = row['CorrectLeftDataEntry']
            csv_right = row['CorrectRightDataEntry']
            json_left = case['left_correct']
            json_right = case['right_correct']
            
            if csv_left == json_left and csv_right == json_right:
                matches += 1
            else:
                print(f"MISMATCH {response_id}: CSV({csv_left},{csv_right}) vs JSON({json_left},{json_right})")
    
    print(f"\nSuccess flag verification: {matches}/{total} match ({matches/total*100:.1f}%)")
    
    # Now check what "success" actually means by looking at the step counting logic
    print("\n=== SUCCESS DEFINITION ANALYSIS ===")
    print("Success = (num_valid_steps == max_steps)")
    print("This measures annotation completeness, NOT task completion success!")
    print()
    
    # Sample some cases to show the difference
    sample_cases = df_csv.head(10)
    for _, row in sample_cases.iterrows():
        if pd.notna(row['num_valid_steps_left']) and pd.notna(row['max_steps_left']):
            left_success = row['num_valid_steps_left'] == row['max_steps_left']
            right_success = row['num_valid_steps_right'] == row['max_steps_right']
            
            print(f"ResponseId {row['ResponseId']}:")
            print(f"  Left: {row['num_valid_steps_left']}/{row['max_steps_left']} steps annotated -> Success={left_success}")
            print(f"  Right: {row['num_valid_steps_right']}/{row['max_steps_right']} steps annotated -> Success={right_success}")
            print(f"  Human vote: {row.get('Q52', 'Unknown')}")
            print()

if __name__ == "__main__":
    main()