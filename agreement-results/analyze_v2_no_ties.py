import pandas as pd
import numpy as np
from collections import defaultdict
import json

def main():
    # Load the V2 data
    df = pd.read_csv('BrowserArenaAgreementv2_July_18_2025_11.01.csv')
    
    # Skip the second header row and any non-data rows
    # Data rows should have a valid date in StartDate
    df = df[df['StartDate'].str.contains('^2025-', na=False)]
    
    # Get the question columns (Q2 through Q22, excluding Q14, Q15, Q21 which don't exist)
    question_cols = [col for col in df.columns if col.startswith('Q') and col[1:].isdigit()]
    question_cols = sorted(question_cols, key=lambda x: int(x[1:]))
    
    print(f"Found {len(question_cols)} question columns: {question_cols}")
    
    # Convert vote columns to numeric
    for col in question_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Initialize results tracking
    results = {
        'total_responses': 0,
        'responses_with_ties': 0,
        'responses_without_ties': 0,
        'total_comparisons': 0,
        'comparisons_with_ties': 0,
        'comparisons_without_ties': 0,
        'agreements_no_ties': 0,
        'disagreements_no_ties': 0,
        'per_question_stats': defaultdict(lambda: {
            'total': 0,
            'with_ties': 0,
            'without_ties': 0,
            'agreements': 0,
            'disagreements': 0
        })
    }
    
    # Group by question to analyze inter-annotator agreement
    for question in question_cols:
        question_data = df[['PROLIFIC_PID', question]].copy()
        question_data = question_data.dropna()
        
        # Group by annotator to find questions with multiple responses
        annotator_groups = question_data.groupby('PROLIFIC_PID')
        
        # For inter-annotator agreement, we need to compare different annotators' responses
        # to the same question
        unique_annotators = question_data['PROLIFIC_PID'].unique()
        
        # Create all possible pairs of annotators
        for i in range(len(unique_annotators)):
            for j in range(i+1, len(unique_annotators)):
                annotator1 = unique_annotators[i]
                annotator2 = unique_annotators[j]
                
                # Get votes from each annotator for this question
                vote1_data = question_data[question_data['PROLIFIC_PID'] == annotator1][question]
                vote2_data = question_data[question_data['PROLIFIC_PID'] == annotator2][question]
                
                if len(vote1_data) > 0 and len(vote2_data) > 0:
                    vote1 = vote1_data.iloc[0]
                    vote2 = vote2_data.iloc[0]
                    
                    results['total_comparisons'] += 1
                    results['per_question_stats'][question]['total'] += 1
                    
                    # Check if either vote is a tie (3)
                    if vote1 == 3 or vote2 == 3:
                        results['comparisons_with_ties'] += 1
                        results['per_question_stats'][question]['with_ties'] += 1
                    else:
                        # Both votes are decisive (1 or 2)
                        results['comparisons_without_ties'] += 1
                        results['per_question_stats'][question]['without_ties'] += 1
                        
                        if vote1 == vote2:
                            results['agreements_no_ties'] += 1
                            results['per_question_stats'][question]['agreements'] += 1
                        else:
                            results['disagreements_no_ties'] += 1
                            results['per_question_stats'][question]['disagreements'] += 1
    
    # Calculate overall statistics
    if results['comparisons_without_ties'] > 0:
        agreement_rate = results['agreements_no_ties'] / results['comparisons_without_ties']
    else:
        agreement_rate = 0
    
    # Print summary
    print("\n=== V2 Inter-Annotator Agreement Analysis (No Ties) ===")
    print(f"\nTotal annotator pairs compared: {results['total_comparisons']}")
    print(f"Comparisons with at least one tie vote: {results['comparisons_with_ties']} ({results['comparisons_with_ties']/results['total_comparisons']*100:.1f}%)")
    print(f"Comparisons with only decisive votes (1 or 2): {results['comparisons_without_ties']} ({results['comparisons_without_ties']/results['total_comparisons']*100:.1f}%)")
    
    print(f"\n--- Agreement on Decisive Votes Only ---")
    print(f"Agreements: {results['agreements_no_ties']}")
    print(f"Disagreements: {results['disagreements_no_ties']}")
    print(f"Agreement Rate: {agreement_rate*100:.2f}%")
    
    # Per-question analysis
    print("\n--- Per Question Analysis (No Ties) ---")
    print(f"{'Question':<10} {'Total':<8} {'No Ties':<10} {'Agree':<8} {'Disagree':<10} {'Agreement %':<12}")
    print("-" * 60)
    
    for question in sorted(results['per_question_stats'].keys(), 
                          key=lambda x: int(x[1:])):
        stats = results['per_question_stats'][question]
        if stats['without_ties'] > 0:
            q_agreement_rate = stats['agreements'] / stats['without_ties'] * 100
        else:
            q_agreement_rate = 0
        
        print(f"{question:<10} {stats['total']:<8} {stats['without_ties']:<10} "
              f"{stats['agreements']:<8} {stats['disagreements']:<10} {q_agreement_rate:<12.1f}")
    
    # Save detailed results
    output = {
        'summary': {
            'total_comparisons': results['total_comparisons'],
            'comparisons_with_ties': results['comparisons_with_ties'],
            'comparisons_without_ties': results['comparisons_without_ties'],
            'agreements_no_ties': results['agreements_no_ties'],
            'disagreements_no_ties': results['disagreements_no_ties'],
            'agreement_rate_no_ties': agreement_rate
        },
        'per_question': dict(results['per_question_stats'])
    }
    
    with open('v2_no_ties_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nDetailed results saved to v2_no_ties_analysis.json")

if __name__ == "__main__":
    main()