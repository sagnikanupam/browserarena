#!/usr/bin/env python3
"""
Compute complete V2 inter-annotator agreement with all pairwise combinations.
"""

import pandas as pd
import numpy as np
import json
from itertools import combinations
from statsmodels.stats.inter_rater import fleiss_kappa

def compute_complete_v2_agreements():
    # Read the V2 survey data
    print("Reading V2 survey data...")
    df = pd.read_csv('/Users/davisbrown/browserarena/agreement-results/BrowserArenaAgreementv2_July_18_2025_11.01.csv')
    
    # Skip header rows and get actual data
    data_df = df.iloc[2:]  # Skip both header rows
    
    # Get prolific IDs (annotators)
    prolific_id_col = 'PROLIFIC_PID' if 'PROLIFIC_PID' in df.columns else df.columns[-1]
    
    # Find columns that contain voting data
    vote_columns = [col for col in df.columns if col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 
                                                          'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 
                                                          'Q13', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q22']]
    
    print(f"Found {len(vote_columns)} voting columns: {vote_columns}")
    
    # Filter out rows with missing prolific IDs
    data_df = data_df[data_df[prolific_id_col].notna()]
    
    # Remove any test responses
    if prolific_id_col in data_df.columns:
        data_df = data_df[~data_df[prolific_id_col].astype(str).str.contains('test', case=False, na=False)]
    
    # Get unique annotators
    annotators = list(data_df[prolific_id_col].unique())
    n_annotators = len(annotators)
    print(f"\nFound {n_annotators} V2 annotators")
    
    # Create a dictionary to store votes by annotator and question
    votes_dict = {}
    for idx, row in data_df.iterrows():
        annotator = row[prolific_id_col]
        votes_dict[annotator] = {}
        
        for q_col in vote_columns:
            vote = row[q_col]
            # Convert to string and check if it's a valid vote
            vote_str = str(vote).strip()
            if vote_str in ['1', '2', '3', '1.0', '2.0', '3.0']:
                votes_dict[annotator][q_col] = int(float(vote_str))
    
    # Initialize results dictionary
    results = {
        'pair_agreements': {},
        'question_agreements': {},
        'annotator_list': annotators,
        'overall_stats': {},
        'agreement_matrix': {}
    }
    
    # Compute pairwise agreements for ALL annotator combinations
    print(f"\nComputing pairwise agreements for {n_annotators * (n_annotators - 1) // 2} pairs...")
    
    total_pairs = 0
    all_agreement_rates = []
    
    for i, ann1 in enumerate(annotators):
        if i % 10 == 0:
            print(f"  Processing annotator {i+1}/{n_annotators}...")
        
        for j in range(i + 1, n_annotators):
            ann2 = annotators[j]
            
            # Count agreements across questions
            agreements = 0
            comparisons = 0
            
            for question in vote_columns:
                if question in votes_dict[ann1] and question in votes_dict[ann2]:
                    comparisons += 1
                    if votes_dict[ann1][question] == votes_dict[ann2][question]:
                        agreements += 1
            
            if comparisons > 0:
                agreement_rate = agreements / comparisons
                results['pair_agreements'][f"{ann1}_{ann2}"] = agreement_rate
                all_agreement_rates.append(agreement_rate)
                total_pairs += 1
    
    print(f"Computed {total_pairs} pairwise agreements")
    
    # Compute per-question agreement
    print("\nComputing per-question agreements...")
    
    for question in vote_columns:
        agreements = 0
        total_comparisons = 0
        
        # Get all annotators who voted on this question
        annotators_for_q = [ann for ann in annotators if question in votes_dict[ann]]
        
        # Compute pairwise agreements for this question
        for i, ann1 in enumerate(annotators_for_q):
            for j in range(i + 1, len(annotators_for_q)):
                ann2 = annotators_for_q[j]
                total_comparisons += 1
                if votes_dict[ann1][question] == votes_dict[ann2][question]:
                    agreements += 1
        
        if total_comparisons > 0:
            results['question_agreements'][question] = agreements / total_comparisons
    
    # Compute overall statistics
    results['overall_stats'] = {
        'overall_agreement': np.mean(all_agreement_rates),
        'std_deviation': np.std(all_agreement_rates),
        'min_agreement': np.min(all_agreement_rates),
        'max_agreement': np.max(all_agreement_rates),
        'num_annotators': n_annotators,
        'num_pairs': total_pairs,
        'num_valid_pairs': total_pairs,
        'num_questions': len(vote_columns)
    }
    
    # Compute Fleiss' Kappa
    print("\nComputing Fleiss' Kappa...")
    
    # Create matrix for Fleiss' Kappa
    rater_matrix = []
    
    for question in vote_columns:
        # Count votes for each category
        vote_counts = [0, 0, 0]  # Agent 1, Agent 2, Tie
        
        for annotator in annotators:
            if question in votes_dict[annotator]:
                vote = votes_dict[annotator][question]
                vote_counts[vote - 1] += 1
        
        if sum(vote_counts) >= 2:  # At least 2 raters
            rater_matrix.append(vote_counts)
    
    if rater_matrix:
        try:
            kappa = fleiss_kappa(np.array(rater_matrix), method='fleiss')
            results['overall_stats']['fleiss_kappa'] = kappa
        except Exception as e:
            print(f"Error computing Fleiss' Kappa: {e}")
            results['overall_stats']['fleiss_kappa'] = None
    
    # Create agreement matrix for top annotators (for visualization)
    print("\nCreating agreement matrix for visualization...")
    
    # Sort annotators by number of questions they answered
    annotator_counts = {}
    for ann in annotators:
        annotator_counts[ann] = len(votes_dict[ann])
    
    # Get top 20 most active annotators
    top_annotators = sorted(annotator_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    top_annotators = [ann[0] for ann in top_annotators]
    
    # Create agreement matrix
    n_top = len(top_annotators)
    agreement_matrix = np.zeros((n_top, n_top))
    
    for i, ann1 in enumerate(top_annotators):
        for j, ann2 in enumerate(top_annotators):
            if i != j:
                key1 = f"{ann1}_{ann2}"
                key2 = f"{ann2}_{ann1}"
                
                if key1 in results['pair_agreements']:
                    agreement_matrix[i, j] = results['pair_agreements'][key1]
                elif key2 in results['pair_agreements']:
                    agreement_matrix[i, j] = results['pair_agreements'][key2]
            else:
                agreement_matrix[i, j] = 1.0  # Perfect agreement with self
    
    # Store matrix data
    results['agreement_matrix'] = {
        'matrix': agreement_matrix.tolist(),
        'annotators': top_annotators
    }
    
    # Save complete results
    output_file = '/Users/davisbrown/browserarena/agreement-results/v2_inter_annotator_results_complete.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved complete results to {output_file}")
    
    print(f"\nSummary:")
    print(f"  Total annotators: {n_annotators}")
    print(f"  Total pairwise combinations: {total_pairs}")
    print(f"  Overall agreement: {results['overall_stats']['overall_agreement']:.3f}")
    print(f"  Standard deviation: {results['overall_stats']['std_deviation']:.3f}")
    print(f"  Min agreement: {results['overall_stats']['min_agreement']:.3f}")
    print(f"  Max agreement: {results['overall_stats']['max_agreement']:.3f}")
    if results['overall_stats']['fleiss_kappa'] is not None:
        print(f"  Fleiss' Kappa: {results['overall_stats']['fleiss_kappa']:.3f}")
    
    return results

if __name__ == "__main__":
    compute_complete_v2_agreements()