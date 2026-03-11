#!/usr/bin/env python3
"""
Compute full V2 inter-annotator agreement with all pairwise combinations.
"""

import pandas as pd
import numpy as np
import json
from itertools import combinations
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

def compute_full_v2_agreements():
    # Load the V2 data
    df = pd.read_csv('/Users/davisbrown/browserarena/agreement-results/BrowserArenaAgreementv2_July_18_2025_11.01.csv')
    
    # Get V2 responses
    response_cols = [col for col in df.columns if col.startswith('What is your') and 'V2 Responses' in col]
    
    # Extract votes for each question
    questions = df['Case ID'].unique()
    
    # Dictionary to store all results
    results = {
        'pair_agreements': {},
        'question_agreements': {},
        'annotator_list': [],
        'overall_stats': {}
    }
    
    # For each question, get all annotator votes
    question_data = {}
    all_annotators = set()
    
    for _, row in df.iterrows():
        question = row['Case ID']
        if question not in question_data:
            question_data[question] = {}
        
        # Process V2 responses
        for col in response_cols:
            response = row[col]
            if pd.notna(response) and response in ['1', '2', '3']:
                # Extract annotator ID from column name
                annotator_id = col.split(' - ')[-1] if ' - ' in col else col
                question_data[question][annotator_id] = int(response)
                all_annotators.add(annotator_id)
    
    print(f"Found {len(all_annotators)} unique V2 annotators")
    print(f"Processing {len(questions)} questions")
    
    # Convert to list for consistent ordering
    annotator_list = sorted(list(all_annotators))
    results['annotator_list'] = annotator_list
    
    # Compute pairwise agreements for all annotator combinations
    print("Computing pairwise agreements...")
    pair_count = 0
    
    for ann1_idx, ann1 in enumerate(annotator_list):
        for ann2_idx in range(ann1_idx + 1, len(annotator_list)):
            ann2 = annotator_list[ann2_idx]
            
            # Count agreements across questions
            agreements = 0
            comparisons = 0
            
            for question, votes in question_data.items():
                if ann1 in votes and ann2 in votes:
                    comparisons += 1
                    if votes[ann1] == votes[ann2]:
                        agreements += 1
            
            if comparisons > 0:
                agreement_rate = agreements / comparisons
                results['pair_agreements'][f"{ann1}_{ann2}"] = agreement_rate
                pair_count += 1
        
        if ann1_idx % 10 == 0:
            print(f"  Processed {ann1_idx}/{len(annotator_list)} annotators...")
    
    print(f"Computed {pair_count} pairwise agreements")
    
    # Compute per-question agreement
    print("Computing per-question agreements...")
    
    for question, votes in question_data.items():
        if len(votes) < 2:
            continue
            
        # Get all pairs of annotators who voted on this question
        annotators_for_q = list(votes.keys())
        agreements = 0
        total_pairs = 0
        
        for i, ann1 in enumerate(annotators_for_q):
            for j in range(i + 1, len(annotators_for_q)):
                ann2 = annotators_for_q[j]
                total_pairs += 1
                if votes[ann1] == votes[ann2]:
                    agreements += 1
        
        if total_pairs > 0:
            results['question_agreements'][question] = agreements / total_pairs
    
    # Compute overall statistics
    all_agreements = list(results['pair_agreements'].values())
    
    results['overall_stats'] = {
        'overall_agreement': np.mean(all_agreements),
        'std_deviation': np.std(all_agreements),
        'min_agreement': np.min(all_agreements),
        'max_agreement': np.max(all_agreements),
        'num_annotators': len(annotator_list),
        'num_pairs': len(results['pair_agreements']),
        'num_questions': len(questions)
    }
    
    # Compute Fleiss' Kappa
    print("Computing Fleiss' Kappa...")
    
    # Create matrix for Fleiss' Kappa
    rater_matrix = []
    
    for question in questions:
        if question in question_data:
            votes = question_data[question]
            if len(votes) >= 2:
                # Count votes for each category
                vote_counts = [0, 0, 0]  # Agent 1, Agent 2, Tie
                for vote in votes.values():
                    vote_counts[vote - 1] += 1
                rater_matrix.append(vote_counts)
    
    if rater_matrix:
        try:
            kappa = fleiss_kappa(np.array(rater_matrix), method='fleiss')
            results['overall_stats']['fleiss_kappa'] = kappa
        except:
            results['overall_stats']['fleiss_kappa'] = None
    
    # Save results
    with open('/Users/davisbrown/browserarena/agreement-results/v2_inter_annotator_results_full.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSummary:")
    print(f"  Total annotators: {len(annotator_list)}")
    print(f"  Total pairwise combinations: {len(results['pair_agreements'])}")
    print(f"  Overall agreement: {results['overall_stats']['overall_agreement']:.3f}")
    print(f"  Standard deviation: {results['overall_stats']['std_deviation']:.3f}")
    
    return results

if __name__ == "__main__":
    compute_full_v2_agreements()