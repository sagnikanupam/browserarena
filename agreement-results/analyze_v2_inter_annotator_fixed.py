#!/usr/bin/env python3
"""
Analyze V2 human annotator data to compute inter-annotator agreement within the V2 group.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import json

def main():
    # Read the V2 survey data
    print("Reading V2 survey data...")
    df = pd.read_csv('/Users/davisbrown/browserarena/agreement-results/BrowserArenaAgreementv2_July_18_2025_11.01.csv')
    
    # The first row contains the questions/tasks
    questions_row = df.iloc[0]
    
    # Skip the header rows
    data_df = df.iloc[2:]  # Skip both header rows
    
    # Get prolific IDs (annotators)
    prolific_id_col = 'PROLIFIC_PID' if 'PROLIFIC_PID' in df.columns else df.columns[-1]
    
    # Find columns that contain voting data (Q3, Q4, etc.)
    vote_columns = [col for col in df.columns if col in ['Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 
                                                          'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 
                                                          'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q22']]
    
    print(f"Found {len(vote_columns)} voting columns: {vote_columns}")
    
    # Filter out rows with missing prolific IDs
    data_df = data_df[data_df[prolific_id_col].notna()]
    
    # Remove any test responses
    if prolific_id_col in data_df.columns:
        data_df = data_df[~data_df[prolific_id_col].astype(str).str.contains('test', case=False, na=False)]
    
    # Get unique annotators
    annotators = data_df[prolific_id_col].unique()
    print(f"\nFound {len(annotators)} V2 annotators")
    
    # Map numeric values to votes
    # Typically: 1 = Agent 1, 2 = Agent 2, 3 = Tie/Neither
    vote_mapping = {
        '1': 'Agent 1',
        '2': 'Agent 2', 
        '3': 'Tie',
        '1.0': 'Agent 1',
        '2.0': 'Agent 2',
        '3.0': 'Tie'
    }
    
    # Prepare data structure for agreement calculation
    # Structure: {question_id: {annotator_id: vote}}
    votes_by_question = defaultdict(lambda: defaultdict(str))
    
    for idx, row in data_df.iterrows():
        annotator = row[prolific_id_col]
        
        for col in vote_columns:
            if pd.notna(row[col]):
                # Get the vote value
                vote_value = str(row[col]).strip()
                
                # Map to standardized vote
                vote = vote_mapping.get(vote_value)
                
                if vote:
                    votes_by_question[col][annotator] = vote
    
    # Print sample of collected votes
    print("\nSample of collected votes:")
    for i, (question, votes) in enumerate(list(votes_by_question.items())[:3]):
        print(f"\n{question}:")
        for j, (annotator, vote) in enumerate(list(votes.items())[:3]):
            print(f"  {annotator[:10]}...: {vote}")
    
    # Calculate pairwise agreement
    print("\n\nCalculating pairwise inter-annotator agreement...")
    
    # Get all pairs of annotators
    annotator_pairs = []
    for i, ann1 in enumerate(annotators):
        for j in range(i+1, len(annotators)):
            ann2 = annotators[j]
            annotator_pairs.append((ann1, ann2))
    
    print(f"Total annotator pairs to analyze: {len(annotator_pairs)}")
    
    # Calculate agreement for each pair
    pair_agreements = {}
    pair_counts = {}
    
    for ann1, ann2 in annotator_pairs:
        agreements = []
        common_questions = []
        
        for question in votes_by_question:
            if ann1 in votes_by_question[question] and ann2 in votes_by_question[question]:
                vote1 = votes_by_question[question][ann1]
                vote2 = votes_by_question[question][ann2]
                
                if vote1 and vote2:  # Both have valid votes
                    common_questions.append(question)
                    agreements.append(1 if vote1 == vote2 else 0)
        
        if agreements:
            agreement_rate = np.mean(agreements)
            pair_agreements[(ann1, ann2)] = agreement_rate
            pair_counts[(ann1, ann2)] = len(agreements)
    
    print(f"Valid pairs with common questions: {len(pair_agreements)}")
    
    # Calculate overall statistics
    if pair_agreements:
        all_agreements = list(pair_agreements.values())
        overall_agreement = np.mean(all_agreements)
        std_agreement = np.std(all_agreements)
        
        print(f"\n=== V2 Inter-Annotator Agreement Results ===")
        print(f"Overall V2 agreement rate: {overall_agreement:.3f}")
        print(f"Standard deviation: {std_agreement:.3f}")
        print(f"Min agreement: {min(all_agreements):.3f}")
        print(f"Max agreement: {max(all_agreements):.3f}")
        
        # Show distribution
        print("\nAgreement distribution:")
        bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(all_agreements, bins=bins)
        for i in range(len(bins)-1):
            print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} pairs ({hist[i]/len(all_agreements)*100:.1f}%)")
        
        # Show top and bottom pairs
        sorted_pairs = sorted(pair_agreements.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 5 most agreeing pairs:")
        for i, ((ann1, ann2), agreement) in enumerate(sorted_pairs[:5]):
            count = pair_counts[(ann1, ann2)]
            print(f"  {i+1}. {ann1[:12]}... & {ann2[:12]}...: {agreement:.3f} ({count} questions)")
        
        print("\nBottom 5 least agreeing pairs:")
        for i, ((ann1, ann2), agreement) in enumerate(sorted_pairs[-5:]):
            count = pair_counts[(ann1, ann2)]
            print(f"  {i+1}. {ann1[:12]}... & {ann2[:12]}...: {agreement:.3f} ({count} questions)")
        
        # Per-question agreement
        print("\n\nPer-question agreement rates:")
        question_agreements = {}
        
        for question in vote_columns:
            if question in votes_by_question:
                votes = votes_by_question[question]
                annotator_list = list(votes.keys())
                
                if len(annotator_list) >= 2:
                    agreements = []
                    for i, ann1 in enumerate(annotator_list):
                        for j in range(i+1, len(annotator_list)):
                            ann2 = annotator_list[j]
                            if votes[ann1] and votes[ann2]:
                                agreements.append(1 if votes[ann1] == votes[ann2] else 0)
                    
                    if agreements:
                        question_agreements[question] = np.mean(agreements)
        
        sorted_questions = sorted(question_agreements.items(), key=lambda x: x[1], reverse=True)
        print(f"Questions analyzed: {len(sorted_questions)}")
        print("\nTop 5 questions with highest agreement:")
        for question, agreement in sorted_questions[:5]:
            print(f"  {question}: {agreement:.3f}")
        
        print("\nBottom 5 questions with lowest agreement:")
        for question, agreement in sorted_questions[-5:]:
            print(f"  {question}: {agreement:.3f}")
        
        # Calculate Fleiss' Kappa
        print("\n\nCalculating Fleiss' Kappa...")
        
        # Create matrix for Fleiss' Kappa
        questions_list = list(votes_by_question.keys())
        n_questions = len(questions_list)
        n_categories = 3  # Agent 1, Agent 2, Tie
        
        # Count matrix: rows = questions, columns = categories
        count_matrix = np.zeros((n_questions, n_categories))
        
        for q_idx, question in enumerate(questions_list):
            votes = votes_by_question[question]
            for vote in votes.values():
                if vote == 'Agent 1':
                    count_matrix[q_idx, 0] += 1
                elif vote == 'Agent 2':
                    count_matrix[q_idx, 1] += 1
                elif vote == 'Tie':
                    count_matrix[q_idx, 2] += 1
        
        # Calculate Fleiss' Kappa
        n_raters = count_matrix.sum(axis=1).max()  # Max number of raters for any question
        
        if n_raters >= 2:
            # P_i (proportion of agreeing pairs for each subject)
            P_i = np.zeros(n_questions)
            for i in range(n_questions):
                n_ij = count_matrix[i]
                n_i = n_ij.sum()
                if n_i >= 2:
                    P_i[i] = (np.sum(n_ij**2) - n_i) / (n_i * (n_i - 1))
            
            # P_bar (mean of P_i)
            valid_P_i = P_i[P_i > 0]
            if len(valid_P_i) > 0:
                P_bar = np.mean(valid_P_i)
                
                # P_e (chance agreement)
                p_j = count_matrix.sum(axis=0) / count_matrix.sum()
                P_e = np.sum(p_j**2)
                
                # Fleiss' Kappa
                if P_e < 1:
                    kappa = (P_bar - P_e) / (1 - P_e)
                    print(f"Fleiss' Kappa: {kappa:.3f}")
                    
                    # Interpretation
                    if kappa < 0:
                        interpretation = "Poor agreement"
                    elif kappa <= 0.20:
                        interpretation = "Slight agreement"
                    elif kappa <= 0.40:
                        interpretation = "Fair agreement"
                    elif kappa <= 0.60:
                        interpretation = "Moderate agreement"
                    elif kappa <= 0.80:
                        interpretation = "Substantial agreement"
                    else:
                        interpretation = "Almost perfect agreement"
                    
                    print(f"Interpretation: {interpretation}")
        
        # Save results
        results = {
            'overall_agreement': overall_agreement,
            'std_deviation': std_agreement,
            'min_agreement': min(all_agreements),
            'max_agreement': max(all_agreements),
            'num_annotators': len(annotators),
            'num_pairs': len(annotator_pairs),
            'num_valid_pairs': len(pair_agreements),
            'pair_agreements': {f"{k[0]}_{k[1]}": v for k, v in list(pair_agreements.items())[:50]},  # Limit to first 50
            'question_agreements': question_agreements,
            'fleiss_kappa': kappa if 'kappa' in locals() else None
        }
        
        with open('/Users/davisbrown/browserarena/agreement-results/v2_inter_annotator_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to v2_inter_annotator_results.json")
    else:
        print("\nNo valid annotator pairs found with common questions!")
    
    # Analyze vote distribution
    print("\n\n=== Vote Distribution Analysis ===")
    all_votes = defaultdict(int)
    
    for question in votes_by_question:
        for annotator, vote in votes_by_question[question].items():
            if vote:
                all_votes[vote] += 1
    
    total_votes = sum(all_votes.values())
    if total_votes > 0:
        print(f"Total votes: {total_votes}")
        for vote, count in sorted(all_votes.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_votes) * 100
            print(f"  {vote}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()