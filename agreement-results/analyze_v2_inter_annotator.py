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
    
    # Skip the second header row
    df = df.iloc[1:]
    
    # Get prolific IDs (annotators)
    prolific_id_col = 'PROLIFIC_PID' if 'PROLIFIC_PID' in df.columns else df.columns[-1]
    
    # Find columns that contain the actual voting data
    # These columns typically have the pattern Q[number] and contain the votes
    vote_columns = []
    question_mapping = {}
    
    print("\nAnalyzing column structure...")
    for col in df.columns:
        if col.startswith('Q') and col[1:].replace('.','').isdigit():
            # Skip Q1 which seems to be the task description
            if col not in ['Q1']:
                vote_columns.append(col)
    
    # Also check for columns like Q3, Q4, Q5... which contain votes
    # Typically these are in sequence after Q2
    vote_columns = [col for col in df.columns if col in ['Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 
                                                          'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 
                                                          'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q22']]
    
    print(f"Found {len(vote_columns)} potential voting columns: {vote_columns}")
    
    # Filter out rows with missing prolific IDs and test responses
    df = df[df[prolific_id_col].notna()]
    df = df[~df[prolific_id_col].str.contains('test', case=False, na=False)]
    
    # Get unique annotators
    annotators = df[prolific_id_col].unique()
    print(f"\nFound {len(annotators)} V2 annotators")
    
    # Prepare data structure for agreement calculation
    # Structure: {question_id: {annotator_id: vote}}
    votes_by_question = defaultdict(lambda: defaultdict(str))
    
    for idx, row in df.iterrows():
        annotator = row[prolific_id_col]
        
        for col in vote_columns:
            if pd.notna(row[col]):
                # Extract vote from the response
                vote_text = str(row[col]).strip()
                
                # Determine the vote (Agent 1, Agent 2, Tie, etc.)
                vote = None
                if 'Agent 1' in vote_text:
                    vote = 'Agent 1'
                elif 'Agent 2' in vote_text:
                    vote = 'Agent 2'
                elif 'Tie' in vote_text or 'Neither' in vote_text:
                    vote = 'Tie'
                
                if vote:
                    votes_by_question[col][annotator] = vote
    
    # Calculate pairwise agreement
    print("\nCalculating pairwise inter-annotator agreement...")
    
    # Get all pairs of annotators
    annotator_pairs = []
    for i, ann1 in enumerate(annotators):
        for j in range(i+1, len(annotators)):
            ann2 = annotators[j]
            annotator_pairs.append((ann1, ann2))
    
    print(f"Total annotator pairs: {len(annotator_pairs)}")
    
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
            print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} pairs")
        
        # Show top and bottom pairs
        sorted_pairs = sorted(pair_agreements.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 5 most agreeing pairs:")
        for i, ((ann1, ann2), agreement) in enumerate(sorted_pairs[:5]):
            count = pair_counts[(ann1, ann2)]
            print(f"  {i+1}. {ann1[:8]}... & {ann2[:8]}...: {agreement:.3f} ({count} questions)")
        
        print("\nBottom 5 least agreeing pairs:")
        for i, ((ann1, ann2), agreement) in enumerate(sorted_pairs[-5:]):
            count = pair_counts[(ann1, ann2)]
            print(f"  {i+1}. {ann1[:8]}... & {ann2[:8]}...: {agreement:.3f} ({count} questions)")
        
        # Per-question agreement
        print("\nPer-question agreement rates:")
        question_agreements = {}
        
        for question in vote_columns[:10]:  # Show first 10 questions
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
        for question, agreement in sorted_questions:
            print(f"  {question}: {agreement:.3f}")
        
        # Save results
        results = {
            'overall_agreement': overall_agreement,
            'std_deviation': std_agreement,
            'min_agreement': min(all_agreements),
            'max_agreement': max(all_agreements),
            'num_annotators': len(annotators),
            'num_pairs': len(annotator_pairs),
            'pair_agreements': {f"{k[0]}_{k[1]}": v for k, v in pair_agreements.items()},
            'question_agreements': question_agreements
        }
        
        with open('/Users/davisbrown/browserarena/agreement-results/v2_inter_annotator_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to v2_inter_annotator_results.json")
    else:
        print("\nNo valid annotator pairs found with common questions!")
    
    # Analyze vote distribution
    print("\n=== Vote Distribution Analysis ===")
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