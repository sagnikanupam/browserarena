#!/usr/bin/env python3
"""
Analyze the evaluation results we have so far
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results():
    """Analyze the evaluation results collected so far"""
    
    # Load evaluation results
    results_file = Path('gpt4o_full_dataset_evaluations/evaluation_results.json')
    if not results_file.exists():
        print("No evaluation results found")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Total evaluations in file: {len(results)}")
    
    # Filter out error cases
    valid_results = [r for r in results if r['gpt4o_preference'] != 'error']
    error_results = [r for r in results if r['gpt4o_preference'] == 'error']
    
    print(f"\nValid evaluations: {len(valid_results)}")
    print(f"Error evaluations: {len(error_results)}")
    
    if not valid_results:
        print("\nNo valid evaluations to analyze")
        # Analyze errors
        if error_results:
            print("\nError analysis:")
            error_types = {}
            for r in error_results:
                error_msg = r.get('gpt4o_reasoning', 'Unknown error')
                if 'missing_files' in r:
                    error_type = 'Missing GIF files'
                elif 'API key' in error_msg:
                    error_type = 'API key error'
                elif 'openai.ChatCompletion' in error_msg:
                    error_type = 'OpenAI library version error'
                else:
                    error_type = 'Other error'
                
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            print("\nError breakdown:")
            for error_type, count in error_types.items():
                print(f"  {error_type}: {count}")
        return
    
    # Analyze valid results
    print("\n" + "="*60)
    print("ANALYSIS OF VALID EVALUATIONS")
    print("="*60)
    
    # Normalize votes for comparison
    def normalize_vote(vote):
        if pd.isna(vote) or vote == '':
            return 'Unknown'
        vote_str = str(vote).strip().lower()
        if any(x in vote_str for x in ['left', 'a is better', 'model a']):
            return 'Left'
        elif any(x in vote_str for x in ['right', 'b is better', 'model b']):
            return 'Right'
        elif 'tie' in vote_str:
            return 'Tie'
        else:
            return 'Unknown'
    
    # Normalize all votes
    for r in valid_results:
        r['normalized_original_vote'] = normalize_vote(r['original_vote'])
        r['normalized_gpt4o_vote'] = normalize_vote(r['gpt4o_preference'])
    
    # Calculate agreement
    agreements = sum(1 for r in valid_results 
                    if r['normalized_original_vote'] == r['normalized_gpt4o_vote'] 
                    and r['normalized_original_vote'] != 'Unknown')
    
    total_comparable = sum(1 for r in valid_results 
                          if r['normalized_original_vote'] != 'Unknown')
    
    if total_comparable > 0:
        agreement_rate = agreements / total_comparable
        print(f"\n1. AGREEMENT STATISTICS")
        print(f"   Total comparable cases: {total_comparable}")
        print(f"   Agreements: {agreements}")
        print(f"   Agreement rate: {agreement_rate:.1%}")
    
    # Vote distribution
    print(f"\n2. VOTE DISTRIBUTION")
    
    original_votes = {}
    gpt4o_votes = {}
    
    for r in valid_results:
        orig = r['normalized_original_vote']
        gpt4 = r['normalized_gpt4o_vote']
        
        if orig != 'Unknown':
            original_votes[orig] = original_votes.get(orig, 0) + 1
        if gpt4 != 'Unknown':
            gpt4o_votes[gpt4] = gpt4o_votes.get(gpt4, 0) + 1
    
    print("\n   Original human votes:")
    for vote, count in sorted(original_votes.items()):
        print(f"      {vote}: {count} ({count/sum(original_votes.values())*100:.1f}%)")
    
    print("\n   GPT-4o votes:")
    for vote, count in sorted(gpt4o_votes.items()):
        print(f"      {vote}: {count} ({count/sum(gpt4o_votes.values())*100:.1f}%)")
    
    # Confidence analysis
    confidences = [r.get('gpt4o_confidence', 0) for r in valid_results if r.get('gpt4o_confidence', 0) > 0]
    if confidences:
        print(f"\n3. GPT-4o CONFIDENCE ANALYSIS")
        print(f"   Average confidence: {sum(confidences)/len(confidences):.2f}")
        print(f"   Min confidence: {min(confidences):.2f}")
        print(f"   Max confidence: {max(confidences):.2f}")
    
    # Save detailed CSV
    df = pd.DataFrame(valid_results)
    if len(df) > 0:
        output_csv = 'gpt4o_evaluation_analysis.csv'
        df.to_csv(output_csv, index=False)
        print(f"\n4. OUTPUT FILES")
        print(f"   Detailed results saved to: {output_csv}")
    
    # Create confusion matrix if we have enough data
    if total_comparable >= 5:
        create_confusion_matrix(valid_results)

def create_confusion_matrix(valid_results):
    """Create a confusion matrix visualization"""
    
    # Prepare data
    vote_map = {'Left': 0, 'Right': 1, 'Tie': 2}
    
    original = []
    gpt4o = []
    
    for r in valid_results:
        if r['normalized_original_vote'] in vote_map and r['normalized_gpt4o_vote'] in vote_map:
            original.append(vote_map[r['normalized_original_vote']])
            gpt4o.append(vote_map[r['normalized_gpt4o_vote']])
    
    if len(original) < 5:
        return
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(original, gpt4o, labels=[0, 1, 2])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Left', 'Right', 'Tie'],
                yticklabels=['Left', 'Right', 'Tie'])
    plt.title('Human vs GPT-4o Agreement\n(Both Agents Successful)')
    plt.xlabel('GPT-4o Vote')
    plt.ylabel('Human Vote')
    plt.tight_layout()
    plt.savefig('gpt4o_human_confusion_matrix.png', dpi=300)
    print("   Confusion matrix saved to: gpt4o_human_confusion_matrix.png")

if __name__ == "__main__":
    analyze_results()