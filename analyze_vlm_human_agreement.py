#!/usr/bin/env python3
"""
Analyze VLM (GPT-4) vs Human agreement on browser task evaluations
where both agents successfully completed the task.
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_existing_evaluations():
    """Load existing GPT-4 evaluation results"""
    evaluations = {}
    
    # Load GPT-4 multisampling results
    if Path('gpt4_multisampling_results.json').exists():
        with open('gpt4_multisampling_results.json', 'r') as f:
            data = json.load(f)
            evaluations['gpt4_multisampling'] = data
    
    # Load standard GPT-4 evaluation results
    if Path('gpt4_evaluation_results.json').exists():
        with open('gpt4_evaluation_results.json', 'r') as f:
            data = json.load(f)
            evaluations['gpt4_standard'] = data
    
    return evaluations

def map_evaluations_to_human_votes():
    """Map VLM evaluations to human votes for cases where both agents succeeded"""
    
    # Load human evaluation data
    df_human = pd.read_excel('agreement-results/data_updated/ProlificBrowserArenaGIFFeedbackForm_May 14, 2025_22.32_usable_filtered.xlsx')
    
    # Load successful cases
    with open('both_agents_successful_cases.json', 'r') as f:
        successful_cases = json.load(f)['both_agents_successful']
    
    # Load existing evaluations
    evaluations = load_existing_evaluations()
    
    # Create mapping from task IDs to evaluations
    comparison_data = []
    
    for case in successful_cases:
        # Extract relevant data
        comparison = {
            'left_task': case['left_task'],
            'right_task': case['right_task'],
            'human_vote': case['vote'],
            'left_success_rate': case['left_success_rate'],
            'right_success_rate': case['right_success_rate']
        }
        
        # Try to find corresponding GPT-4 evaluation
        # Check multisampling results first
        if 'gpt4_multisampling' in evaluations:
            gpt4_data = evaluations['gpt4_multisampling']
            # Look for matching interaction
            for interaction_id, eval_data in gpt4_data.get('evaluations', {}).items():
                # Match based on task pattern or interaction number
                if any(task_id in str(eval_data) for task_id in [case['left_task'], case['right_task']]):
                    comparison['gpt4_vote'] = eval_data['consistency']['majority_vote']
                    comparison['gpt4_consistency'] = eval_data['consistency']['consistency_score']
                    comparison['gpt4_samples'] = eval_data['samples']
                    break
        
        # If we found a GPT-4 evaluation, add to comparison data
        if 'gpt4_vote' in comparison:
            comparison_data.append(comparison)
    
    return comparison_data

def analyze_agreement_patterns(comparison_data):
    """Analyze agreement patterns between VLM and human evaluations"""
    
    if not comparison_data:
        print("No comparison data available")
        return
    
    print("="*70)
    print("VLM vs HUMAN AGREEMENT ANALYSIS")
    print("(Cases where both agents successfully completed the task)")
    print("="*70)
    
    # Convert votes to consistent format
    vote_mapping = {
        'Left': 'Left',
        'Right': 'Right',
        'Tie': 'Tie',
        'left': 'Left',
        'right': 'Right',
        'tie': 'Tie'
    }
    
    agreements = 0
    disagreements = 0
    disagreement_details = []
    
    for comp in comparison_data:
        human_vote = vote_mapping.get(comp['human_vote'], comp['human_vote'])
        gpt4_vote = vote_mapping.get(comp.get('gpt4_vote', ''), '')
        
        if gpt4_vote and human_vote == gpt4_vote:
            agreements += 1
        elif gpt4_vote:
            disagreements += 1
            disagreement_details.append({
                'tasks': f"{comp['left_task']} vs {comp['right_task']}",
                'human': human_vote,
                'gpt4': gpt4_vote,
                'gpt4_consistency': comp.get('gpt4_consistency', 'N/A')
            })
    
    total_compared = agreements + disagreements
    
    if total_compared > 0:
        print(f"\n1. OVERALL AGREEMENT")
        print(f"   Total cases analyzed: {total_compared}")
        print(f"   Agreements: {agreements} ({agreements/total_compared*100:.1f}%)")
        print(f"   Disagreements: {disagreements} ({disagreements/total_compared*100:.1f}%)")
        
        # Vote distribution analysis
        print(f"\n2. VOTE DISTRIBUTION")
        human_votes = Counter(vote_mapping.get(c['human_vote'], c['human_vote']) for c in comparison_data)
        gpt4_votes = Counter(vote_mapping.get(c.get('gpt4_vote', ''), '') for c in comparison_data if c.get('gpt4_vote'))
        
        print(f"\n   Human votes:")
        for vote, count in human_votes.most_common():
            print(f"      {vote}: {count} ({count/len(comparison_data)*100:.1f}%)")
        
        print(f"\n   GPT-4 votes:")
        for vote, count in gpt4_votes.most_common():
            if vote:  # Skip empty votes
                print(f"      {vote}: {count} ({count/total_compared*100:.1f}%)")
        
        # Disagreement patterns
        if disagreement_details:
            print(f"\n3. DISAGREEMENT PATTERNS")
            print(f"   Cases where VLM and humans disagreed:")
            
            # Group by disagreement type
            disagreement_types = Counter((d['human'], d['gpt4']) for d in disagreement_details)
            
            for (human, gpt4), count in disagreement_types.most_common():
                print(f"\n   Human: {human} vs GPT-4: {gpt4} ({count} cases)")
                # Show examples
                examples = [d for d in disagreement_details if d['human'] == human and d['gpt4'] == gpt4][:2]
                for ex in examples:
                    print(f"      - Tasks: {ex['tasks']}")
                    print(f"        GPT-4 consistency: {ex['gpt4_consistency']}")
        
        # GPT-4 consistency analysis
        consistencies = [c.get('gpt4_consistency', 0) for c in comparison_data if 'gpt4_consistency' in c]
        if consistencies:
            print(f"\n4. GPT-4 CONSISTENCY ANALYSIS")
            print(f"   Average consistency score: {np.mean(consistencies):.3f}")
            print(f"   Min consistency: {min(consistencies):.3f}")
            print(f"   Max consistency: {max(consistencies):.3f}")
            
            # Check if low consistency correlates with disagreement
            low_consistency_disagreements = [
                d for d in disagreement_details 
                if isinstance(d.get('gpt4_consistency'), (int, float)) and d['gpt4_consistency'] < 0.8
            ]
            if low_consistency_disagreements:
                print(f"\n   Disagreements with low GPT-4 consistency (<0.8): {len(low_consistency_disagreements)}")
    
    else:
        print("No matching evaluations found between GPT-4 and human votes")
    
    return comparison_data, agreements, disagreements

def create_visualization(comparison_data):
    """Create visualization of VLM vs Human agreement"""
    if not comparison_data or len(comparison_data) < 5:
        print("\nInsufficient data for visualization")
        return
    
    # Prepare data for confusion matrix
    vote_mapping = {
        'Left': 0, 'left': 0,
        'Right': 1, 'right': 1,
        'Tie': 2, 'tie': 2
    }
    
    human_votes = []
    gpt4_votes = []
    
    for comp in comparison_data:
        if 'gpt4_vote' in comp:
            human_votes.append(vote_mapping.get(comp['human_vote'], -1))
            gpt4_votes.append(vote_mapping.get(comp['gpt4_vote'], -1))
    
    # Filter out invalid votes
    valid_pairs = [(h, g) for h, g in zip(human_votes, gpt4_votes) if h >= 0 and g >= 0]
    if not valid_pairs:
        print("\nNo valid vote pairs for visualization")
        return
    
    human_votes, gpt4_votes = zip(*valid_pairs)
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(human_votes, gpt4_votes, labels=[0, 1, 2])
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Left', 'Right', 'Tie'],
                yticklabels=['Left', 'Right', 'Tie'])
    plt.title('VLM vs Human Agreement\n(Both Agents Successful Cases)')
    plt.xlabel('GPT-4 Vote')
    plt.ylabel('Human Vote')
    plt.tight_layout()
    plt.savefig('vlm_human_agreement_successful_cases.png', dpi=300)
    print("\nConfusion matrix saved to: vlm_human_agreement_successful_cases.png")

def main():
    """Main analysis pipeline"""
    print("Loading and analyzing VLM vs Human evaluations...")
    
    # Map evaluations to human votes
    comparison_data = map_evaluations_to_human_votes()
    
    # Analyze agreement patterns
    comparison_data, agreements, disagreements = analyze_agreement_patterns(comparison_data)
    
    # Create visualization
    create_visualization(comparison_data)
    
    # Save detailed results
    if comparison_data:
        output_file = 'vlm_human_comparison_successful_cases.json'
        with open(output_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_cases': len(comparison_data),
                    'agreements': agreements,
                    'disagreements': disagreements,
                    'agreement_rate': agreements / (agreements + disagreements) if (agreements + disagreements) > 0 else 0
                },
                'detailed_comparisons': comparison_data
            }, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()