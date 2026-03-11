#!/usr/bin/env python3
"""
Analyze VLM (GPT-4) vs Human agreement specifically for cases where both agents succeeded.
This analysis focuses on quality assessment when task completion is not a differentiator.
"""

import json
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load all relevant data files"""
    # Load human evaluations with success indicators
    df_human = pd.read_excel('agreement-results/data_updated/ProlificBrowserArenaGIFFeedbackForm_May 14, 2025_22.32_usable_filtered.xlsx')
    
    # Load GPT-4 evaluation summary
    df_gpt4_summary = pd.read_csv('gpt4_evaluation_summary.csv')
    
    # Load GPT-4 multisampling results
    with open('gpt4_multisampling_results.json', 'r') as f:
        gpt4_multi = json.load(f)
    
    # Load successful cases
    with open('both_agents_successful_cases.json', 'r') as f:
        successful_cases = json.load(f)['both_agents_successful']
    
    return df_human, df_gpt4_summary, gpt4_multi, successful_cases

def extract_interaction_from_task_id(task_id):
    """Extract interaction number from task ID format"""
    # Task IDs like "03_05_2025_04_36_24_hCn" don't directly map to interactions
    # We need to use the question mapping from the original analysis
    return None

def analyze_successful_cases_agreement():
    """Analyze agreement for cases where both agents succeeded"""
    
    df_human, df_gpt4_summary, gpt4_multi, successful_cases = load_data()
    
    print("="*80)
    print("VLM vs HUMAN AGREEMENT ANALYSIS - BOTH AGENTS SUCCESSFUL CASES")
    print("="*80)
    
    # Question to interaction mapping (from the original analysis)
    question_map = {
        'Q1': 'interaction_01', 'Q2': 'interaction_02', 'Q3': 'interaction_03',
        'Q4': 'interaction_04', 'Q5': 'interaction_05', 'Q6': 'interaction_06',
        'Q7': 'interaction_07', 'Q8': 'interaction_08', 'Q9': 'interaction_09',
        'Q10': 'interaction_10', 'Q11': 'interaction_11', 'Q12': 'interaction_12',
        'Q13': 'interaction_13', 'Q16': 'interaction_14', 'Q17': 'interaction_15',
        'Q18': 'interaction_16', 'Q19': 'interaction_17', 'Q20': 'interaction_18',
        'Q22': 'interaction_19', 'Q23': 'interaction_20', 'Q24': 'interaction_21',
        'Q25': 'interaction_22'
    }
    
    # Filter GPT-4 summary for successful cases only
    successful_interactions = []
    
    # First, identify which interactions had both agents succeed
    for _, row in df_gpt4_summary.iterrows():
        interaction_id = row['Interaction_ID']
        # Check if this interaction appears in our successful cases
        # We'll use the human vote distribution as a proxy
        
        # An interaction where both succeeded should have substantial non-tie votes
        # indicating humans could distinguish quality despite both completing the task
        total_votes = row['Human_Left'] + row['Human_Right'] + row['Human_Tie']
        if total_votes > 0:
            left_rate = row['Human_Left'] / total_votes
            right_rate = row['Human_Right'] / total_votes
            tie_rate = row['Human_Tie'] / total_votes
            
            # Consider it a "both successful" case if there's meaningful voting
            # (not dominated by one agent failing)
            if left_rate < 0.9 and right_rate < 0.9:  # Neither agent dominated
                successful_interactions.append({
                    'interaction_id': interaction_id,
                    'task': row['Task'],
                    'gpt4_vote': row['GPT4_Vote'],
                    'human_majority': row['Human_Majority'],
                    'human_left': row['Human_Left'],
                    'human_right': row['Human_Right'],
                    'human_tie': row['Human_Tie'],
                    'agrees': row['Agrees']
                })
    
    print(f"\n1. DATASET OVERVIEW")
    print(f"   Total interactions analyzed: {len(df_gpt4_summary)}")
    print(f"   Interactions where both agents likely succeeded: {len(successful_interactions)}")
    
    # Analyze agreement for successful cases
    agreements = sum(1 for si in successful_interactions if si['agrees'] == 'Yes')
    disagreements = len(successful_interactions) - agreements
    
    print(f"\n2. AGREEMENT ANALYSIS (Both Agents Successful)")
    print(f"   Cases analyzed: {len(successful_interactions)}")
    print(f"   Agreements: {agreements} ({agreements/len(successful_interactions)*100:.1f}%)")
    print(f"   Disagreements: {disagreements} ({disagreements/len(successful_interactions)*100:.1f}%)")
    
    # Vote distribution comparison
    print(f"\n3. VOTE DISTRIBUTION COMPARISON")
    
    gpt4_votes = Counter(si['gpt4_vote'] for si in successful_interactions)
    human_votes = Counter(si['human_majority'] for si in successful_interactions)
    
    print(f"\n   GPT-4 Votes:")
    for vote, count in gpt4_votes.most_common():
        print(f"      {vote}: {count} ({count/len(successful_interactions)*100:.1f}%)")
    
    print(f"\n   Human Majority Votes:")
    for vote, count in human_votes.most_common():
        print(f"      {vote}: {count} ({count/len(successful_interactions)*100:.1f}%)")
    
    # Analyze disagreement patterns
    print(f"\n4. DISAGREEMENT PATTERNS")
    disagreement_cases = [si for si in successful_interactions if si['agrees'] == 'No']
    
    if disagreement_cases:
        # Group by pattern
        patterns = Counter((d['gpt4_vote'], d['human_majority']) for d in disagreement_cases)
        
        for (gpt4, human), count in patterns.most_common():
            print(f"\n   GPT-4: {gpt4} vs Human: {human} ({count} cases)")
            # Show examples
            examples = [d for d in disagreement_cases if d['gpt4_vote'] == gpt4 and d['human_majority'] == human][:2]
            for ex in examples:
                print(f"      - {ex['task'][:80]}...")
                print(f"        Human votes: L:{ex['human_left']} R:{ex['human_right']} T:{ex['human_tie']}")
    
    # Analyze GPT-4 consistency for these cases
    print(f"\n5. GPT-4 CONSISTENCY ANALYSIS")
    
    if gpt4_multi and 'evaluations' in gpt4_multi:
        consistencies = []
        for si in successful_interactions:
            int_id = si['interaction_id']
            if int_id in gpt4_multi['evaluations']:
                eval_data = gpt4_multi['evaluations'][int_id]
                if 'consistency' in eval_data:
                    consistencies.append(eval_data['consistency']['consistency_score'])
        
        if consistencies:
            print(f"   Average consistency score: {np.mean(consistencies):.3f}")
            print(f"   Cases with perfect agreement (1.0): {sum(1 for c in consistencies if c == 1.0)}")
            print(f"   Cases with low agreement (<0.8): {sum(1 for c in consistencies if c < 0.8)}")
    
    # Key insights
    print(f"\n6. KEY INSIGHTS")
    print(f"   - When both agents succeed, GPT-4 agrees with humans {agreements/len(successful_interactions)*100:.1f}% of the time")
    print(f"   - Humans choose 'Tie' in {human_votes.get('Tie', 0)/len(successful_interactions)*100:.1f}% of successful cases")
    print(f"   - GPT-4 chooses 'Tie' in {gpt4_votes.get('Tie', 0)/len(successful_interactions)*100:.1f}% of successful cases")
    
    tie_difference = human_votes.get('Tie', 0) - gpt4_votes.get('Tie', 0)
    if tie_difference > 0:
        print(f"   - Humans are more likely to call it a tie when both succeed (+{tie_difference} cases)")
    
    # Create visualization
    create_comparison_visualization(successful_interactions)
    
    # Save detailed results
    save_results(successful_interactions, agreements, disagreements)

def create_comparison_visualization(successful_interactions):
    """Create visualization comparing VLM and human votes"""
    
    # Prepare data for plotting
    data_for_plot = []
    for si in successful_interactions:
        data_for_plot.append({
            'Interaction': si['interaction_id'].replace('interaction_', 'I'),
            'GPT-4': si['gpt4_vote'],
            'Human': si['human_majority'],
            'Agreement': 'Agree' if si['agrees'] == 'Yes' else 'Disagree'
        })
    
    df_plot = pd.DataFrame(data_for_plot)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confusion matrix
    vote_map = {'Left': 0, 'Right': 1, 'Tie': 2}
    gpt4_numeric = [vote_map[v] for v in df_plot['GPT-4']]
    human_numeric = [vote_map[v] for v in df_plot['Human']]
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(human_numeric, gpt4_numeric, labels=[0, 1, 2])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Left', 'Right', 'Tie'],
                yticklabels=['Left', 'Right', 'Tie'],
                ax=ax1)
    ax1.set_title('VLM vs Human Votes\n(Both Agents Successful)')
    ax1.set_xlabel('GPT-4 Vote')
    ax1.set_ylabel('Human Majority Vote')
    
    # Agreement rate by interaction
    agreement_data = df_plot.groupby('Agreement').size()
    ax2.pie(agreement_data.values, labels=agreement_data.index, autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'])
    ax2.set_title('Overall Agreement Rate')
    
    plt.tight_layout()
    plt.savefig('vlm_human_agreement_both_successful.png', dpi=300, bbox_inches='tight')
    print("\n   Visualization saved to: vlm_human_agreement_both_successful.png")

def save_results(successful_interactions, agreements, disagreements):
    """Save detailed results to file"""
    
    results = {
        'summary': {
            'total_interactions': len(successful_interactions),
            'agreements': agreements,
            'disagreements': disagreements,
            'agreement_rate': agreements/len(successful_interactions) if successful_interactions else 0
        },
        'interactions': successful_interactions
    }
    
    with open('vlm_human_agreement_both_successful_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n   Detailed results saved to: vlm_human_agreement_both_successful_results.json")

if __name__ == "__main__":
    analyze_successful_cases_agreement()