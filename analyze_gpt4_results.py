#!/usr/bin/env python3
"""
Analyze GPT-4 evaluation results compared to both survey and baseline human data.
"""

import json
import csv
from pathlib import Path
import pandas as pd
import numpy as np

def load_baseline_results():
    """Load baseline results for comparison."""
    results_file = Path("agreement-results/baseline.csv")
    
    baseline_votes = {}
    question_map = {
        'Q1': 'interaction_01',
        'Q2': 'interaction_02', 
        'Q3': 'interaction_03',
        'Q4': 'interaction_04',
        'Q5': 'interaction_05',
        'Q6': 'interaction_06',
        'Q7': 'interaction_07',
        'Q8': 'interaction_08',
        'Q9': 'interaction_09',
        'Q10': 'interaction_10',
        'Q11': 'interaction_11',
        'Q12': 'interaction_12',
        'Q13': 'interaction_13',
        'Q16': 'interaction_14',
        'Q17': 'interaction_15',
        'Q18': 'interaction_16',
        'Q19': 'interaction_17',
        'Q20': 'interaction_18',
        'Q22': 'interaction_19'
    }
    
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row['question']
            if question in question_map:
                interaction_id = question_map[question]
                
                # Convert winner to our format
                winner = row['winner']
                if winner == 'Agent 1':
                    vote = 'Left'
                elif winner == 'Agent 2':
                    vote = 'Right'
                else:
                    vote = 'Tie'
                
                baseline_votes[interaction_id] = {
                    'vote': vote,
                    'agent1_model': row['agent1_model'],
                    'agent2_model': row['agent2_model']
                }
    
    return baseline_votes

def main():
    # Load existing GPT-4 results
    with open('gpt4_evaluation_results.json', 'r') as f:
        gpt4_data = json.load(f)
    
    # Load human comparison data
    with open('gpt4_human_comparison.json', 'r') as f:
        comparison_data = json.load(f)
    
    # Load baseline results
    baseline_results = load_baseline_results()
    
    # Load survey results CSV for additional analysis
    survey_df = pd.read_csv('gpt4_evaluation_summary.csv')
    
    print("="*80)
    print("COMPREHENSIVE GPT-4 EVALUATION ANALYSIS")
    print("="*80)
    
    # Basic statistics
    print("\n1. BASIC STATISTICS")
    print("-"*40)
    print(f"Total interactions evaluated: {len(gpt4_data['evaluations'])}")
    print(f"Survey comparison available: {comparison_data['total_compared']}")
    print(f"Baseline comparison available: {len(baseline_results)}")
    
    # Vote distribution
    vote_counts = {'Left': 0, 'Right': 0, 'Tie': 0}
    for vote in gpt4_data['evaluations'].values():
        vote_counts[vote] += 1
    
    print("\n2. GPT-4 VOTE DISTRIBUTION")
    print("-"*40)
    for vote, count in vote_counts.items():
        percentage = (count / len(gpt4_data['evaluations'])) * 100
        print(f"{vote}: {count} ({percentage:.1f}%)")
    
    # Agreement analysis
    print("\n3. AGREEMENT WITH HUMAN EVALUATORS")
    print("-"*40)
    
    # Survey agreement
    survey_agreement_rate = comparison_data['agreement_rate']
    print(f"Survey agreement rate: {survey_agreement_rate:.2%}")
    
    # Calculate baseline agreement
    baseline_agreement = 0
    baseline_compared = 0
    for int_id, gpt_vote in gpt4_data['evaluations'].items():
        if int_id in baseline_results:
            if gpt_vote == baseline_results[int_id]['vote']:
                baseline_agreement += 1
            baseline_compared += 1
    
    baseline_agreement_rate = baseline_agreement / baseline_compared if baseline_compared > 0 else 0
    print(f"Baseline agreement rate: {baseline_agreement_rate:.2%}")
    
    # Detailed comparison
    print("\n4. DETAILED COMPARISON TABLE")
    print("-"*80)
    print(f"{'ID':<15} {'GPT-4':<10} {'Survey':<10} {'Baseline':<10} {'Survey Match':<12} {'Base Match':<10}")
    print("-"*80)
    
    detailed_results = []
    for result in comparison_data['comparison_results']:
        int_id = result['interaction_id']
        gpt_vote = result['gpt_vote']
        survey_vote = result['human_majority']
        baseline_vote = baseline_results.get(int_id, {}).get('vote', 'N/A')
        
        survey_match = '✓' if result['agrees'] else '✗'
        baseline_match = '✓' if baseline_vote != 'N/A' and gpt_vote == baseline_vote else '✗' if baseline_vote != 'N/A' else '-'
        
        print(f"{int_id:<15} {gpt_vote:<10} {survey_vote:<10} {baseline_vote:<10} {survey_match:<12} {baseline_match:<10}")
        
        detailed_results.append({
            'id': int_id,
            'gpt': gpt_vote,
            'survey': survey_vote,
            'baseline': baseline_vote,
            'survey_match': result['agrees'],
            'baseline_match': baseline_vote != 'N/A' and gpt_vote == baseline_vote
        })
    
    # Agreement patterns
    print("\n5. AGREEMENT PATTERNS ANALYSIS")
    print("-"*40)
    
    # When all three agree
    all_agree = sum(1 for r in detailed_results if r['survey_match'] and r['baseline_match'])
    print(f"All three agree: {all_agree} interactions")
    
    # When GPT-4 agrees with survey but not baseline
    gpt_survey_only = sum(1 for r in detailed_results if r['survey_match'] and not r['baseline_match'] and r['baseline'] != 'N/A')
    print(f"GPT-4 agrees with survey only: {gpt_survey_only} interactions")
    
    # When GPT-4 agrees with baseline but not survey
    gpt_baseline_only = sum(1 for r in detailed_results if not r['survey_match'] and r['baseline_match'])
    print(f"GPT-4 agrees with baseline only: {gpt_baseline_only} interactions")
    
    # When GPT-4 disagrees with both
    gpt_disagrees_both = sum(1 for r in detailed_results if not r['survey_match'] and not r['baseline_match'] and r['baseline'] != 'N/A')
    print(f"GPT-4 disagrees with both: {gpt_disagrees_both} interactions")
    
    # Tie analysis
    print("\n6. TIE VOTE ANALYSIS")
    print("-"*40)
    
    # Count ties in each dataset
    gpt_ties = sum(1 for v in gpt4_data['evaluations'].values() if v == 'Tie')
    survey_ties = sum(1 for r in comparison_data['comparison_results'] if r['human_majority'] == 'Tie')
    baseline_ties = sum(1 for r in baseline_results.values() if r['vote'] == 'Tie')
    
    print(f"GPT-4 tie votes: {gpt_ties}/{len(gpt4_data['evaluations'])} ({gpt_ties/len(gpt4_data['evaluations'])*100:.1f}%)")
    print(f"Survey tie votes: {survey_ties}/{len(comparison_data['comparison_results'])} ({survey_ties/len(comparison_data['comparison_results'])*100:.1f}%)")
    print(f"Baseline tie votes: {baseline_ties}/{len(baseline_results)} ({baseline_ties/len(baseline_results)*100:.1f}%)")
    
    # Model performance in baseline
    print("\n7. MODEL MATCHUPS IN BASELINE")
    print("-"*40)
    
    model_pairs = {}
    for int_id, data in baseline_results.items():
        pair = f"{data['agent1_model'].split('/')[-1]} vs {data['agent2_model'].split('/')[-1]}"
        if pair not in model_pairs:
            model_pairs[pair] = {'total': 0, 'gpt_agrees': 0}
        model_pairs[pair]['total'] += 1
        if int_id in gpt4_data['evaluations'] and gpt4_data['evaluations'][int_id] == data['vote']:
            model_pairs[pair]['gpt_agrees'] += 1
    
    for pair, stats in sorted(model_pairs.items(), key=lambda x: x[1]['total'], reverse=True)[:5]:
        agreement = stats['gpt_agrees'] / stats['total'] * 100
        print(f"{pair}: {agreement:.1f}% agreement ({stats['gpt_agrees']}/{stats['total']})")
    
    # Save comprehensive report
    report = {
        'summary': {
            'total_evaluated': len(gpt4_data['evaluations']),
            'gpt4_vote_distribution': vote_counts,
            'survey_agreement_rate': survey_agreement_rate,
            'baseline_agreement_rate': baseline_agreement_rate,
            'all_three_agree': all_agree,
            'tie_rates': {
                'gpt4': gpt_ties/len(gpt4_data['evaluations']),
                'survey': survey_ties/len(comparison_data['comparison_results']),
                'baseline': baseline_ties/len(baseline_results)
            }
        },
        'detailed_results': detailed_results,
        'model_matchup_agreement': model_pairs
    }
    
    with open('gpt4_comprehensive_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("Analysis complete. Results saved to gpt4_comprehensive_analysis.json")

if __name__ == "__main__":
    main()