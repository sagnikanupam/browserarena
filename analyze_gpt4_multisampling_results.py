#!/usr/bin/env python3
"""
Analyze GPT-4 multisampling results with three-way comparison
"""

import json
import csv
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

def load_all_data():
    """Load all relevant data files"""
    # Load multisampling results
    with open('gpt4_multisampling_results.json', 'r') as f:
        multisampling_data = json.load(f)
    
    # Load survey results
    survey_df = pd.read_csv('agreement-results/agreement_analysis_results.csv')
    
    # Load baseline results
    baseline_df = pd.read_csv('agreement-results/baseline.csv')
    
    return multisampling_data, survey_df, baseline_df

def create_comparison_table():
    """Create detailed three-way comparison table"""
    multisampling_data, survey_df, baseline_df = load_all_data()
    
    # Question to interaction mapping
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
    
    # Process survey results
    survey_votes = {}
    for _, row in survey_df.iterrows():
        if row['question'] in question_map:
            int_id = question_map[row['question']]
            majority = row['majority_label']
            if majority == 'Agent 1':
                majority = 'Left'
            elif majority == 'Agent 2':
                majority = 'Right'
            else:
                majority = 'Tie'
            
            survey_votes[int_id] = {
                'majority': majority,
                'left_votes': int(row['agent_1_votes']),
                'right_votes': int(row['agent_2_votes']),
                'tie_votes': int(row['tie_votes']),
                'total_votes': int(row['n_responses'])
            }
    
    # Process baseline results
    baseline_votes = {}
    for _, row in baseline_df.iterrows():
        if row['question'] in question_map:
            int_id = question_map[row['question']]
            winner = row['winner']
            if winner == 'Agent 1':
                vote = 'Left'
            elif winner == 'Agent 2':
                vote = 'Right'
            else:
                vote = 'Tie'
            
            baseline_votes[int_id] = {
                'vote': vote,
                'agent1_model': row['agent1_model'],
                'agent2_model': row['agent2_model']
            }
    
    return multisampling_data, survey_votes, baseline_votes

def analyze_agreement_patterns():
    """Analyze agreement patterns across all three sources"""
    multisampling_data, survey_votes, baseline_votes = create_comparison_table()
    
    print("="*100)
    print("THREE-WAY COMPARISON: GPT-4 (5x sampling) vs HUMAN SURVEY vs BASELINE")
    print("="*100)
    
    # Summary statistics
    print("\n1. VOTE DISTRIBUTION")
    print("-"*80)
    
    # GPT-4 vote distribution
    gpt4_vote_counts = {'Left': 0, 'Right': 0, 'Tie': 0}
    for int_id, data in multisampling_data['evaluations'].items():
        majority = data['consistency']['majority_vote']
        gpt4_vote_counts[majority] += 1
    
    # Survey vote distribution
    survey_vote_counts = {'Left': 0, 'Right': 0, 'Tie': 0}
    for data in survey_votes.values():
        survey_vote_counts[data['majority']] += 1
    
    # Baseline vote distribution
    baseline_vote_counts = {'Left': 0, 'Right': 0, 'Tie': 0}
    for data in baseline_votes.values():
        baseline_vote_counts[data['vote']] += 1
    
    print(f"{'Source':<15} {'Left':<15} {'Right':<15} {'Tie':<15}")
    print("-"*60)
    print(f"{'GPT-4':<15} {gpt4_vote_counts['Left']}/{len(multisampling_data['evaluations'])} ({gpt4_vote_counts['Left']/len(multisampling_data['evaluations'])*100:.1f}%){'':<5} "
          f"{gpt4_vote_counts['Right']}/{len(multisampling_data['evaluations'])} ({gpt4_vote_counts['Right']/len(multisampling_data['evaluations'])*100:.1f}%){'':<5} "
          f"{gpt4_vote_counts['Tie']}/{len(multisampling_data['evaluations'])} ({gpt4_vote_counts['Tie']/len(multisampling_data['evaluations'])*100:.1f}%)")
    print(f"{'Survey':<15} {survey_vote_counts['Left']}/{len(survey_votes)} ({survey_vote_counts['Left']/len(survey_votes)*100:.1f}%){'':<5} "
          f"{survey_vote_counts['Right']}/{len(survey_votes)} ({survey_vote_counts['Right']/len(survey_votes)*100:.1f}%){'':<5} "
          f"{survey_vote_counts['Tie']}/{len(survey_votes)} ({survey_vote_counts['Tie']/len(survey_votes)*100:.1f}%)")
    print(f"{'Baseline':<15} {baseline_vote_counts['Left']}/{len(baseline_votes)} ({baseline_vote_counts['Left']/len(baseline_votes)*100:.1f}%){'':<5} "
          f"{baseline_vote_counts['Right']}/{len(baseline_votes)} ({baseline_vote_counts['Right']/len(baseline_votes)*100:.1f}%){'':<5} "
          f"{baseline_vote_counts['Tie']}/{len(baseline_votes)} ({baseline_vote_counts['Tie']/len(baseline_votes)*100:.1f}%)")
    
    # Detailed comparison
    print("\n2. INTERACTION-BY-INTERACTION COMPARISON")
    print("-"*100)
    print(f"{'ID':<15} {'GPT-4':<10} {'Survey':<10} {'Baseline':<10} {'GPT4-Surv':<10} {'GPT4-Base':<10} {'Surv-Base':<10} {'All Agree':<10}")
    print("-"*100)
    
    # Agreement counters
    gpt4_survey_agree = 0
    gpt4_baseline_agree = 0
    survey_baseline_agree = 0
    all_three_agree = 0
    total_compared = 0
    
    # Pattern counters
    patterns = {
        'all_agree': [],
        'gpt4_alone': [],
        'survey_alone': [],
        'baseline_alone': [],
        'gpt4_survey': [],
        'gpt4_baseline': [],
        'survey_baseline': []
    }
    
    for int_id in sorted(multisampling_data['evaluations'].keys()):
        if int_id in survey_votes and int_id in baseline_votes:
            total_compared += 1
            
            gpt4_majority = multisampling_data['evaluations'][int_id]['consistency']['majority_vote']
            survey_majority = survey_votes[int_id]['majority']
            baseline_vote = baseline_votes[int_id]['vote']
            
            # Check agreements
            gpt4_surv = '✓' if gpt4_majority == survey_majority else '✗'
            gpt4_base = '✓' if gpt4_majority == baseline_vote else '✗'
            surv_base = '✓' if survey_majority == baseline_vote else '✗'
            
            if gpt4_majority == survey_majority:
                gpt4_survey_agree += 1
            if gpt4_majority == baseline_vote:
                gpt4_baseline_agree += 1
            if survey_majority == baseline_vote:
                survey_baseline_agree += 1
            
            all_agree = '✓' if gpt4_majority == survey_majority == baseline_vote else '✗'
            if all_agree == '✓':
                all_three_agree += 1
                patterns['all_agree'].append(int_id)
            
            # Categorize disagreement patterns
            if gpt4_majority == survey_majority == baseline_vote:
                patterns['all_agree'].append(int_id)
            elif gpt4_majority == survey_majority and gpt4_majority != baseline_vote:
                patterns['gpt4_survey'].append(int_id)
            elif gpt4_majority == baseline_vote and gpt4_majority != survey_majority:
                patterns['gpt4_baseline'].append(int_id)
            elif survey_majority == baseline_vote and survey_majority != gpt4_majority:
                patterns['survey_baseline'].append(int_id)
            elif gpt4_majority != survey_majority and gpt4_majority != baseline_vote and survey_majority != baseline_vote:
                # All three disagree - determine who stands alone
                if gpt4_majority != survey_majority and gpt4_majority != baseline_vote:
                    patterns['gpt4_alone'].append(int_id)
            
            print(f"{int_id:<15} {gpt4_majority:<10} {survey_majority:<10} {baseline_vote:<10} "
                  f"{gpt4_surv:<10} {gpt4_base:<10} {surv_base:<10} {all_agree:<10}")
    
    # Agreement summary
    print("\n3. AGREEMENT SUMMARY")
    print("-"*80)
    print(f"GPT-4 vs Survey agreement: {gpt4_survey_agree}/{total_compared} = {gpt4_survey_agree/total_compared*100:.1f}%")
    print(f"GPT-4 vs Baseline agreement: {gpt4_baseline_agree}/{total_compared} = {gpt4_baseline_agree/total_compared*100:.1f}%")
    print(f"Survey vs Baseline agreement: {survey_baseline_agree}/{total_compared} = {survey_baseline_agree/total_compared*100:.1f}%")
    print(f"All three agree: {all_three_agree}/{total_compared} = {all_three_agree/total_compared*100:.1f}%")
    
    # GPT-4 consistency analysis
    print("\n4. GPT-4 CONSISTENCY ANALYSIS")
    print("-"*80)
    
    consistency_scores = []
    unanimous_count = 0
    for int_id, data in multisampling_data['evaluations'].items():
        consistency = data['consistency']['consistency_score']
        consistency_scores.append(consistency)
        if data['consistency']['agreement_rate'] == 1.0:
            unanimous_count += 1
    
    print(f"Average consistency score: {np.mean(consistency_scores):.3f}")
    print(f"Unanimous decisions: {unanimous_count}/{len(multisampling_data['evaluations'])} = {unanimous_count/len(multisampling_data['evaluations'])*100:.1f}%")
    
    # Show non-unanimous cases
    print("\nNon-unanimous GPT-4 decisions:")
    for int_id, data in sorted(multisampling_data['evaluations'].items()):
        if data['consistency']['agreement_rate'] < 1.0:
            samples = data['samples']
            vote_counts = Counter(samples)
            print(f"  {int_id}: {dict(vote_counts)} → Majority: {data['consistency']['majority_vote']}")
    
    # Disagreement patterns
    print("\n5. DISAGREEMENT PATTERNS")
    print("-"*80)
    
    # Cases where survey participants chose tie but others picked winners
    survey_tie_cases = []
    for int_id in sorted(multisampling_data['evaluations'].keys()):
        if int_id in survey_votes and survey_votes[int_id]['majority'] == 'Tie':
            gpt4_vote = multisampling_data['evaluations'][int_id]['consistency']['majority_vote']
            baseline_vote = baseline_votes.get(int_id, {}).get('vote', 'N/A')
            if gpt4_vote != 'Tie' or baseline_vote != 'Tie':
                survey_tie_cases.append({
                    'id': int_id,
                    'gpt4': gpt4_vote,
                    'baseline': baseline_vote,
                    'survey_breakdown': f"L:{survey_votes[int_id]['left_votes']} R:{survey_votes[int_id]['right_votes']} T:{survey_votes[int_id]['tie_votes']}"
                })
    
    print(f"\nCases where survey chose Tie but others picked winners ({len(survey_tie_cases)} cases):")
    for case in survey_tie_cases:
        print(f"  {case['id']}: Survey=Tie ({case['survey_breakdown']}), GPT-4={case['gpt4']}, Baseline={case['baseline']}")
    
    # Save comprehensive report
    report = {
        'summary': {
            'total_interactions': len(multisampling_data['evaluations']),
            'total_compared': total_compared,
            'vote_distributions': {
                'gpt4': gpt4_vote_counts,
                'survey': survey_vote_counts,
                'baseline': baseline_vote_counts
            },
            'agreement_rates': {
                'gpt4_survey': gpt4_survey_agree/total_compared,
                'gpt4_baseline': gpt4_baseline_agree/total_compared,
                'survey_baseline': survey_baseline_agree/total_compared,
                'all_three': all_three_agree/total_compared
            },
            'gpt4_consistency': {
                'mean_score': float(np.mean(consistency_scores)),
                'unanimous_rate': unanimous_count/len(multisampling_data['evaluations'])
            }
        },
        'patterns': patterns,
        'survey_tie_analysis': survey_tie_cases
    }
    
    with open('gpt4_threeway_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*100)
    print("Analysis complete. Full report saved to gpt4_threeway_comparison_report.json")

if __name__ == "__main__":
    analyze_agreement_patterns()