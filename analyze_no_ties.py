#!/usr/bin/env python3
"""
Analyze pairwise agreements excluding tie votes
"""

import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def load_all_data():
    """Load all evaluation data"""
    # Load multisampling results
    with open('gpt4_multisampling_results.json', 'r') as f:
        multisampling_data = json.load(f)
    
    # Load survey results
    survey_df = pd.read_csv('agreement-results/agreement_analysis_results.csv')
    
    # Load baseline results
    baseline_df = pd.read_csv('agreement-results/baseline.csv')
    
    return multisampling_data, survey_df, baseline_df

def calculate_no_tie_agreement():
    """Calculate agreement rates excluding interactions where any evaluator voted Tie"""
    multisampling_data, survey_df, baseline_df = load_all_data()
    
    # Question to interaction mapping
    question_map = {
        'Q1': 'interaction_01', 'Q2': 'interaction_02', 'Q3': 'interaction_03',
        'Q4': 'interaction_04', 'Q5': 'interaction_05', 'Q6': 'interaction_06',
        'Q7': 'interaction_07', 'Q8': 'interaction_08', 'Q9': 'interaction_09',
        'Q10': 'interaction_10', 'Q11': 'interaction_11', 'Q12': 'interaction_12',
        'Q13': 'interaction_13', 'Q16': 'interaction_14', 'Q17': 'interaction_15',
        'Q18': 'interaction_16', 'Q19': 'interaction_17', 'Q20': 'interaction_18',
        'Q22': 'interaction_19'
    }
    
    # Process votes
    all_votes = {}
    
    # Get GPT-4 votes
    for int_id, data in multisampling_data['evaluations'].items():
        if int_id not in all_votes:
            all_votes[int_id] = {}
        all_votes[int_id]['gpt4'] = data['consistency']['majority_vote']
    
    # Get survey votes
    for _, row in survey_df.iterrows():
        if row['question'] in question_map:
            int_id = question_map[row['question']]
            if int_id not in all_votes:
                all_votes[int_id] = {}
            majority = row['majority_label']
            if majority == 'Agent 1':
                all_votes[int_id]['survey'] = 'Left'
            elif majority == 'Agent 2':
                all_votes[int_id]['survey'] = 'Right'
            else:
                all_votes[int_id]['survey'] = 'Tie'
    
    # Get baseline votes
    for _, row in baseline_df.iterrows():
        if row['question'] in question_map:
            int_id = question_map[row['question']]
            if int_id not in all_votes:
                all_votes[int_id] = {}
            winner = row['winner']
            if winner == 'Agent 1':
                all_votes[int_id]['baseline'] = 'Left'
            elif winner == 'Agent 2':
                all_votes[int_id]['baseline'] = 'Right'
            else:
                all_votes[int_id]['baseline'] = 'Tie'
    
    return all_votes

def analyze_no_ties():
    """Analyze agreements excluding all tie votes"""
    all_votes = calculate_no_tie_agreement()
    
    print("="*80)
    print("PAIRWISE AGREEMENT ANALYSIS - EXCLUDING TIES")
    print("="*80)
    
    # Filter out interactions where any evaluator voted Tie
    no_tie_votes = {}
    tie_excluded_count = 0
    
    for int_id, votes in all_votes.items():
        if 'gpt4' in votes and 'survey' in votes and 'baseline' in votes:
            if votes['gpt4'] != 'Tie' and votes['survey'] != 'Tie' and votes['baseline'] != 'Tie':
                no_tie_votes[int_id] = votes
            else:
                tie_excluded_count += 1
    
    print(f"\nExcluded {tie_excluded_count} interactions where at least one evaluator voted Tie")
    print(f"Analyzing {len(no_tie_votes)} interactions with only Left/Right decisions\n")
    
    # Calculate pairwise agreements
    gpt4_survey_agree = 0
    gpt4_baseline_agree = 0
    survey_baseline_agree = 0
    all_agree = 0
    
    print("INTERACTION-BY-INTERACTION (No Ties)")
    print("-"*80)
    print(f"{'ID':<15} {'GPT-4':<10} {'Survey':<10} {'Baseline':<10} {'All Agree':<10}")
    print("-"*80)
    
    for int_id in sorted(no_tie_votes.keys()):
        votes = no_tie_votes[int_id]
        gpt4 = votes['gpt4']
        survey = votes['survey']
        baseline = votes['baseline']
        
        if gpt4 == survey:
            gpt4_survey_agree += 1
        if gpt4 == baseline:
            gpt4_baseline_agree += 1
        if survey == baseline:
            survey_baseline_agree += 1
        if gpt4 == survey == baseline:
            all_agree += 1
            
        all_agree_mark = '✓' if gpt4 == survey == baseline else '✗'
        print(f"{int_id:<15} {gpt4:<10} {survey:<10} {baseline:<10} {all_agree_mark:<10}")
    
    total = len(no_tie_votes)
    
    print("\n" + "-"*80)
    print("AGREEMENT RATES (Excluding Tie Votes)")
    print("-"*80)
    print(f"GPT-4 vs Survey: {gpt4_survey_agree}/{total} = {gpt4_survey_agree/total*100:.1f}%")
    print(f"GPT-4 vs Baseline: {gpt4_baseline_agree}/{total} = {gpt4_baseline_agree/total*100:.1f}%")
    print(f"Survey vs Baseline: {survey_baseline_agree}/{total} = {survey_baseline_agree/total*100:.1f}%")
    print(f"All three agree: {all_agree}/{total} = {all_agree/total*100:.1f}%")
    
    # Also analyze when we force ties to be resolved
    print("\n\n" + "="*80)
    print("FORCED BINARY CHOICE ANALYSIS (Resolving Ties)")
    print("="*80)
    
    # For survey ties, use the higher vote count between Left and Right
    forced_votes = {}
    survey_df = pd.read_csv('agreement-results/agreement_analysis_results.csv')
    
    question_map = {
        'Q1': 'interaction_01', 'Q2': 'interaction_02', 'Q3': 'interaction_03',
        'Q4': 'interaction_04', 'Q5': 'interaction_05', 'Q6': 'interaction_06',
        'Q7': 'interaction_07', 'Q8': 'interaction_08', 'Q9': 'interaction_09',
        'Q10': 'interaction_10', 'Q11': 'interaction_11', 'Q12': 'interaction_12',
        'Q13': 'interaction_13', 'Q16': 'interaction_14', 'Q17': 'interaction_15',
        'Q18': 'interaction_16', 'Q19': 'interaction_17', 'Q20': 'interaction_18',
        'Q22': 'interaction_19'
    }
    
    # Get forced choices for survey
    for _, row in survey_df.iterrows():
        if row['question'] in question_map:
            int_id = question_map[row['question']]
            if int_id not in forced_votes:
                forced_votes[int_id] = {}
            
            # Original vote
            if row['majority_label'] == 'Tie':
                # Force to higher vote count
                left_votes = int(row['agent_1_votes'])
                right_votes = int(row['agent_2_votes'])
                forced_votes[int_id]['survey_forced'] = 'Left' if left_votes > right_votes else 'Right'
                forced_votes[int_id]['survey_original'] = 'Tie'
                forced_votes[int_id]['survey_margin'] = abs(left_votes - right_votes)
            else:
                forced_votes[int_id]['survey_forced'] = 'Left' if row['majority_label'] == 'Agent 1' else 'Right'
                forced_votes[int_id]['survey_original'] = forced_votes[int_id]['survey_forced']
    
    # Add GPT-4 and baseline votes
    for int_id, votes in all_votes.items():
        if int_id in forced_votes:
            forced_votes[int_id]['gpt4'] = votes.get('gpt4', 'N/A')
            forced_votes[int_id]['baseline'] = votes.get('baseline', 'N/A')
    
    # Calculate forced binary agreements
    forced_agreements = {'gpt4_survey': 0, 'gpt4_baseline': 0, 'survey_baseline': 0, 'all': 0}
    forced_total = 0
    
    print("\nForced Binary Choices (Survey Ties → Plurality Winner)")
    print("-"*80)
    
    for int_id in sorted(forced_votes.keys()):
        if all(k in forced_votes[int_id] for k in ['gpt4', 'survey_forced', 'baseline']):
            if forced_votes[int_id]['gpt4'] != 'Tie' and forced_votes[int_id]['baseline'] != 'Tie':
                forced_total += 1
                gpt4 = forced_votes[int_id]['gpt4']
                survey = forced_votes[int_id]['survey_forced']
                baseline = forced_votes[int_id]['baseline']
                
                if gpt4 == survey:
                    forced_agreements['gpt4_survey'] += 1
                if gpt4 == baseline:
                    forced_agreements['gpt4_baseline'] += 1
                if survey == baseline:
                    forced_agreements['survey_baseline'] += 1
                if gpt4 == survey == baseline:
                    forced_agreements['all'] += 1
                
                if forced_votes[int_id]['survey_original'] == 'Tie':
                    margin = forced_votes[int_id].get('survey_margin', 0)
                    print(f"{int_id}: Survey Tie → {survey} (margin: {margin} votes)")
    
    print(f"\nForced Binary Agreement Rates:")
    print(f"GPT-4 vs Survey: {forced_agreements['gpt4_survey']}/{forced_total} = {forced_agreements['gpt4_survey']/forced_total*100:.1f}%")
    print(f"GPT-4 vs Baseline: {forced_agreements['gpt4_baseline']}/{forced_total} = {forced_agreements['gpt4_baseline']/forced_total*100:.1f}%")
    print(f"Survey vs Baseline: {forced_agreements['survey_baseline']}/{forced_total} = {forced_agreements['survey_baseline']/forced_total*100:.1f}%")
    
    return no_tie_votes, forced_votes, forced_agreements

def create_no_ties_plots():
    """Create visualizations for no-ties analysis"""
    no_tie_votes, forced_votes, forced_agreements = analyze_no_ties()
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Agreement rates comparison (with vs without ties)
    categories = ['GPT-4 vs\nSurvey', 'GPT-4 vs\nBaseline', 'Survey vs\nBaseline']
    
    # Original rates (from main analysis)
    original_rates = [52.6, 73.7, 63.2]
    
    # No-tie rates (calculated above)
    total_no_tie = len(no_tie_votes)
    no_tie_rates = []
    
    # Recalculate for plotting
    gpt4_survey = sum(1 for v in no_tie_votes.values() if v['gpt4'] == v['survey'])
    gpt4_baseline = sum(1 for v in no_tie_votes.values() if v['gpt4'] == v['baseline'])
    survey_baseline = sum(1 for v in no_tie_votes.values() if v['survey'] == v['baseline'])
    
    no_tie_rates = [
        gpt4_survey/total_no_tie*100,
        gpt4_baseline/total_no_tie*100,
        survey_baseline/total_no_tie*100
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_rates, width, label='All Interactions', alpha=0.8)
    bars2 = ax1.bar(x + width/2, no_tie_rates, width, label='No Ties', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    ax1.set_ylabel('Agreement Rate (%)')
    ax1.set_title('Pairwise Agreement: All Interactions vs No-Tie Only')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # Plot 2: Agreement matrix for no-tie interactions
    agreement_data = {
        'GPT-4': [100.0, no_tie_rates[0], no_tie_rates[1]],
        'Survey': [no_tie_rates[0], 100.0, no_tie_rates[2]],
        'Baseline': [no_tie_rates[1], no_tie_rates[2], 100.0]
    }
    
    df = pd.DataFrame(agreement_data, index=['GPT-4', 'Survey', 'Baseline'])
    
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', 
                vmin=50, vmax=100, square=True, 
                cbar_kws={'label': 'Agreement Rate (%)'},
                ax=ax2)
    
    ax2.set_title('Agreement Matrix - No Tie Interactions Only')
    
    plt.tight_layout()
    plt.savefig('no_ties_agreement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create forced binary choice visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Count forced tie resolutions
    tie_resolutions = {'Left': 0, 'Right': 0}
    tie_margins = []
    
    for int_id, data in forced_votes.items():
        if data.get('survey_original') == 'Tie':
            tie_resolutions[data['survey_forced']] += 1
            tie_margins.append(data.get('survey_margin', 0))
    
    # Plot tie resolution distribution
    labels = ['Forced to Left', 'Forced to Right']
    sizes = [tie_resolutions['Left'], tie_resolutions['Right']]
    colors = ['#1f77b4', '#ff7f0e']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
    ax.set_title(f'Survey Tie Resolution Direction\n({sum(sizes)} total ties resolved)')
    
    plt.tight_layout()
    plt.savefig('tie_resolution_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plots saved: no_ties_agreement_analysis.png, tie_resolution_distribution.png")

if __name__ == "__main__":
    analyze_no_ties()
    create_no_ties_plots()