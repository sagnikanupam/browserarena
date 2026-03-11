#!/usr/bin/env python3
"""
Create a side-by-side comparison of all cases vs both-completed cases
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def normalize_vote(vote):
    """Normalize various vote formats to left/right/tie"""
    if pd.isna(vote):
        return 'unknown'
    
    vote_str = str(vote).lower().strip()
    
    if any(pattern in vote_str for pattern in ['left', 'model a', 'a is better', 'a completed']):
        return 'left'
    
    if any(pattern in vote_str for pattern in ['right', 'model b', 'b is better', 'b completed']):
        return 'right'
    
    if any(pattern in vote_str for pattern in ['tie', 'equal', 'same', 'both']):
        return 'tie'
    
    return 'unknown'

def analyze_dataset(df, name):
    """Analyze a dataset and return key metrics"""
    df['normalized_human_vote'] = df['original_vote'].apply(normalize_vote)
    df['normalized_gpt4o_vote'] = df['gpt4o_preference'].str.lower()
    
    clear_df = df[df['normalized_human_vote'].isin(['left', 'right', 'tie'])].copy()
    
    agreements = clear_df['normalized_human_vote'] == clear_df['normalized_gpt4o_vote']
    agreement_rate = agreements.sum() / len(clear_df) if len(clear_df) > 0 else 0
    
    human_votes = Counter(clear_df['normalized_human_vote'])
    gpt4o_votes = Counter(clear_df['normalized_gpt4o_vote'])
    
    # Agreement by preference type
    pref_agreements = {}
    for pref_type in ['left', 'right', 'tie']:
        type_df = clear_df[clear_df['normalized_human_vote'] == pref_type]
        if len(type_df) > 0:
            type_agreements = type_df['normalized_human_vote'] == type_df['normalized_gpt4o_vote']
            pref_agreements[pref_type] = {
                'rate': type_agreements.sum() / len(type_df),
                'count': len(type_df),
                'agreements': type_agreements.sum()
            }
        else:
            pref_agreements[pref_type] = {'rate': 0, 'count': 0, 'agreements': 0}
    
    avg_confidence = clear_df['gpt4o_confidence'].mean() if len(clear_df) > 0 else 0
    
    agree_df = clear_df[agreements]
    disagree_df = clear_df[~agreements]
    
    return {
        'name': name,
        'total_cases': len(clear_df),
        'agreement_rate': agreement_rate,
        'human_votes': human_votes,
        'gpt4o_votes': gpt4o_votes,
        'pref_agreements': pref_agreements,
        'avg_confidence': avg_confidence,
        'agree_confidence': agree_df['gpt4o_confidence'].mean() if len(agree_df) > 0 else 0,
        'disagree_confidence': disagree_df['gpt4o_confidence'].mean() if len(disagree_df) > 0 else 0
    }

def main():
    """Create comparison analysis"""
    
    # Load all cases
    with open('gpt4o_all_tasks_evaluations/evaluation_results.json', 'r') as f:
        all_results = json.load(f)
    
    all_df = pd.DataFrame(all_results)
    all_valid_df = all_df[all_df['gpt4o_preference'] != 'error'].copy()
    
    # Load both-completed cases
    both_completed_df = pd.read_csv('both_agents_completed_cases.csv')
    
    # Analyze both datasets
    all_analysis = analyze_dataset(all_valid_df, "All Cases")
    both_analysis = analyze_dataset(both_completed_df, "Both Completed")
    
    print("=== COMPARISON ANALYSIS ===")
    print(f"All Cases: {all_analysis['total_cases']} cases, {all_analysis['agreement_rate']*100:.1f}% agreement")
    print(f"Both Completed: {both_analysis['total_cases']} cases, {both_analysis['agreement_rate']*100:.1f}% agreement")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('All Cases vs Both Agents Completed: Comparison Analysis', fontsize=16, fontweight='bold')
    
    datasets = [all_analysis, both_analysis]
    colors = [['#FF6B6B', '#4ECDC4'], ['#FFB6C1', '#87CEEB']]
    
    # 1. Overall Agreement Rates
    ax = axes[0, 0]
    names = [d['name'] for d in datasets]
    rates = [d['agreement_rate'] * 100 for d in datasets]
    
    bars = ax.bar(names, rates, color=['#2E8B57', '#DC143C'], alpha=0.7, edgecolor='black')
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Agreement Rate (%)', fontsize=12)
    ax.set_title('Overall Agreement Rate', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(rates) * 1.2)
    ax.grid(True, alpha=0.3)
    
    # 2. Agreement by Human Preference
    ax = axes[0, 1]
    pref_types = ['left', 'right', 'tie']
    x = np.arange(len(pref_types))
    width = 0.35
    
    all_rates = [all_analysis['pref_agreements'][p]['rate'] * 100 for p in pref_types]
    both_rates = [both_analysis['pref_agreements'][p]['rate'] * 100 for p in pref_types]
    
    bars1 = ax.bar(x - width/2, all_rates, width, label='All Cases', color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, both_rates, width, label='Both Completed', color='#4ECDC4', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars, rates in zip([bars1, bars2], [all_rates, both_rates]):
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(pref_types)
    ax.set_ylabel('Agreement Rate (%)', fontsize=12)
    ax.set_title('Agreement by Human Preference', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Human Vote Distribution
    ax = axes[0, 2]
    vote_types = ['left', 'right', 'tie']
    
    all_counts = [all_analysis['human_votes'].get(v, 0) for v in vote_types]
    both_counts = [both_analysis['human_votes'].get(v, 0) for v in vote_types]
    
    bars1 = ax.bar(x - width/2, all_counts, width, label='All Cases', color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, both_counts, width, label='Both Completed', color='#4ECDC4', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars, counts in zip([bars1, bars2], [all_counts, both_counts]):
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(vote_types)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Human Vote Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. GPT-4o Vote Distribution
    ax = axes[1, 0]
    all_gpt4o_counts = [all_analysis['gpt4o_votes'].get(v, 0) for v in vote_types]
    both_gpt4o_counts = [both_analysis['gpt4o_votes'].get(v, 0) for v in vote_types]
    
    bars1 = ax.bar(x - width/2, all_gpt4o_counts, width, label='All Cases', color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, both_gpt4o_counts, width, label='Both Completed', color='#4ECDC4', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars, counts in zip([bars1, bars2], [all_gpt4o_counts, both_gpt4o_counts]):
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(vote_types)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('GPT-4o Vote Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Confidence Comparison
    ax = axes[1, 1]
    conf_types = ['Overall', 'When Agreeing', 'When Disagreeing']
    all_confs = [all_analysis['avg_confidence'], all_analysis['agree_confidence'], all_analysis['disagree_confidence']]
    both_confs = [both_analysis['avg_confidence'], both_analysis['agree_confidence'], both_analysis['disagree_confidence']]
    
    x_conf = np.arange(len(conf_types))
    bars1 = ax.bar(x_conf - width/2, all_confs, width, label='All Cases', color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x_conf + width/2, both_confs, width, label='Both Completed', color='#4ECDC4', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars, confs in zip([bars1, bars2], [all_confs, both_confs]):
        for bar, conf in zip(bars, confs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{conf:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(x_conf)
    ax.set_xticklabels(conf_types, rotation=45, ha='right')
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title('GPT-4o Confidence Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Summary Statistics Table
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'All Cases', 'Both Completed'],
        ['Total Cases', f'{all_analysis["total_cases"]}', f'{both_analysis["total_cases"]}'],
        ['Agreement Rate', f'{all_analysis["agreement_rate"]*100:.1f}%', f'{both_analysis["agreement_rate"]*100:.1f}%'],
        ['Human Left %', f'{all_analysis["human_votes"]["left"]/all_analysis["total_cases"]*100:.1f}%', f'{both_analysis["human_votes"]["left"]/both_analysis["total_cases"]*100:.1f}%'],
        ['Human Right %', f'{all_analysis["human_votes"]["right"]/all_analysis["total_cases"]*100:.1f}%', f'{both_analysis["human_votes"]["right"]/both_analysis["total_cases"]*100:.1f}%'],
        ['Human Tie %', f'{all_analysis["human_votes"]["tie"]/all_analysis["total_cases"]*100:.1f}%', f'{both_analysis["human_votes"]["tie"]/both_analysis["total_cases"]*100:.1f}%'],
        ['GPT-4o Avg Confidence', f'{all_analysis["avg_confidence"]:.2f}', f'{both_analysis["avg_confidence"]:.2f}'],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center', 
                    colWidths=[0.4, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(3):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('all_vs_both_completed_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison visualization saved as: all_vs_both_completed_comparison.png")

if __name__ == "__main__":
    main()