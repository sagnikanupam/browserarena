#!/usr/bin/env python3
"""
Create paper-ready individual plots for publication
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Set publication-ready style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8
})

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

def create_agreement_by_preference_plot(clear_df, title_suffix="", filename_suffix=""):
    """Create agreement rate by human preference plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pref_types = ['Left', 'Right', 'Tie']
    agreement_rates = []
    counts = []
    
    for pref_type in ['left', 'right', 'tie']:
        type_df = clear_df[clear_df['normalized_human_vote'] == pref_type]
        if len(type_df) > 0:
            type_agreements = type_df['normalized_human_vote'] == type_df['normalized_gpt4o_vote']
            agreement_rates.append(type_agreements.sum() / len(type_df) * 100)
            counts.append(len(type_df))
        else:
            agreement_rates.append(0)
            counts.append(0)
    
    colors = ['#2E8B57', '#DC143C', '#4682B4']
    bars = ax.bar(pref_types, agreement_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'n={count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Agreement Rate (%)', fontweight='bold')
    ax.set_xlabel('Human Preference', fontweight='bold')
    ax.set_title(f'GPT-4o Agreement Rate by Human Preference{title_suffix}', fontweight='bold')
    ax.set_ylim(0, max(agreement_rates) * 1.15 if max(agreement_rates) > 0 else 100)
    ax.grid(True, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, agreement_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{rate:.1f}%', ha='center', va='center', fontweight='bold', color='white', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'paper_agreement_by_preference{filename_suffix}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'paper_agreement_by_preference{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_vote_distribution_plot(clear_df, title_suffix="", filename_suffix=""):
    """Create vote distribution comparison plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    human_votes = Counter(clear_df['normalized_human_vote'])
    gpt4o_votes = Counter(clear_df['normalized_gpt4o_vote'])
    
    vote_types = ['Left', 'Right', 'Tie']
    human_counts = [human_votes.get(v.lower(), 0) for v in vote_types]
    gpt4o_counts = [gpt4o_votes.get(v.lower(), 0) for v in vote_types]
    
    x = np.arange(len(vote_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, human_counts, width, label='Human Evaluators', 
                   color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, gpt4o_counts, width, label='GPT-4o', 
                   color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(vote_types)
    ax.set_ylabel('Number of Cases', fontweight='bold')
    ax.set_xlabel('Preference', fontweight='bold')
    ax.set_title(f'Vote Distribution Comparison{title_suffix}', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'paper_vote_distribution{filename_suffix}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'paper_vote_distribution{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confidence_distribution_plot(clear_df, title_suffix="", filename_suffix=""):
    """Create confidence distribution plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    avg_confidence = clear_df['gpt4o_confidence'].mean()
    
    n, bins, patches = ax.hist(clear_df['gpt4o_confidence'], bins=15, color='#95A5A6', 
                              alpha=0.7, edgecolor='black', linewidth=1.2)
    
    ax.axvline(avg_confidence, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {avg_confidence:.2f}')
    
    ax.set_xlabel('Confidence Score', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'GPT-4o Confidence Distribution{title_suffix}', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'paper_confidence_distribution{filename_suffix}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'paper_confidence_distribution{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confidence_by_agreement_plot(clear_df, title_suffix="", filename_suffix=""):
    """Create confidence by agreement status plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    agreements = clear_df['normalized_human_vote'] == clear_df['normalized_gpt4o_vote']
    agree_df = clear_df[agreements]
    disagree_df = clear_df[~agreements]
    
    if len(agree_df) > 0 and len(disagree_df) > 0:
        agree_conf = agree_df['gpt4o_confidence'].values
        disagree_conf = disagree_df['gpt4o_confidence'].values
        
        # Create box plots
        bp = ax.boxplot([agree_conf, disagree_conf], positions=[1, 2], widths=0.6, 
                       patch_artist=True, showfliers=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
        
        # Color the boxes differently
        colors = ['#58D68D', '#F1948A']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Agreement', 'Disagreement'])
        ax.set_ylabel('Confidence Score', fontweight='bold')
        ax.set_xlabel('Agreement Status', fontweight='bold')
        ax.set_title(f'GPT-4o Confidence by Agreement Status{title_suffix}', fontweight='bold')
        
        # Add mean values as text
        mean_agree = agree_conf.mean()
        mean_disagree = disagree_conf.mean()
        ax.text(1, mean_agree + 0.05, f'μ = {mean_agree:.2f}', ha='center', fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.text(2, mean_disagree + 0.05, f'μ = {mean_disagree:.2f}', ha='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'paper_confidence_by_agreement{filename_suffix}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'paper_confidence_by_agreement{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_overall_agreement_comparison_plot():
    """Create overall agreement rate comparison"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Load data for comparison
    with open('gpt4o_all_tasks_evaluations/evaluation_results.json', 'r') as f:
        all_results = json.load(f)
    
    all_df = pd.DataFrame(all_results)
    all_valid_df = all_df[all_df['gpt4o_preference'] != 'error'].copy()
    all_valid_df['normalized_human_vote'] = all_valid_df['original_vote'].apply(normalize_vote)
    all_valid_df['normalized_gpt4o_vote'] = all_valid_df['gpt4o_preference'].str.lower()
    all_clear_df = all_valid_df[all_valid_df['normalized_human_vote'].isin(['left', 'right', 'tie'])].copy()
    
    both_completed_df = pd.read_csv('both_agents_completed_cases.csv')
    both_completed_df['normalized_human_vote'] = both_completed_df['original_vote'].apply(normalize_vote)
    both_completed_df['normalized_gpt4o_vote'] = both_completed_df['gpt4o_preference'].str.lower()
    both_clear_df = both_completed_df[both_completed_df['normalized_human_vote'].isin(['left', 'right', 'tie'])].copy()
    
    # Calculate agreement rates
    all_agreements = all_clear_df['normalized_human_vote'] == all_clear_df['normalized_gpt4o_vote']
    both_agreements = both_clear_df['normalized_human_vote'] == both_clear_df['normalized_gpt4o_vote']
    
    all_rate = all_agreements.sum() / len(all_clear_df) * 100
    both_rate = both_agreements.sum() / len(both_clear_df) * 100
    
    categories = ['All Cases', 'Both Agents\nCompleted']
    rates = [all_rate, both_rate]
    counts = [len(all_clear_df), len(both_clear_df)]
    
    colors = ['#2E8B57', '#DC143C']
    bars = ax.bar(categories, rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, rate, count in zip(bars, rates, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%\n(n={count})', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Agreement Rate (%)', fontweight='bold')
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_title('Overall Agreement Rate Comparison', fontweight='bold')
    ax.set_ylim(0, max(rates) * 1.2)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('paper_overall_agreement_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper_overall_agreement_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_preference_agreement_comparison_plot():
    """Create agreement by preference comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Load data for comparison
    with open('gpt4o_all_tasks_evaluations/evaluation_results.json', 'r') as f:
        all_results = json.load(f)
    
    all_df = pd.DataFrame(all_results)
    all_valid_df = all_df[all_df['gpt4o_preference'] != 'error'].copy()
    all_valid_df['normalized_human_vote'] = all_valid_df['original_vote'].apply(normalize_vote)
    all_valid_df['normalized_gpt4o_vote'] = all_valid_df['gpt4o_preference'].str.lower()
    all_clear_df = all_valid_df[all_valid_df['normalized_human_vote'].isin(['left', 'right', 'tie'])].copy()
    
    both_completed_df = pd.read_csv('both_agents_completed_cases.csv')
    both_completed_df['normalized_human_vote'] = both_completed_df['original_vote'].apply(normalize_vote)
    both_completed_df['normalized_gpt4o_vote'] = both_completed_df['gpt4o_preference'].str.lower()
    both_clear_df = both_completed_df[both_completed_df['normalized_human_vote'].isin(['left', 'right', 'tie'])].copy()
    
    # Calculate agreement rates by preference
    pref_types = ['left', 'right', 'tie']
    pref_labels = ['Left', 'Right', 'Tie']
    
    all_rates = []
    both_rates = []
    
    for pref_type in pref_types:
        # All cases
        all_type_df = all_clear_df[all_clear_df['normalized_human_vote'] == pref_type]
        if len(all_type_df) > 0:
            all_type_agreements = all_type_df['normalized_human_vote'] == all_type_df['normalized_gpt4o_vote']
            all_rates.append(all_type_agreements.sum() / len(all_type_df) * 100)
        else:
            all_rates.append(0)
        
        # Both completed cases
        both_type_df = both_clear_df[both_clear_df['normalized_human_vote'] == pref_type]
        if len(both_type_df) > 0:
            both_type_agreements = both_type_df['normalized_human_vote'] == both_type_df['normalized_gpt4o_vote']
            both_rates.append(both_type_agreements.sum() / len(both_type_df) * 100)
        else:
            both_rates.append(0)
    
    x = np.arange(len(pref_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, all_rates, width, label='All Cases', 
                   color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, both_rates, width, label='Both Agents Completed', 
                   color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bars, rates in zip([bars1, bars2], [all_rates, both_rates]):
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(pref_labels)
    ax.set_ylabel('Agreement Rate (%)', fontweight='bold')
    ax.set_xlabel('Human Preference', fontweight='bold')
    ax.set_title('Agreement Rate by Human Preference: Comparison', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('paper_preference_agreement_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper_preference_agreement_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Create all paper-ready plots"""
    
    print("Creating paper-ready plots...")
    
    # Load all cases data
    with open('gpt4o_all_tasks_evaluations/evaluation_results.json', 'r') as f:
        all_results = json.load(f)
    
    all_df = pd.DataFrame(all_results)
    all_valid_df = all_df[all_df['gpt4o_preference'] != 'error'].copy()
    all_valid_df['normalized_human_vote'] = all_valid_df['original_vote'].apply(normalize_vote)
    all_valid_df['normalized_gpt4o_vote'] = all_valid_df['gpt4o_preference'].str.lower()
    all_clear_df = all_valid_df[all_valid_df['normalized_human_vote'].isin(['left', 'right', 'tie'])].copy()
    
    # Load both completed cases data
    both_completed_df = pd.read_csv('both_agents_completed_cases.csv')
    both_completed_df['normalized_human_vote'] = both_completed_df['original_vote'].apply(normalize_vote)
    both_completed_df['normalized_gpt4o_vote'] = both_completed_df['gpt4o_preference'].str.lower()
    both_clear_df = both_completed_df[both_completed_df['normalized_human_vote'].isin(['left', 'right', 'tie'])].copy()
    
    # Create individual plots for all cases
    print("Creating plots for all cases...")
    create_agreement_by_preference_plot(all_clear_df, "", "_all_cases")
    create_vote_distribution_plot(all_clear_df, "", "_all_cases")
    create_confidence_distribution_plot(all_clear_df, "", "_all_cases")
    create_confidence_by_agreement_plot(all_clear_df, "", "_all_cases")
    
    # Create individual plots for both completed cases
    print("Creating plots for both completed cases...")
    create_agreement_by_preference_plot(both_clear_df, " (Both Agents Completed)", "_both_completed")
    create_vote_distribution_plot(both_clear_df, " (Both Agents Completed)", "_both_completed")
    create_confidence_distribution_plot(both_clear_df, " (Both Agents Completed)", "_both_completed")
    create_confidence_by_agreement_plot(both_clear_df, " (Both Agents Completed)", "_both_completed")
    
    # Create comparison plots
    print("Creating comparison plots...")
    create_overall_agreement_comparison_plot()
    create_preference_agreement_comparison_plot()
    
    print("All paper-ready plots created successfully!")
    print("\nFiles generated:")
    
    plot_types = [
        "agreement_by_preference",
        "vote_distribution", 
        "confidence_distribution",
        "confidence_by_agreement"
    ]
    
    for plot_type in plot_types:
        print(f"  paper_{plot_type}_all_cases.pdf/.png")
        print(f"  paper_{plot_type}_both_completed.pdf/.png")
    
    print("  paper_overall_agreement_comparison.pdf/.png")
    print("  paper_preference_agreement_comparison.pdf/.png")

if __name__ == "__main__":
    main()