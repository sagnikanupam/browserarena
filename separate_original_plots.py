#!/usr/bin/env python3
"""
Separate the original improved_gpt4o_human_analysis.png into individual paper-ready plots
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
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

def main():
    """Recreate the original analysis plots as separate figures"""
    
    # Load and process data (same as original)
    with open('gpt4o_all_tasks_evaluations/evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    valid_df = df[df['gpt4o_preference'] != 'error'].copy()
    
    valid_df['normalized_human_vote'] = valid_df['original_vote'].apply(normalize_vote)
    valid_df['normalized_gpt4o_vote'] = valid_df['gpt4o_preference'].str.lower()
    
    clear_df = valid_df[valid_df['normalized_human_vote'].isin(['left', 'right', 'tie'])].copy()
    
    # Calculate agreement
    agreements = clear_df['normalized_human_vote'] == clear_df['normalized_gpt4o_vote']
    agree_df = clear_df[agreements]
    disagree_df = clear_df[~agreements]
    
    # 1. Agreement rate by human preference type (original style)
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
    
    colors = ['#2E8B57', '#DC143C', '#4682B4']  # Sea green, crimson, steel blue
    bars = ax.bar(pref_types, agreement_rates, color=colors, alpha=0.7, edgecolor='black')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'n={count}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Agreement Rate (%)', fontweight='bold')
    ax.set_xlabel('Human Preference', fontweight='bold')
    ax.set_title('Agreement Rate by Human Preference', fontweight='bold')
    ax.set_ylim(0, max(agreement_rates) * 1.2)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('paper_original_agreement_by_preference.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper_original_agreement_by_preference.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Vote distribution comparison (original style)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    human_votes = Counter(clear_df['normalized_human_vote'])
    gpt4o_votes = Counter(clear_df['normalized_gpt4o_vote'])
    
    vote_types = ['Left', 'Right', 'Tie']
    human_counts = [human_votes.get(v.lower(), 0) for v in vote_types]
    gpt4o_counts = [gpt4o_votes.get(v.lower(), 0) for v in vote_types]
    
    x = np.arange(len(vote_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, human_counts, width, label='Human', color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, gpt4o_counts, width, label='GPT-4o', color='#4ECDC4', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(vote_types)
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_xlabel('Preference', fontweight='bold')
    ax.set_title('Vote Distribution Comparison', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('paper_original_vote_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper_original_vote_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confidence distribution (original style)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    avg_confidence = clear_df['gpt4o_confidence'].mean()
    ax.hist(clear_df['gpt4o_confidence'], bins=15, color='#95A5A6', alpha=0.7, edgecolor='black')
    
    ax.axvline(avg_confidence, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {avg_confidence:.2f}')
    
    ax.set_xlabel('Confidence Score', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('GPT-4o Confidence Distribution', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('paper_original_confidence_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper_original_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Confidence by agreement status (original style with violin + box)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(agree_df) > 0 and len(disagree_df) > 0:
        agree_conf = agree_df['gpt4o_confidence'].values
        disagree_conf = disagree_df['gpt4o_confidence'].values
        
        # Create violin plot for better visualization
        parts = ax.violinplot([agree_conf, disagree_conf], positions=[1, 2], showmeans=True)
        
        # Customize violin plot colors
        for pc, color in zip(parts['bodies'], ['#58D68D', '#F1948A']):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Add box plot overlay
        bp = ax.boxplot([agree_conf, disagree_conf], positions=[1, 2], widths=0.1, 
                       patch_artist=True, showfliers=False)
        
        for patch, color in zip(bp['boxes'], ['#27AE60', '#E74C3C']):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Agree', 'Disagree'])
        ax.set_ylabel('Confidence Score', fontweight='bold')
        ax.set_xlabel('Agreement Status', fontweight='bold')
        ax.set_title('Confidence by Agreement Status', fontweight='bold')
        
        # Add mean values as text
        mean_agree = agree_conf.mean()
        mean_disagree = disagree_conf.mean()
        ax.text(1, mean_agree + 0.05, f'{mean_agree:.2f}', ha='center', fontweight='bold')
        ax.text(2, mean_disagree + 0.05, f'{mean_disagree:.2f}', ha='center', fontweight='bold')
        
        ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('paper_original_confidence_by_agreement.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper_original_confidence_by_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Original plots separated into individual paper-ready figures:")
    print("  paper_original_agreement_by_preference.pdf/.png")
    print("  paper_original_vote_distribution.pdf/.png") 
    print("  paper_original_confidence_distribution.pdf/.png")
    print("  paper_original_confidence_by_agreement.pdf/.png")

if __name__ == "__main__":
    main()