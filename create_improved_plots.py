#!/usr/bin/env python3
"""
Create improved visualizations for the human preference analysis
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

def normalize_vote(vote):
    """Normalize various vote formats to left/right/tie"""
    if pd.isna(vote):
        return 'unknown'
    
    vote_str = str(vote).lower().strip()
    
    # Handle various left patterns
    if any(pattern in vote_str for pattern in ['left', 'model a', 'a is better', 'a completed']):
        return 'left'
    
    # Handle various right patterns  
    if any(pattern in vote_str for pattern in ['right', 'model b', 'b is better', 'b completed']):
        return 'right'
    
    # Handle tie patterns
    if any(pattern in vote_str for pattern in ['tie', 'equal', 'same', 'both']):
        return 'tie'
    
    return 'unknown'

def main():
    # Load and process data
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
    
    # Set up the plot style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GPT-4o vs Human Evaluation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Agreement rate by human preference type
    ax = axes[0, 0]
    pref_types = ['left', 'right', 'tie']
    agreement_rates = []
    counts = []
    
    for pref_type in pref_types:
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
    
    ax.set_ylabel('Agreement Rate (%)', fontsize=12)
    ax.set_title('Agreement Rate by Human Preference', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(agreement_rates) * 1.2)
    ax.grid(True, alpha=0.3)
    
    # 2. Vote distribution comparison
    ax = axes[0, 1]
    human_votes = Counter(clear_df['normalized_human_vote'])
    gpt4o_votes = Counter(clear_df['normalized_gpt4o_vote'])
    
    vote_types = ['left', 'right', 'tie']
    human_counts = [human_votes.get(v, 0) for v in vote_types]
    gpt4o_counts = [gpt4o_votes.get(v, 0) for v in vote_types]
    
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
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Vote Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Confidence distribution
    ax = axes[1, 0]
    ax.hist(clear_df['gpt4o_confidence'], bins=15, color='#95A5A6', alpha=0.7, edgecolor='black')
    
    avg_confidence = clear_df['gpt4o_confidence'].mean()
    ax.axvline(avg_confidence, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {avg_confidence:.2f}')
    
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('GPT-4o Confidence Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Confidence by agreement status
    ax = axes[1, 1]
    
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
        ax.set_ylabel('Confidence Score', fontsize=12)
        ax.set_title('Confidence by Agreement Status', fontsize=14, fontweight='bold')
        
        # Add mean values as text
        mean_agree = agree_conf.mean()
        mean_disagree = disagree_conf.mean()
        ax.text(1, mean_agree + 0.05, f'{mean_agree:.2f}', ha='center', fontweight='bold')
        ax.text(2, mean_disagree + 0.05, f'{mean_disagree:.2f}', ha='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_gpt4o_human_analysis.png', dpi=300, bbox_inches='tight')
    print("Improved visualization saved as: improved_gpt4o_human_analysis.png")
    
    # Create a summary statistics table
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare summary data
    summary_data = [
        ['Metric', 'Value'],
        ['Total Cases Analyzed', f'{len(clear_df)}'],
        ['Overall Agreement Rate', f'{agreements.sum()}/{len(clear_df)} ({agreements.sum()/len(clear_df)*100:.1f}%)'],
        ['', ''],
        ['Human Preferences', ''],
        ['  Left Agent', f'{human_votes["left"]} ({human_votes["left"]/len(clear_df)*100:.1f}%)'],
        ['  Right Agent', f'{human_votes["right"]} ({human_votes["right"]/len(clear_df)*100:.1f}%)'],
        ['  Tie', f'{human_votes["tie"]} ({human_votes["tie"]/len(clear_df)*100:.1f}%)'],
        ['', ''],
        ['GPT-4o Preferences', ''],
        ['  Left Agent', f'{gpt4o_votes["left"]} ({gpt4o_votes["left"]/len(clear_df)*100:.1f}%)'],
        ['  Right Agent', f'{gpt4o_votes["right"]} ({gpt4o_votes["right"]/len(clear_df)*100:.1f}%)'],
        ['  Tie', f'{gpt4o_votes["tie"]} ({gpt4o_votes["tie"]/len(clear_df)*100:.1f}%)'],
        ['', ''],
        ['Confidence Statistics', ''],
        ['  Average Confidence', f'{avg_confidence:.2f}'],
        ['  When Agreeing', f'{agree_df["gpt4o_confidence"].mean():.2f}'],
        ['  When Disagreeing', f'{disagree_df["gpt4o_confidence"].mean():.2f}'],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='left', loc='center', 
                    colWidths=[0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Style the header row
    for i in range(2):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style category rows
    category_rows = [4, 9, 14]
    for row in category_rows:
        table[(row, 0)].set_facecolor('#BDC3C7')
        table[(row, 0)].set_text_props(weight='bold')
    
    plt.title('GPT-4o vs Human Evaluation Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('summary_statistics_table.png', dpi=300, bbox_inches='tight')
    print("Summary table saved as: summary_statistics_table.png")

if __name__ == "__main__":
    main()