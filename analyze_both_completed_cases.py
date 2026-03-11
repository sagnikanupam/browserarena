#!/usr/bin/env python3
"""
Create the same analysis and plots but only for cases where both agents 
actually completed their tasks successfully
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
    """Analyze evaluations for cases where both agents completed tasks"""
    
    print("Loading both-completed cases...")
    
    # Load the both-completed cases
    both_completed_df = pd.read_csv('both_agents_completed_cases.csv')
    
    print(f"Total both-completed cases: {len(both_completed_df)}")
    
    # Normalize votes for comparison
    both_completed_df['normalized_human_vote'] = both_completed_df['original_vote'].apply(normalize_vote)
    both_completed_df['normalized_gpt4o_vote'] = both_completed_df['gpt4o_preference'].str.lower()
    
    # Remove cases where human vote is unknown/unclear
    clear_df = both_completed_df[both_completed_df['normalized_human_vote'].isin(['left', 'right', 'tie'])].copy()
    print(f"Cases with clear human votes: {len(clear_df)}")
    
    print("\n=== ANALYSIS: BOTH AGENTS COMPLETED TASKS ===")
    
    # Overall agreement analysis
    agreements = clear_df['normalized_human_vote'] == clear_df['normalized_gpt4o_vote']
    agreement_rate = agreements.sum() / len(clear_df)
    
    print(f"Overall agreement rate: {agreements.sum()}/{len(clear_df)} ({agreement_rate*100:.1f}%)")
    
    # Vote distribution
    print("\n=== Vote Distribution (Both Completed) ===")
    human_votes = Counter(clear_df['normalized_human_vote'])
    gpt4o_votes = Counter(clear_df['normalized_gpt4o_vote'])
    
    print(f"Human votes: {dict(human_votes)}")
    print(f"GPT-4o votes: {dict(gpt4o_votes)}")
    
    # Agreement by human preference type
    print("\n=== Agreement by Human Preference Type (Both Completed) ===")
    
    for pref_type in ['left', 'right', 'tie']:
        type_df = clear_df[clear_df['normalized_human_vote'] == pref_type]
        if len(type_df) > 0:
            type_agreements = type_df['normalized_human_vote'] == type_df['normalized_gpt4o_vote']
            type_rate = type_agreements.sum() / len(type_df)
            print(f"Human preferred {pref_type}: {type_agreements.sum()}/{len(type_df)} ({type_rate*100:.1f}%) agreement")
            
            # Show what GPT-4o said when humans preferred this option
            gpt4o_responses = Counter(type_df['normalized_gpt4o_vote'])
            print(f"  GPT-4o responses: {dict(gpt4o_responses)}")
            print()
    
    # Confidence analysis
    print("=== GPT-4o Confidence Analysis (Both Completed) ===")
    avg_confidence = clear_df['gpt4o_confidence'].mean()
    print(f"Average confidence: {avg_confidence:.2f}")
    
    # Confidence by agreement
    agree_df = clear_df[agreements]
    disagree_df = clear_df[~agreements]
    
    if len(agree_df) > 0 and len(disagree_df) > 0:
        print(f"Average confidence when agreeing: {agree_df['gpt4o_confidence'].mean():.2f}")
        print(f"Average confidence when disagreeing: {disagree_df['gpt4o_confidence'].mean():.2f}")
    
    # Confidence by GPT-4o vote type
    print("\nConfidence by GPT-4o vote type:")
    for vote_type in ['left', 'right', 'tie']:
        vote_df = clear_df[clear_df['normalized_gpt4o_vote'] == vote_type]
        if len(vote_df) > 0:
            print(f"  {vote_type}: {vote_df['gpt4o_confidence'].mean():.2f} (n={len(vote_df)})")
    
    # Create visualizations
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GPT-4o vs Human Analysis: Both Agents Completed Tasks', fontsize=16, fontweight='bold')
    
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
    ax.set_title('Agreement Rate by Human Preference\n(Both Agents Completed)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(agreement_rates) * 1.2 if max(agreement_rates) > 0 else 100)
    ax.grid(True, alpha=0.3)
    
    # 2. Vote distribution comparison
    ax = axes[0, 1]
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
    ax.set_title('Vote Distribution Comparison\n(Both Agents Completed)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Confidence distribution
    ax = axes[1, 0]
    ax.hist(clear_df['gpt4o_confidence'], bins=15, color='#95A5A6', alpha=0.7, edgecolor='black')
    
    ax.axvline(avg_confidence, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {avg_confidence:.2f}')
    
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('GPT-4o Confidence Distribution\n(Both Agents Completed)', fontsize=14, fontweight='bold')
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
        ax.set_title('Confidence by Agreement Status\n(Both Agents Completed)', fontsize=14, fontweight='bold')
        
        # Add mean values as text
        mean_agree = agree_conf.mean()
        mean_disagree = disagree_conf.mean()
        ax.text(1, mean_agree + 0.05, f'{mean_agree:.2f}', ha='center', fontweight='bold')
        ax.text(2, mean_disagree + 0.05, f'{mean_disagree:.2f}', ha='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('both_completed_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as: both_completed_analysis.png")
    
    # Create a comparison summary table
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare summary data
    summary_data = [
        ['Metric', 'Both Completed'],
        ['Total Cases Analyzed', f'{len(clear_df)}'],
        ['Overall Agreement Rate', f'{agreements.sum()}/{len(clear_df)} ({agreement_rate*100:.1f}%)'],
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
        ['  When Agreeing', f'{agree_df["gpt4o_confidence"].mean():.2f}' if len(agree_df) > 0 else 'N/A'],
        ['  When Disagreeing', f'{disagree_df["gpt4o_confidence"].mean():.2f}' if len(disagree_df) > 0 else 'N/A'],
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
    
    plt.title('Both Agents Completed Tasks - Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('both_completed_summary_table.png', dpi=300, bbox_inches='tight')
    print("Summary table saved as: both_completed_summary_table.png")
    
    # Save detailed analysis
    detailed_df = clear_df[[
        'case_id', 'left_task', 'right_task',
        'normalized_human_vote', 'normalized_gpt4o_vote', 
        'gpt4o_confidence', 'gpt4o_reasoning'
    ]].copy()
    detailed_df['agreement'] = detailed_df['normalized_human_vote'] == detailed_df['normalized_gpt4o_vote']
    
    detailed_df.to_csv('both_completed_detailed_analysis.csv', index=False)
    print("Detailed analysis saved as: both_completed_detailed_analysis.csv")
    
    # Print disagreement examples
    print("\n=== High-Confidence Disagreement Cases (Both Completed) ===")
    high_conf_disagree = disagree_df[disagree_df['gpt4o_confidence'] >= 0.8].head(3)
    
    for _, row in high_conf_disagree.iterrows():
        print(f"\nCase: {row['case_id']}")
        print(f"Human preference: {row['normalized_human_vote']}")
        print(f"GPT-4o preference: {row['normalized_gpt4o_vote']} (confidence: {row['gpt4o_confidence']:.2f})")
        print(f"Reasoning: {row['gpt4o_reasoning'][:150]}...")

if __name__ == "__main__":
    main()