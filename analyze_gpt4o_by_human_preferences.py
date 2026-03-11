#!/usr/bin/env python3
"""
Analyze GPT-4o evaluation results categorized by human preferences instead of 
misleading annotation success flags.
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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
    
    # Handle other responses
    if len(vote_str) < 10 and vote_str not in ['left', 'right', 'tie', 'unknown']:
        return 'other'
    
    return 'unknown'

def main():
    """Analyze evaluations categorized by human preference patterns"""
    
    print("Loading GPT-4o evaluation results...")
    
    # Load the results
    with open('gpt4o_all_tasks_evaluations/evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    print(f"Total evaluations: {len(results)}")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Filter out error cases
    valid_df = df[df['gpt4o_preference'] != 'error'].copy()
    print(f"Valid evaluations: {len(valid_df)}")
    
    # Normalize votes for comparison
    valid_df['normalized_human_vote'] = valid_df['original_vote'].apply(normalize_vote)
    valid_df['normalized_gpt4o_vote'] = valid_df['gpt4o_preference'].str.lower()
    
    # Remove cases where human vote is unknown/unclear
    clear_df = valid_df[valid_df['normalized_human_vote'].isin(['left', 'right', 'tie'])].copy()
    print(f"Cases with clear human votes: {len(clear_df)}")
    
    print("\n=== ANALYSIS BY HUMAN PREFERENCE PATTERNS ===")
    
    # Overall agreement analysis
    agreements = clear_df['normalized_human_vote'] == clear_df['normalized_gpt4o_vote']
    agreement_rate = agreements.sum() / len(clear_df)
    
    print(f"Overall agreement rate: {agreements.sum()}/{len(clear_df)} ({agreement_rate*100:.1f}%)")
    
    # Vote distribution
    print("\n=== Vote Distribution ===")
    human_votes = Counter(clear_df['normalized_human_vote'])
    gpt4o_votes = Counter(clear_df['normalized_gpt4o_vote'])
    
    print(f"Human votes: {dict(human_votes)}")
    print(f"GPT-4o votes: {dict(gpt4o_votes)}")
    
    # Agreement by human preference type
    print("\n=== Agreement by Human Preference Type ===")
    
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
    print("=== GPT-4o Confidence Analysis ===")
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Agreement rate by human preference type
    ax = axes[0, 0]
    pref_data = []
    for pref_type in ['left', 'right', 'tie']:
        type_df = clear_df[clear_df['normalized_human_vote'] == pref_type]
        if len(type_df) > 0:
            type_agreements = type_df['normalized_human_vote'] == type_df['normalized_gpt4o_vote']
            pref_data.append({
                'Preference': f'Human chose {pref_type}',
                'Agreement Rate': type_agreements.sum() / len(type_df) * 100,
                'Count': len(type_df)
            })
    
    pref_df_plot = pd.DataFrame(pref_data)
    bars = ax.bar(pref_df_plot['Preference'], pref_df_plot['Agreement Rate'])
    ax.set_ylabel('Agreement Rate (%)')
    ax.set_title('GPT-4o Agreement by Human Preference Type')
    ax.set_xticklabels(pref_df_plot['Preference'], rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, pref_df_plot['Count']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'n={count}', ha='center', va='bottom')
    
    # 2. Vote distribution comparison
    ax = axes[0, 1]
    vote_types = ['left', 'right', 'tie']
    human_counts = [human_votes.get(v, 0) for v in vote_types]
    gpt4o_counts = [gpt4o_votes.get(v, 0) for v in vote_types]
    
    x = range(len(vote_types))
    width = 0.35
    ax.bar([i - width/2 for i in x], human_counts, width, label='Human', alpha=0.8)
    ax.bar([i + width/2 for i in x], gpt4o_counts, width, label='GPT-4o', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(vote_types)
    ax.set_ylabel('Count')
    ax.set_title('Vote Distribution: Human vs GPT-4o')
    ax.legend()
    
    # 3. Confidence distribution
    ax = axes[1, 0]
    ax.hist(clear_df['gpt4o_confidence'], bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title('GPT-4o Confidence Distribution')
    ax.axvline(avg_confidence, color='red', linestyle='--', label=f'Mean: {avg_confidence:.2f}')
    ax.legend()
    
    # 4. Confidence by agreement status
    ax = axes[1, 1]
    if len(agree_df) > 0 and len(disagree_df) > 0:
        confidence_data = [
            agree_df['gpt4o_confidence'].values,
            disagree_df['gpt4o_confidence'].values
        ]
        
        box_plot = ax.boxplot(confidence_data, labels=['Agree', 'Disagree'])
        ax.set_ylabel('Confidence Score')
        ax.set_title('GPT-4o Confidence by Agreement with Human')
        
        # Add mean markers
        means = [agree_df['gpt4o_confidence'].mean(), disagree_df['gpt4o_confidence'].mean()]
        ax.plot([1, 2], means, 'ro', markersize=8, label='Mean')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('gpt4o_human_preference_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as: gpt4o_human_preference_analysis.png")
    
    # Disagreement analysis - cases where GPT-4o and humans strongly disagree
    print("\n=== High-Confidence Disagreement Cases ===")
    high_conf_disagree = disagree_df[disagree_df['gpt4o_confidence'] >= 0.8].head(5)
    
    for _, row in high_conf_disagree.iterrows():
        print(f"\nCase: {row['case_id']}")
        print(f"Human preference: {row['normalized_human_vote']}")
        print(f"GPT-4o preference: {row['normalized_gpt4o_vote']} (confidence: {row['gpt4o_confidence']:.2f})")
        print(f"Reasoning: {row['gpt4o_reasoning'][:200]}...")
    
    # Save detailed analysis
    analysis_df = clear_df[[
        'case_id', 'left_task', 'right_task',
        'normalized_human_vote', 'normalized_gpt4o_vote', 
        'gpt4o_confidence', 'gpt4o_reasoning'
    ]].copy()
    analysis_df['agreement'] = analysis_df['normalized_human_vote'] == analysis_df['normalized_gpt4o_vote']
    
    analysis_df.to_csv('gpt4o_human_preference_detailed_analysis.csv', index=False)
    print(f"\nDetailed analysis saved as: gpt4o_human_preference_detailed_analysis.csv")
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Total valid cases analyzed: {len(clear_df)}")
    print(f"Overall agreement rate: {agreement_rate*100:.1f}%")
    print(f"Human left preference: {human_votes['left']} ({human_votes['left']/len(clear_df)*100:.1f}%)")
    print(f"Human right preference: {human_votes['right']} ({human_votes['right']/len(clear_df)*100:.1f}%)")  
    print(f"Human tie preference: {human_votes['tie']} ({human_votes['tie']/len(clear_df)*100:.1f}%)")
    print(f"GPT-4o average confidence: {avg_confidence:.2f}")

if __name__ == "__main__":
    main()