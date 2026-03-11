#!/usr/bin/env python3
"""
Analyze GPT-4o evaluation results for ALL tasks (not just successful ones)
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def main():
    """Analyze all task evaluations"""
    
    print("Loading GPT-4o evaluation results for all tasks...")
    
    # Load the results
    with open('gpt4o_all_tasks_evaluations/evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    print(f"Total evaluations: {len(results)}")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Filter out error cases
    valid_df = df[df['gpt4o_preference'] != 'error']
    print(f"Valid evaluations: {len(valid_df)}")
    
    # Overall agreement analysis
    print("\n=== Overall Agreement Analysis ===")
    
    # Normalize votes for comparison
    valid_df['normalized_human_vote'] = valid_df['original_vote'].str.lower().str.replace(r'.*left.*', 'left', regex=True).str.replace(r'.*right.*', 'right', regex=True)
    valid_df['normalized_gpt4o_vote'] = valid_df['gpt4o_preference'].str.lower()
    
    # Agreement calculation
    agreements = valid_df['normalized_human_vote'] == valid_df['normalized_gpt4o_vote']
    agreement_rate = agreements.sum() / len(valid_df)
    
    print(f"Overall agreement rate: {agreements.sum()}/{len(valid_df)} ({agreement_rate*100:.1f}%)")
    
    # Vote distribution
    print("\n=== Vote Distribution ===")
    human_votes = Counter(valid_df['normalized_human_vote'])
    gpt4o_votes = Counter(valid_df['normalized_gpt4o_vote'])
    
    print(f"Human votes: {dict(human_votes)}")
    print(f"GPT-4o votes: {dict(gpt4o_votes)}")
    
    # Agreement by success pattern
    print("\n=== Agreement by Task Success Pattern ===")
    
    success_patterns = []
    for _, row in valid_df.iterrows():
        if row['both_succeeded']:
            pattern = "Both succeeded"
        elif row['left_success'] and not row['right_success']:
            pattern = "Left only succeeded"
        elif not row['left_success'] and row['right_success']:
            pattern = "Right only succeeded"
        else:
            pattern = "Neither succeeded"
        success_patterns.append(pattern)
    
    valid_df['success_pattern'] = success_patterns
    
    for pattern in ["Both succeeded", "Left only succeeded", "Right only succeeded", "Neither succeeded"]:
        pattern_df = valid_df[valid_df['success_pattern'] == pattern]
        if len(pattern_df) > 0:
            pattern_agreements = pattern_df['normalized_human_vote'] == pattern_df['normalized_gpt4o_vote']
            pattern_rate = pattern_agreements.sum() / len(pattern_df)
            print(f"{pattern}: {pattern_agreements.sum()}/{len(pattern_df)} ({pattern_rate*100:.1f}%)")
    
    # Confidence analysis
    print("\n=== GPT-4o Confidence Analysis ===")
    avg_confidence = valid_df['gpt4o_confidence'].mean()
    print(f"Average confidence: {avg_confidence:.2f}")
    
    # Confidence by agreement
    agree_df = valid_df[agreements]
    disagree_df = valid_df[~agreements]
    
    print(f"Average confidence when agreeing: {agree_df['gpt4o_confidence'].mean():.2f}")
    print(f"Average confidence when disagreeing: {disagree_df['gpt4o_confidence'].mean():.2f}")
    
    # Confidence by vote type
    print("\nConfidence by GPT-4o vote type:")
    for vote_type in ['left', 'right', 'tie']:
        vote_df = valid_df[valid_df['normalized_gpt4o_vote'] == vote_type]
        if len(vote_df) > 0:
            print(f"  {vote_type}: {vote_df['gpt4o_confidence'].mean():.2f} (n={len(vote_df)})")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Agreement rate by success pattern
    ax = axes[0, 0]
    pattern_data = []
    for pattern in ["Both succeeded", "Left only succeeded", "Right only succeeded", "Neither succeeded"]:
        pattern_df = valid_df[valid_df['success_pattern'] == pattern]
        if len(pattern_df) > 0:
            pattern_agreements = pattern_df['normalized_human_vote'] == pattern_df['normalized_gpt4o_vote']
            pattern_data.append({
                'Pattern': pattern,
                'Agreement Rate': pattern_agreements.sum() / len(pattern_df) * 100,
                'Count': len(pattern_df)
            })
    
    pattern_df_plot = pd.DataFrame(pattern_data)
    bars = ax.bar(pattern_df_plot['Pattern'], pattern_df_plot['Agreement Rate'])
    ax.set_ylabel('Agreement Rate (%)')
    ax.set_title('GPT-4o vs Human Agreement by Success Pattern')
    ax.set_xticklabels(pattern_df_plot['Pattern'], rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, pattern_df_plot['Count']):
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
    ax.hist(valid_df['gpt4o_confidence'], bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title('GPT-4o Confidence Distribution')
    ax.axvline(avg_confidence, color='red', linestyle='--', label=f'Mean: {avg_confidence:.2f}')
    ax.legend()
    
    # 4. Confidence by agreement status
    ax = axes[1, 1]
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
    plt.savefig('gpt4o_all_tasks_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as: gpt4o_all_tasks_analysis.png")
    
    # Save detailed analysis CSV
    analysis_df = valid_df[[
        'case_id', 'left_task', 'right_task', 
        'left_success', 'right_success', 'success_pattern',
        'normalized_human_vote', 'normalized_gpt4o_vote', 
        'gpt4o_confidence', 'gpt4o_reasoning'
    ]]
    analysis_df['agreement'] = analysis_df['normalized_human_vote'] == analysis_df['normalized_gpt4o_vote']
    
    analysis_df.to_csv('gpt4o_all_tasks_detailed_analysis.csv', index=False)
    print("Detailed analysis saved as: gpt4o_all_tasks_detailed_analysis.csv")
    
    # Print interesting disagreement cases
    print("\n=== Notable Disagreement Cases ===")
    high_conf_disagree = disagree_df[disagree_df['gpt4o_confidence'] >= 0.8].head(5)
    
    for _, row in high_conf_disagree.iterrows():
        print(f"\nCase: {row['case_id']}")
        print(f"Success: Left={row['left_success']}, Right={row['right_success']}")
        print(f"Human vote: {row['normalized_human_vote']}")
        print(f"GPT-4o vote: {row['normalized_gpt4o_vote']} (confidence: {row['gpt4o_confidence']:.2f})")
        print(f"Reasoning: {row['gpt4o_reasoning'][:200]}...")

if __name__ == "__main__":
    main()