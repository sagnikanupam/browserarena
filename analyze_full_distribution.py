#!/usr/bin/env python3
"""
Analyze the full distribution of human votes instead of just majority vote
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']

def calculate_distribution_metrics(agent1_votes, agent2_votes, tie_votes):
    """Calculate various metrics from vote distribution"""
    total = agent1_votes + agent2_votes + tie_votes
    if total == 0:
        return {}
    
    # Basic percentages
    agent1_pct = agent1_votes / total * 100
    agent2_pct = agent2_votes / total * 100
    tie_pct = tie_votes / total * 100
    
    # Entropy (uncertainty in the distribution)
    probs = np.array([agent1_votes, agent2_votes, tie_votes]) / total
    probs = probs[probs > 0]  # Remove zeros for log
    entropy = -np.sum(probs * np.log2(probs))
    
    # Consensus strength (how concentrated votes are)
    consensus_strength = max(agent1_pct, agent2_pct, tie_pct)
    
    # Decisiveness (how many didn't choose tie)
    decisiveness = (agent1_votes + agent2_votes) / total * 100
    
    # Polarization (how split between agent 1 and 2, ignoring ties)
    if agent1_votes + agent2_votes > 0:
        polarization = min(agent1_votes, agent2_votes) / (agent1_votes + agent2_votes) * 100
    else:
        polarization = 0
    
    return {
        'agent1_pct': agent1_pct,
        'agent2_pct': agent2_pct,
        'tie_pct': tie_pct,
        'entropy': entropy,
        'consensus_strength': consensus_strength,
        'decisiveness': decisiveness,
        'polarization': polarization,
        'total_votes': total
    }

def main():
    """Main analysis of full vote distributions"""
    print("Loading V2 human evaluation data...")
    
    # Load the data with V2 distributions
    df = pd.read_csv('/Users/davisbrown/browserarena/agreement-results/three_way_comparison_v3.csv')
    print(f"Loaded data for {len(df)} questions")
    
    # Calculate metrics for each question
    metrics_list = []
    for idx, row in df.iterrows():
        agent1 = int(row['V2_Agent1_Votes'])
        agent2 = int(row['V2_Agent2_Votes'])
        tie = int(row['V2_Tie_Votes'])
        
        metrics = calculate_distribution_metrics(agent1, agent2, tie)
        metrics['Question'] = row['Question']
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Print summary statistics
    print("\n=== HUMAN VOTE DISTRIBUTION SUMMARY ===")
    print(f"Total evaluators per question: {metrics_df['total_votes'].iloc[0]}")
    print(f"\nAverage vote distribution:")
    print(f"  Agent 1: {metrics_df['agent1_pct'].mean():.1f}% (±{metrics_df['agent1_pct'].std():.1f}%)")
    print(f"  Agent 2: {metrics_df['agent2_pct'].mean():.1f}% (±{metrics_df['agent2_pct'].std():.1f}%)")
    print(f"  Tie: {metrics_df['tie_pct'].mean():.1f}% (±{metrics_df['tie_pct'].std():.1f}%)")
    print(f"\nConsensus metrics:")
    print(f"  Average Consensus Strength: {metrics_df['consensus_strength'].mean():.1f}%")
    print(f"  Average Decisiveness: {metrics_df['decisiveness'].mean():.1f}%")
    print(f"  Average Entropy: {metrics_df['entropy'].mean():.2f} bits")
    print(f"  Average Polarization: {metrics_df['polarization'].mean():.1f}%")
    
    # Create visualizations
    create_distribution_plots(metrics_df, df)
    
    # Analyze AI model alignment with distributions
    analyze_ai_alignment_with_distributions(df, metrics_df)
    
    # Save detailed results
    full_analysis = pd.merge(df, metrics_df[['Question', 'entropy', 'consensus_strength', 
                                             'decisiveness', 'polarization']], on='Question')
    full_analysis.to_csv('human_vote_distribution_full_analysis.csv', index=False)
    print("\nDetailed analysis saved to: human_vote_distribution_full_analysis.csv")

def create_distribution_plots(metrics_df, full_df):
    """Create visualizations of vote distributions"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Stacked bar chart of vote distributions
    ax = axes[0, 0]
    
    # Sort by consensus strength
    sorted_df = metrics_df.sort_values('consensus_strength', ascending=False)
    questions = sorted_df['Question']
    x = np.arange(len(questions))
    
    # Create stacked bar chart
    p1 = ax.bar(x, sorted_df['agent1_pct'], label='Agent 1', color='#2E86AB', alpha=0.8)
    p2 = ax.bar(x, sorted_df['agent2_pct'], bottom=sorted_df['agent1_pct'], 
                label='Agent 2', color='#E94B3C', alpha=0.8)
    p3 = ax.bar(x, sorted_df['tie_pct'], 
                bottom=sorted_df['agent1_pct'] + sorted_df['agent2_pct'],
                label='Tie', color='#6B7280', alpha=0.8)
    
    ax.set_ylabel('Vote Percentage', fontsize=12)
    ax.set_title('Human Vote Distribution by Question\n(Sorted by Consensus Strength)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace('Q', '') for q in questions], rotation=0)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add consensus strength line
    ax2 = ax.twinx()
    ax2.plot(x, sorted_df['consensus_strength'], 'k--', alpha=0.5, linewidth=2, label='Consensus')
    ax2.set_ylabel('Consensus Strength (%)', fontsize=12)
    
    # 2. Entropy vs Consensus Strength scatter
    ax = axes[0, 1]
    scatter = ax.scatter(metrics_df['consensus_strength'], metrics_df['entropy'], 
                        s=100, alpha=0.6, c=metrics_df['tie_pct'],
                        cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Add question labels for outliers
    for _, row in metrics_df.iterrows():
        if row['entropy'] > 1.4 or row['consensus_strength'] < 45:
            ax.annotate(row['Question'], (row['consensus_strength'], row['entropy']), 
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Consensus Strength (%)', fontsize=12)
    ax.set_ylabel('Entropy (bits)', fontsize=12)
    ax.set_title('Consensus Strength vs Vote Uncertainty', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Tie Vote %', fontsize=10)
    
    # Add theoretical maximum entropy line
    max_entropy = -np.log2(1/3)  # Maximum entropy for 3 choices
    ax.axhline(max_entropy, color='red', linestyle=':', alpha=0.5, label=f'Max entropy: {max_entropy:.2f}')
    ax.legend()
    
    # 3. Tie percentage vs Decisiveness
    ax = axes[1, 0]
    
    # Create histogram
    ax.hist(metrics_df['tie_pct'], bins=12, edgecolor='black', alpha=0.7, color='#6B7280')
    ax.axvline(metrics_df['tie_pct'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {metrics_df["tie_pct"].mean():.1f}%')
    ax.axvline(metrics_df['tie_pct'].median(), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {metrics_df["tie_pct"].median():.1f}%')
    
    ax.set_xlabel('Tie Vote Percentage', fontsize=12)
    ax.set_ylabel('Number of Questions', fontsize=12)
    ax.set_title('Distribution of Tie Vote Percentages', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # 4. Agreement patterns - comparing Human majority vs AI models
    ax = axes[1, 1]
    
    # Calculate how often each evaluator agrees with human distribution
    evaluators = []
    agreement_scores = []
    
    for evaluator, col in [('Baseline', 'Original_Baseline'), ('GPT-4o', 'GPT4o_Exact')]:
        if col in full_df.columns:
            scores = []
            for idx, row in full_df.iterrows():
                if pd.notna(row[col]):
                    choice = row[col]
                    agent1 = row['V2_Agent1_Votes']
                    agent2 = row['V2_Agent2_Votes']
                    tie = row['V2_Tie_Votes']
                    total = agent1 + agent2 + tie
                    
                    if choice == 'Agent 1':
                        score = agent1 / total * 100
                    elif choice == 'Agent 2':
                        score = agent2 / total * 100
                    elif choice == 'Tie':
                        score = tie / total * 100
                    else:
                        score = 0
                    scores.append(score)
            
            if scores:
                evaluators.append(evaluator)
                agreement_scores.append(np.mean(scores))
    
    if evaluators:
        bars = ax.bar(evaluators, agreement_scores, color=['#2E86AB', '#C73E1D'], alpha=0.8)
        ax.set_ylabel('Average % of Humans Who Agreed', fontsize=12)
        ax.set_title('How Well Evaluators Align with Human Vote Distribution', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.axhline(50, color='black', linestyle='--', alpha=0.5, label='Random chance')
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('human_vote_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('human_vote_distribution_analysis.pdf', dpi=300, bbox_inches='tight')
    print("\nVote distribution plots saved!")

def analyze_ai_alignment_with_distributions(df, metrics_df):
    """Analyze how AI model choices align with human vote distributions"""
    
    print("\n=== AI MODEL ALIGNMENT WITH HUMAN DISTRIBUTIONS ===")
    
    results_list = []
    
    for model, col in [('Baseline', 'Original_Baseline'), ('GPT-4o', 'GPT4o_Exact')]:
        if col not in df.columns:
            continue
            
        print(f"\n{model} Analysis:")
        
        # Calculate alignment for each question
        alignment_scores = []
        high_consensus_alignment = []
        low_consensus_alignment = []
        
        for idx, row in df.iterrows():
            if pd.notna(row[col]):
                choice = row[col]
                agent1 = row['V2_Agent1_Votes']
                agent2 = row['V2_Agent2_Votes']
                tie = row['V2_Tie_Votes']
                total = agent1 + agent2 + tie
                
                # Get consensus strength for this question
                consensus = metrics_df[metrics_df['Question'] == row['Question']]['consensus_strength'].iloc[0]
                
                # Calculate alignment
                if choice == 'Agent 1':
                    alignment = agent1 / total * 100
                elif choice == 'Agent 2':
                    alignment = agent2 / total * 100
                elif choice == 'Tie':
                    alignment = tie / total * 100
                else:
                    alignment = 0
                
                alignment_scores.append(alignment)
                
                if consensus > 60:  # High consensus questions
                    high_consensus_alignment.append(alignment)
                else:  # Low consensus questions
                    low_consensus_alignment.append(alignment)
                
                results_list.append({
                    'Question': row['Question'],
                    'Model': model,
                    'Model_Choice': choice,
                    'Human_Alignment': alignment,
                    'Consensus_Strength': consensus,
                    'Agent1_Pct': agent1 / total * 100,
                    'Agent2_Pct': agent2 / total * 100,
                    'Tie_Pct': tie / total * 100
                })
        
        # Print statistics
        print(f"  Average alignment with human distribution: {np.mean(alignment_scores):.1f}%")
        print(f"  Alignment on high consensus questions (>60%): {np.mean(high_consensus_alignment):.1f}%")
        print(f"  Alignment on low consensus questions (≤60%): {np.mean(low_consensus_alignment):.1f}%")
        print(f"  Questions where >50% humans agreed with {model}: {sum(s > 50 for s in alignment_scores)}/{len(alignment_scores)}")
        print(f"  Questions where >75% humans agreed with {model}: {sum(s > 75 for s in alignment_scores)}/{len(alignment_scores)}")
    
    # Create detailed comparison plot
    if results_list:
        results_df = pd.DataFrame(results_list)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot alignment by question for each model
        for model in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model].sort_values('Question')
            x = range(len(model_data))
            ax.scatter(x, model_data['Human_Alignment'], label=model, s=100, alpha=0.7)
        
        ax.set_xlabel('Question', fontsize=12)
        ax.set_ylabel('% of Humans Who Agreed with Model Choice', fontsize=12)
        ax.set_title('Model Alignment with Human Vote Distribution by Question', fontsize=14, fontweight='bold')
        ax.axhline(50, color='black', linestyle='--', alpha=0.5, label='50% threshold')
        ax.axhline(33.33, color='gray', linestyle=':', alpha=0.5, label='Random (1/3)')
        ax.set_xticks(range(len(model_data)))
        ax.set_xticklabels([q.replace('Q', '') for q in model_data['Question']], rotation=0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig('model_alignment_with_human_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig('model_alignment_with_human_distribution.pdf', dpi=300, bbox_inches='tight')
        print("\nModel alignment plot saved!")
        
        # Save detailed results
        results_df.to_csv('model_human_distribution_alignment.csv', index=False)

if __name__ == "__main__":
    main()