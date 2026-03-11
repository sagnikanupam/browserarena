#!/usr/bin/env python3
"""
Create plots for GPT-4 multisampling vs Survey vs Baseline comparison
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_analysis_data():
    """Load the analysis results"""
    with open('gpt4_threeway_comparison_report.json', 'r') as f:
        report = json.load(f)
    
    with open('gpt4_multisampling_results.json', 'r') as f:
        multisampling = json.load(f)
    
    return report, multisampling

def create_vote_distribution_plot():
    """Create bar plot comparing vote distributions"""
    report, _ = load_analysis_data()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Data
    sources = ['GPT-4', 'Survey', 'Baseline']
    left_pct = [76.0, 52.6, 86.4]
    right_pct = [20.0, 5.3, 9.1]
    tie_pct = [4.0, 42.1, 4.5]
    
    # Create grouped bar plot
    x = np.arange(len(sources))
    width = 0.25
    
    bars1 = ax.bar(x - width, left_pct, width, label='Left/Agent 1', color='#1f77b4')
    bars2 = ax.bar(x, right_pct, width, label='Right/Agent 2', color='#ff7f0e')
    bars3 = ax.bar(x + width, tie_pct, width, label='Tie', color='#2ca02c')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    ax.set_xlabel('Evaluator Source', fontsize=12)
    ax.set_ylabel('Percentage of Votes', fontsize=12)
    ax.set_title('Vote Distribution Comparison: GPT-4 vs Survey vs Baseline', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('vote_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def create_agreement_matrix():
    """Create heatmap showing pairwise agreement rates"""
    report, _ = load_analysis_data()
    
    # Create agreement matrix
    agreement_data = {
        'GPT-4': [100.0, 52.6, 73.7],
        'Survey': [52.6, 100.0, 63.2],
        'Baseline': [73.7, 63.2, 100.0]
    }
    
    df = pd.DataFrame(agreement_data, index=['GPT-4', 'Survey', 'Baseline'])
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', 
                vmin=50, vmax=100, square=True, 
                cbar_kws={'label': 'Agreement Rate (%)'},
                ax=ax)
    
    ax.set_title('Pairwise Agreement Rates Between Evaluators', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('agreement_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_consistency_plot():
    """Create plot showing GPT-4 consistency across samples"""
    _, multisampling = load_analysis_data()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Extract consistency scores
    interactions = []
    consistency_scores = []
    unanimous = []
    
    for int_id, data in sorted(multisampling['evaluations'].items()):
        interactions.append(int_id.replace('interaction_', ''))
        consistency_scores.append(data['consistency']['consistency_score'])
        unanimous.append(1 if data['consistency']['agreement_rate'] == 1.0 else 0)
    
    # Plot 1: Consistency scores
    bars = ax1.bar(range(len(interactions)), consistency_scores, 
                    color=['#2ca02c' if u == 1 else '#ff7f0e' for u in unanimous])
    
    ax1.set_xlabel('Interaction ID', fontsize=12)
    ax1.set_ylabel('Consistency Score', fontsize=12)
    ax1.set_title('GPT-4 Consistency Scores Across 5 Samples', fontsize=14, pad=20)
    ax1.set_xticks(range(len(interactions)))
    ax1.set_xticklabels(interactions, rotation=45, ha='right')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='High consistency threshold')
    
    # Add legend
    green_patch = mpatches.Patch(color='#2ca02c', label='Unanimous')
    orange_patch = mpatches.Patch(color='#ff7f0e', label='Not unanimous')
    ax1.legend(handles=[green_patch, orange_patch])
    
    # Plot 2: Vote distribution for non-unanimous cases
    non_unanimous_data = []
    labels = []
    
    for int_id, data in sorted(multisampling['evaluations'].items()):
        if data['consistency']['agreement_rate'] < 1.0:
            samples = data['samples']
            vote_counts = {'Left': 0, 'Right': 0, 'Tie': 0}
            for vote in samples:
                vote_counts[vote] += 1
            non_unanimous_data.append([vote_counts['Left'], vote_counts['Right'], vote_counts['Tie']])
            labels.append(int_id.replace('interaction_', ''))
    
    if non_unanimous_data:
        non_unanimous_df = pd.DataFrame(non_unanimous_data, columns=['Left', 'Right', 'Tie'], index=labels)
        non_unanimous_df.plot(kind='bar', stacked=True, ax=ax2, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_xlabel('Interaction ID', fontsize=12)
        ax2.set_ylabel('Number of Votes', fontsize=12)
        ax2.set_title('Vote Distribution for Non-Unanimous GPT-4 Decisions', fontsize=14, pad=20)
        ax2.set_xticklabels(labels, rotation=0)
        ax2.legend(title='Vote')
    
    plt.tight_layout()
    plt.savefig('gpt4_consistency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_interaction_comparison_plot():
    """Create detailed comparison plot for each interaction"""
    report, multisampling = load_analysis_data()
    
    # Load additional data
    with open('gpt4_multisampling_detailed.csv', 'r') as f:
        import csv
        reader = csv.DictReader(f)
        detailed_data = list(reader)
    
    # Create subset for visualization (first 10 interactions)
    subset = detailed_data[:10]
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Prepare data
    interactions = []
    gpt4_votes = []
    survey_votes = []
    baseline_votes = []
    
    for row in subset:
        interactions.append(row['Interaction_ID'].replace('interaction_', ''))
        gpt4_votes.append(row['GPT4_Majority'])
        survey_votes.append(row['Survey_Majority'])
        baseline_votes.append(row['Baseline_Vote'])
    
    # Create position arrays
    y_pos = np.arange(len(interactions))
    
    # Define vote to position mapping
    vote_map = {'Left': 0, 'Tie': 1, 'Right': 2}
    colors = {'Left': '#1f77b4', 'Tie': '#2ca02c', 'Right': '#ff7f0e'}
    
    # Plot each source
    for i, (gpt4, survey, baseline) in enumerate(zip(gpt4_votes, survey_votes, baseline_votes)):
        # GPT-4
        ax.scatter(vote_map.get(gpt4, 1), i, s=200, c=colors.get(gpt4, 'gray'), 
                  marker='o', label='GPT-4' if i == 0 else "", edgecolors='black', linewidth=2)
        
        # Survey
        ax.scatter(vote_map.get(survey, 1), i, s=200, c=colors.get(survey, 'gray'), 
                  marker='s', label='Survey' if i == 0 else "", edgecolors='black', linewidth=2)
        
        # Baseline
        ax.scatter(vote_map.get(baseline, 1), i, s=200, c=colors.get(baseline, 'gray'), 
                  marker='^', label='Baseline' if i == 0 else "", edgecolors='black', linewidth=2)
        
        # Connect when all agree
        if gpt4 == survey == baseline:
            ax.plot([vote_map[gpt4]], [i], 'k-', alpha=0.3, linewidth=8, zorder=0)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(interactions)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Left/Agent 1', 'Tie', 'Right/Agent 2'])
    ax.set_xlabel('Vote', fontsize=12)
    ax.set_ylabel('Interaction ID', fontsize=12)
    ax.set_title('Vote Comparison Across Evaluators (First 10 Interactions)', fontsize=14, pad=20)
    ax.legend(loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    
    # Add vertical lines
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=2, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('interaction_vote_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_tie_analysis_plot():
    """Create plot analyzing tie voting patterns"""
    report, _ = load_analysis_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Tie vote breakdown for survey
    tie_cases = report['survey_tie_analysis']
    
    if tie_cases:
        interactions = []
        left_votes = []
        right_votes = []
        tie_votes = []
        
        for case in tie_cases[:8]:  # Show first 8
            interactions.append(case['id'].replace('interaction_', ''))
            # Parse survey breakdown
            breakdown = case['survey_breakdown']
            parts = breakdown.split()
            left_votes.append(int(parts[0].split(':')[1]))
            right_votes.append(int(parts[1].split(':')[1]))
            tie_votes.append(int(parts[2].split(':')[1]))
        
        # Create stacked bar chart
        ind = np.arange(len(interactions))
        width = 0.6
        
        p1 = ax1.bar(ind, left_votes, width, label='Left', color='#1f77b4')
        p2 = ax1.bar(ind, right_votes, width, bottom=left_votes, label='Right', color='#ff7f0e')
        p3 = ax1.bar(ind, tie_votes, width, bottom=np.array(left_votes)+np.array(right_votes), 
                     label='Tie', color='#2ca02c')
        
        ax1.set_xlabel('Interaction ID', fontsize=12)
        ax1.set_ylabel('Number of Survey Votes', fontsize=12)
        ax1.set_title('Survey Vote Breakdown for Tie Majority Cases', fontsize=14, pad=20)
        ax1.set_xticks(ind)
        ax1.set_xticklabels(interactions, rotation=45, ha='right')
        ax1.legend()
    
    # Plot 2: Comparison of tie rates
    sources = ['GPT-4', 'Survey', 'Baseline']
    tie_rates = [4.0, 42.1, 4.5]
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax2.bar(sources, tie_rates, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Percentage of Tie Votes', fontsize=12)
    ax2.set_title('Tie Vote Rates Across Evaluators', fontsize=14, pad=20)
    ax2.set_ylim(0, 50)
    
    # Add horizontal line at survey level
    ax2.axhline(y=42.1, color='red', linestyle='--', alpha=0.5, 
                label='Survey tie rate')
    
    plt.tight_layout()
    plt.savefig('tie_voting_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_infographic():
    """Create a summary infographic with key statistics"""
    report, _ = load_analysis_data()
    
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('GPT-4 vs Human Evaluation: Key Findings', fontsize=20, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Agreement rates circle
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    # Draw agreement circles
    circles_data = [
        ('GPT-4 vs\nSurvey', 52.6, 0.2),
        ('GPT-4 vs\nBaseline', 73.7, 0.5),
        ('Survey vs\nBaseline', 63.2, 0.8)
    ]
    
    for label, rate, x in circles_data:
        circle = plt.Circle((x, 0.5), 0.15, color=plt.cm.RdYlGn(rate/100), alpha=0.7)
        ax1.add_patch(circle)
        ax1.text(x, 0.5, f'{rate:.1f}%', ha='center', va='center', fontsize=16, fontweight='bold')
        ax1.text(x, 0.2, label, ha='center', va='center', fontsize=12)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.text(0.5, 0.9, 'Pairwise Agreement Rates', ha='center', fontsize=16, fontweight='bold')
    
    # GPT-4 consistency
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    
    consistency_score = 93.9
    unanimous_rate = 88.0
    
    # Create consistency meter
    wedges, texts = ax2.pie([consistency_score, 100-consistency_score], 
                            colors=['#2ca02c', '#e0e0e0'],
                            startangle=90, counterclock=False)
    ax2.text(0, 0, f'{consistency_score:.1f}%', ha='center', va='center', 
             fontsize=20, fontweight='bold')
    ax2.text(0, -1.5, 'GPT-4 Avg\nConsistency', ha='center', fontsize=12)
    
    # All agree statistic
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    all_agree_rate = 52.6
    ax3.text(0.5, 0.7, f'{all_agree_rate:.1f}%', ha='center', fontsize=36, fontweight='bold', color='#1f77b4')
    ax3.text(0.5, 0.3, 'All Three\nEvaluators\nAgree', ha='center', fontsize=14)
    
    # Tie rates comparison
    ax4 = fig.add_subplot(gs[1, 2])
    sources = ['GPT-4', 'Survey', 'Baseline']
    tie_rates = [4.0, 42.1, 4.5]
    bars = ax4.barh(sources, tie_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax4.set_xlabel('Tie Vote %')
    ax4.set_title('Tie Vote Rates', fontsize=14)
    for i, (source, rate) in enumerate(zip(sources, tie_rates)):
        ax4.text(rate + 1, i, f'{rate:.1f}%', va='center')
    
    # Key insights
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    insights = [
        "• GPT-4 shows 88% unanimous decisions across 5 samples",
        "• GPT-4 agrees more with Baseline (73.7%) than Survey (52.6%)",
        "• Survey participants choose 'Tie' 10x more often than GPT-4 or Baseline",
        "• All three evaluators agree on only 52.6% of interactions"
    ]
    
    ax5.text(0.5, 0.8, 'Key Insights', ha='center', fontsize=16, fontweight='bold')
    for i, insight in enumerate(insights):
        ax5.text(0.1, 0.6 - i*0.15, insight, fontsize=12, va='top')
    
    plt.tight_layout()
    plt.savefig('evaluation_summary_infographic.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Create all plots"""
    print("Creating plots...")
    
    create_vote_distribution_plot()
    print("✓ Vote distribution plot created")
    
    create_agreement_matrix()
    print("✓ Agreement matrix created")
    
    create_consistency_plot()
    print("✓ Consistency analysis plot created")
    
    create_interaction_comparison_plot()
    print("✓ Interaction comparison plot created")
    
    create_tie_analysis_plot()
    print("✓ Tie analysis plot created")
    
    create_summary_infographic()
    print("✓ Summary infographic created")
    
    print("\nAll plots saved successfully!")

if __name__ == "__main__":
    main()