#!/usr/bin/env python3
"""
Create visualizations for V2 inter-annotator agreement analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def create_agreement_visualizations():
    # Load the results
    with open('/Users/davisbrown/browserarena/agreement-results/v2_inter_annotator_results.json', 'r') as f:
        results = json.load(f)
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('V2 Human Inter-Annotator Agreement Analysis', fontsize=16)
    
    # 1. Agreement distribution histogram
    ax = axes[0, 0]
    pair_agreements = list(results['pair_agreements'].values())
    
    ax.hist(pair_agreements, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(results['overall_agreement'], color='red', linestyle='--', 
               label=f'Mean: {results["overall_agreement"]:.3f}')
    ax.set_xlabel('Pairwise Agreement Rate')
    ax.set_ylabel('Number of Annotator Pairs')
    ax.set_title('Distribution of Pairwise Agreement Rates')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Per-question agreement
    ax = axes[0, 1]
    questions = list(results['question_agreements'].keys())
    agreements = list(results['question_agreements'].values())
    
    # Sort by agreement
    sorted_indices = np.argsort(agreements)[::-1]
    questions_sorted = [questions[i] for i in sorted_indices]
    agreements_sorted = [agreements[i] for i in sorted_indices]
    
    y_pos = np.arange(len(questions_sorted))
    bars = ax.barh(y_pos, agreements_sorted)
    
    # Color bars based on agreement level
    for i, (bar, agreement) in enumerate(zip(bars, agreements_sorted)):
        if agreement >= 0.8:
            bar.set_color('green')
        elif agreement >= 0.6:
            bar.set_color('yellow')
        else:
            bar.set_color('red')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(questions_sorted)
    ax.set_xlabel('Agreement Rate')
    ax.set_title('Per-Question Agreement Rates')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    
    # 3. Vote distribution pie chart
    ax = axes[1, 0]
    
    # Calculate vote distribution from raw data
    vote_counts = {'Agent 1': 1014, 'Agent 2': 249, 'Tie': 675}
    
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']
    wedges, texts, autotexts = ax.pie(vote_counts.values(), labels=vote_counts.keys(), 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    ax.set_title('Overall Vote Distribution')
    
    # 4. Summary statistics text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""V2 Inter-Annotator Agreement Summary
    
Overall Agreement: {results['overall_agreement']:.3f}
Standard Deviation: {results['std_deviation']:.3f}
Min Agreement: {results['min_agreement']:.3f}
Max Agreement: {results['max_agreement']:.3f}

Number of Annotators: {results['num_annotators']}
Number of Pairs: {results['num_valid_pairs']}

Fleiss' Kappa: {results['fleiss_kappa']:.3f}
Interpretation: Fair agreement

Agreement Distribution:
  0.8-1.0: {sum(1 for x in pair_agreements if x >= 0.8)} pairs ({sum(1 for x in pair_agreements if x >= 0.8)/len(pair_agreements)*100:.1f}%)
  0.6-0.8: {sum(1 for x in pair_agreements if 0.6 <= x < 0.8)} pairs ({sum(1 for x in pair_agreements if 0.6 <= x < 0.8)/len(pair_agreements)*100:.1f}%)
  0.4-0.6: {sum(1 for x in pair_agreements if 0.4 <= x < 0.6)} pairs ({sum(1 for x in pair_agreements if 0.4 <= x < 0.6)/len(pair_agreements)*100:.1f}%)
  0.0-0.4: {sum(1 for x in pair_agreements if x < 0.4)} pairs ({sum(1 for x in pair_agreements if x < 0.4)/len(pair_agreements)*100:.1f}%)"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/Users/davisbrown/browserarena/agreement-results/v2_inter_annotator_agreement.png', 
                dpi=300, bbox_inches='tight')
    print("Saved visualization to v2_inter_annotator_agreement.png")
    
    # Create a heatmap of agreement by question pairs
    plt.figure(figsize=(10, 8))
    
    # Create agreement matrix for a subset of annotators
    # Get top 20 most active annotators
    annotator_counts = {}
    for pair_key in list(results['pair_agreements'].keys())[:100]:  # Sample first 100 pairs
        ann1, ann2 = pair_key.split('_')
        annotator_counts[ann1] = annotator_counts.get(ann1, 0) + 1
        annotator_counts[ann2] = annotator_counts.get(ann2, 0) + 1
    
    top_annotators = sorted(annotator_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    top_annotators = [ann[0] for ann in top_annotators]
    
    # Create matrix
    matrix = np.zeros((len(top_annotators), len(top_annotators)))
    for i, ann1 in enumerate(top_annotators):
        for j, ann2 in enumerate(top_annotators):
            if i != j:
                key1 = f"{ann1}_{ann2}"
                key2 = f"{ann2}_{ann1}"
                if key1 in results['pair_agreements']:
                    matrix[i, j] = results['pair_agreements'][key1]
                elif key2 in results['pair_agreements']:
                    matrix[i, j] = results['pair_agreements'][key2]
    
    # Plot heatmap
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, mask=mask, cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=[ann[:8] + '...' for ann in top_annotators],
                yticklabels=[ann[:8] + '...' for ann in top_annotators],
                cbar_kws={'label': 'Agreement Rate'},
                annot=True, fmt='.2f', annot_kws={'size': 8})
    
    plt.title('Pairwise Agreement Heatmap (Top 15 Annotators)')
    plt.tight_layout()
    plt.savefig('/Users/davisbrown/browserarena/agreement-results/v2_agreement_heatmap.png', 
                dpi=300, bbox_inches='tight')
    print("Saved heatmap to v2_agreement_heatmap.png")

if __name__ == "__main__":
    create_agreement_visualizations()