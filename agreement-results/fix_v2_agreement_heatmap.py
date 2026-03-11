#!/usr/bin/env python3
"""
Fix the V2 agreement heatmap to properly visualize available pairwise agreement data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_fixed_heatmap():
    # Load the results
    with open('/Users/davisbrown/browserarena/agreement-results/v2_inter_annotator_results.json', 'r') as f:
        results = json.load(f)
    
    # Since we only have 50 pairs in the data, let's visualize them differently
    # Option 1: Show per-question agreement as a heatmap
    questions = list(results['question_agreements'].keys())
    agreements = list(results['question_agreements'].values())
    
    # Sort questions by name for better visualization
    sorted_indices = sorted(range(len(questions)), key=lambda i: questions[i])
    questions_sorted = [questions[i] for i in sorted_indices]
    agreements_sorted = [agreements[i] for i in sorted_indices]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Question agreement heatmap
    # Create a matrix where each cell shows agreement for that question
    n_questions = len(questions_sorted)
    matrix = np.array(agreements_sorted).reshape(1, -1)
    
    # Create custom colormap
    cmap = plt.cm.RdYlGn
    
    im = ax1.imshow(matrix, cmap=cmap, aspect='auto', vmin=0.4, vmax=0.9)
    
    # Set ticks
    ax1.set_xticks(range(n_questions))
    ax1.set_xticklabels(questions_sorted, rotation=45, ha='right')
    ax1.set_yticks([0])
    ax1.set_yticklabels(['Agreement Rate'])
    
    # Add text annotations
    for i in range(n_questions):
        text = ax1.text(i, 0, f'{agreements_sorted[i]:.2f}', 
                       ha='center', va='center', color='white' if agreements_sorted[i] < 0.6 else 'black',
                       fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Agreement Rate', rotation=270, labelpad=20)
    
    ax1.set_title('V2 Inter-Annotator Agreement by Question')
    
    # Plot 2: Sample pairwise agreements visualization
    # Extract available pairs and create a sorted list
    pair_data = []
    for pair_key, agreement in results['pair_agreements'].items():
        ann1, ann2 = pair_key.split('_')
        pair_data.append((ann1[:6], ann2[:6], agreement))
    
    # Sort by agreement
    pair_data.sort(key=lambda x: x[2], reverse=True)
    
    # Show top 25 pairs
    top_pairs = pair_data[:25]
    
    y_pos = np.arange(len(top_pairs))
    agreements_list = [p[2] for p in top_pairs]
    
    bars = ax2.barh(y_pos, agreements_list)
    
    # Color bars based on agreement level
    for bar, agreement in zip(bars, agreements_list):
        if agreement >= 0.8:
            bar.set_color('green')
        elif agreement >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Add value labels
    for i, (ann1, ann2, agreement) in enumerate(top_pairs):
        ax2.text(agreement + 0.01, i, f'{agreement:.2f}', va='center')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'{p[0]}...{p[1]}' for p in top_pairs])
    ax2.set_xlabel('Agreement Rate')
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_title('Sample Pairwise Agreements (Top 25)')
    
    plt.suptitle('V2 Inter-Annotator Agreement Analysis', fontsize=16)
    plt.tight_layout()
    
    # Save the fixed heatmap
    plt.savefig('/Users/davisbrown/browserarena/agreement-results/v2_agreement_heatmap_fixed.png', 
                dpi=300, bbox_inches='tight')
    print("Saved fixed heatmap to v2_agreement_heatmap_fixed.png")
    
    # Create an alternative visualization showing agreement distribution
    plt.figure(figsize=(10, 8))
    
    # Create a 2D histogram of agreement rates
    pair_agreements = list(results['pair_agreements'].values())
    
    # Create bins for agreement levels
    bins = np.arange(0, 1.1, 0.1)
    hist, bin_edges = np.histogram(pair_agreements, bins=bins)
    
    # Create heatmap-style visualization
    matrix = hist.reshape(1, -1)
    
    im = plt.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Set labels
    plt.xticks(range(len(bins)-1), [f'{b:.1f}-{bins[i+1]:.1f}' for i, b in enumerate(bins[:-1])], rotation=45)
    plt.yticks([0], ['Count'])
    
    # Add text annotations
    for i, count in enumerate(hist):
        plt.text(i, 0, str(count), ha='center', va='center', 
                color='white' if count > max(hist)/2 else 'black', fontweight='bold')
    
    plt.colorbar(im, label='Number of Annotator Pairs')
    plt.title('Distribution of V2 Pairwise Agreement Rates')
    plt.xlabel('Agreement Rate Range')
    
    plt.tight_layout()
    plt.savefig('/Users/davisbrown/browserarena/agreement-results/v2_agreement_distribution_heatmap.png', 
                dpi=300, bbox_inches='tight')
    print("Saved agreement distribution heatmap to v2_agreement_distribution_heatmap.png")

if __name__ == "__main__":
    create_fixed_heatmap()