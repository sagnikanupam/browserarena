#!/usr/bin/env python3
"""
Create a proper heatmap visualization for V2 inter-annotator agreement using complete data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_complete_heatmap():
    # Load the complete results
    with open('/Users/davisbrown/browserarena/agreement-results/v2_inter_annotator_results_complete.json', 'r') as f:
        results = json.load(f)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 14))
    
    # Create main heatmap
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
    
    # Get agreement matrix data
    matrix = np.array(results['agreement_matrix']['matrix'])
    annotators = results['agreement_matrix']['annotators']
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    
    # Create the heatmap
    sns.heatmap(matrix, 
                mask=mask,
                cmap='RdYlGn',
                vmin=0, vmax=1,
                xticklabels=[ann[:12] + '...' for ann in annotators],
                yticklabels=[ann[:12] + '...' for ann in annotators],
                cbar_kws={'label': 'Agreement Rate'},
                annot=True,
                fmt='.2f',
                annot_kws={'size': 8},
                square=True,
                linewidths=0.5,
                cbar=True,
                ax=ax1)
    
    ax1.set_title('V2 Inter-Annotator Agreement Matrix (Top 20 Most Active Annotators)', 
                  fontsize=16, pad=20)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    
    # Rotate labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0)
    
    # Create agreement distribution histogram
    ax2 = plt.subplot2grid((3, 2), (2, 0))
    
    all_agreements = list(results['pair_agreements'].values())
    ax2.hist(all_agreements, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(results['overall_stats']['overall_agreement'], 
                color='red', linestyle='--', linewidth=2,
                label=f'Mean: {results["overall_stats"]["overall_agreement"]:.3f}')
    ax2.set_xlabel('Pairwise Agreement Rate')
    ax2.set_ylabel('Number of Pairs')
    ax2.set_title('Distribution of All Pairwise Agreements')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add text stats
    ax2.text(0.02, 0.95, f'N = {len(all_agreements):,} pairs', 
             transform=ax2.transAxes, fontsize=10, va='top')
    
    # Create per-question agreement bar plot
    ax3 = plt.subplot2grid((3, 2), (2, 1))
    
    questions = list(results['question_agreements'].keys())
    agreements = list(results['question_agreements'].values())
    
    # Sort by agreement
    sorted_indices = np.argsort(agreements)
    questions_sorted = [questions[i] for i in sorted_indices]
    agreements_sorted = [agreements[i] for i in sorted_indices]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(questions_sorted))
    bars = ax3.barh(y_pos, agreements_sorted)
    
    # Color bars based on agreement level
    for bar, agreement in zip(bars, agreements_sorted):
        if agreement >= 0.8:
            bar.set_color('green')
        elif agreement >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(questions_sorted, fontsize=8)
    ax3.set_xlabel('Agreement Rate')
    ax3.set_title('Agreement Rates by Question')
    ax3.set_xlim(0, 1)
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (q, val) in enumerate(zip(questions_sorted, agreements_sorted)):
        ax3.text(val + 0.01, i, f'{val:.2f}', va='center', fontsize=8)
    
    # Add overall title
    fig.suptitle('V2 Human Inter-Annotator Agreement Analysis\n' + 
                 f'114 Annotators, {results["overall_stats"]["num_pairs"]:,} Pairs, ' +
                 f'Overall Agreement: {results["overall_stats"]["overall_agreement"]:.1%}',
                 fontsize=18)
    
    plt.tight_layout()
    plt.savefig('/Users/davisbrown/browserarena/agreement-results/v2_agreement_heatmap_complete.png',
                dpi=300, bbox_inches='tight')
    print("Saved complete heatmap to v2_agreement_heatmap_complete.png")
    
    # Create a second figure showing correlations between annotators
    plt.figure(figsize=(12, 10))
    
    # Create a correlation matrix showing how similar annotators are in their overall patterns
    # For each pair of annotators, compute correlation of their votes across questions
    n_top = len(annotators)
    correlation_matrix = np.zeros((n_top, n_top))
    
    # Load original data to compute correlations
    df = pd.read_csv('/Users/davisbrown/browserarena/agreement-results/BrowserArenaAgreementv2_July_18_2025_11.01.csv')
    data_df = df.iloc[2:]  # Skip header rows
    prolific_id_col = 'PROLIFIC_PID' if 'PROLIFIC_PID' in df.columns else df.columns[-1]
    
    vote_columns = [col for col in df.columns if col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 
                                                          'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 
                                                          'Q13', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q22']]
    
    # Get votes for top annotators
    annotator_votes = {}
    for ann in annotators:
        ann_data = data_df[data_df[prolific_id_col] == ann]
        if len(ann_data) > 0:
            votes = []
            for q in vote_columns:
                vote = ann_data.iloc[0][q]
                vote_str = str(vote).strip()
                if vote_str in ['1', '2', '3', '1.0', '2.0', '3.0']:
                    votes.append(int(float(vote_str)))
                else:
                    votes.append(np.nan)
            annotator_votes[ann] = votes
    
    # Compute correlations
    for i, ann1 in enumerate(annotators):
        for j, ann2 in enumerate(annotators):
            if ann1 in annotator_votes and ann2 in annotator_votes:
                votes1 = np.array(annotator_votes[ann1])
                votes2 = np.array(annotator_votes[ann2])
                
                # Get indices where both have valid votes
                valid_idx = ~(np.isnan(votes1) | np.isnan(votes2))
                
                if np.sum(valid_idx) >= 5:  # Need at least 5 common votes
                    corr = np.corrcoef(votes1[valid_idx], votes2[valid_idx])[0, 1]
                    correlation_matrix[i, j] = corr
                else:
                    correlation_matrix[i, j] = np.nan
    
    # Create correlation heatmap
    mask_corr = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    
    sns.heatmap(correlation_matrix,
                mask=mask_corr,
                cmap='coolwarm',
                center=0,
                vmin=-1, vmax=1,
                xticklabels=[ann[:12] + '...' for ann in annotators],
                yticklabels=[ann[:12] + '...' for ann in annotators],
                cbar_kws={'label': 'Vote Pattern Correlation'},
                square=True,
                linewidths=0.5)
    
    plt.title('V2 Annotator Vote Pattern Correlations\n(How similarly annotators vote across questions)', 
              fontsize=14)
    plt.tight_layout()
    plt.savefig('/Users/davisbrown/browserarena/agreement-results/v2_vote_correlation_heatmap.png',
                dpi=300, bbox_inches='tight')
    print("Saved vote correlation heatmap to v2_vote_correlation_heatmap.png")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Overall agreement: {results['overall_stats']['overall_agreement']:.3f}")
    print(f"Standard deviation: {results['overall_stats']['std_deviation']:.3f}")
    print(f"Min agreement: {results['overall_stats']['min_agreement']:.3f}")
    print(f"Max agreement: {results['overall_stats']['max_agreement']:.3f}")
    print(f"Fleiss' Kappa: {results['overall_stats']['fleiss_kappa']:.3f}")
    
    # Agreement distribution stats
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(all_agreements, bins=bins)
    print("\nAgreement Distribution:")
    for i in range(len(bins)-1):
        pct = hist[i] / len(all_agreements) * 100
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:,} pairs ({pct:.1f}%)")

if __name__ == "__main__":
    create_complete_heatmap()