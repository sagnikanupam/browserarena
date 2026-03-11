#!/usr/bin/env python3
"""
Create comprehensive figures analyzing correlation between model confidence and agreement.
Generates 7 different visualizations showing GPT-4o and o4-mini confidence patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
def load_all_data():
    """Load all necessary data files"""
    # GPT-4o evaluations - handle potential quoting issues
    gpt4o_df = pd.read_csv('gpt4o_proper_evaluation_final.csv', 
                          quoting=1, on_bad_lines='skip')
    
    # o4-mini evaluations  
    o4mini_df = pd.read_csv('o4mini_evaluation_final.csv',
                           quoting=1, on_bad_lines='skip')
    
    # V2 human evaluations
    v2_df = pd.read_csv('BrowserArenaAgreementv2_July_18_2025_11.01.csv',
                       encoding='utf-8', on_bad_lines='skip')
    
    # Extract baseline winners
    baseline_map = {}
    for _, row in gpt4o_df.iterrows():
        baseline_map[row['question']] = row['baseline_winner']
    
    return gpt4o_df, o4mini_df, v2_df, baseline_map

def get_v2_majority(v2_df, question):
    """Get V2 human majority vote for a question"""
    if question not in v2_df.columns:
        return None
    
    votes = v2_df[question].dropna()
    if len(votes) == 0:
        return None
        
    # Count votes
    vote_counts = votes.value_counts()
    if len(vote_counts) == 0:
        return None
        
    # Return majority
    return vote_counts.index[0]

def calculate_human_consensus_strength(v2_df, question):
    """Calculate how strongly humans agree (% voting for majority)"""
    if question not in v2_df.columns:
        return None
    
    votes = v2_df[question].dropna()
    if len(votes) == 0:
        return None
        
    vote_counts = votes.value_counts()
    if len(vote_counts) == 0:
        return None
        
    # Return percentage voting for majority
    return vote_counts.iloc[0] / len(votes)

# Figure 1: Confidence vs Agreement Accuracy Scatter Plot
def create_confidence_agreement_scatter(gpt4o_df, o4mini_df, v2_df, baseline_map):
    """Create scatter plot of confidence vs agreement with baseline/V2 majority"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Process GPT-4o
    gpt4o_data = []
    for _, row in gpt4o_df.iterrows():
        q = row['question']
        if q not in baseline_map:
            continue
            
        confidence = row['gpt4o_confidence']
        agrees_baseline = row['agrees_with_baseline']
        v2_majority = get_v2_majority(v2_df, q)
        agrees_v2 = 1 if v2_majority and row['gpt4o_preference'] == v2_majority else 0
        
        # Use baseline agreement as primary
        gpt4o_data.append({
            'confidence': confidence,
            'agrees': int(agrees_baseline),
            'vote_type': row['gpt4o_preference'],
            'question': q
        })
    
    # Process o4-mini
    o4mini_data = []
    for _, row in o4mini_df.iterrows():
        q = row['question']
        if q not in baseline_map:
            continue
            
        confidence = row['o4mini_confidence']
        agrees_baseline = row['agrees_with_baseline']
        v2_majority = get_v2_majority(v2_df, q)
        agrees_v2 = 1 if v2_majority and row['o4mini_preference'] == v2_majority else 0
        
        o4mini_data.append({
            'confidence': confidence,
            'agrees': int(agrees_baseline),
            'vote_type': row['o4mini_preference'],
            'question': q
        })
    
    # Plot GPT-4o
    gpt4o_plot_df = pd.DataFrame(gpt4o_data)
    colors = {'Agent 1': 'blue', 'Agent 2': 'red', 'Tie': 'gray'}
    
    for vote_type in gpt4o_plot_df['vote_type'].unique():
        mask = gpt4o_plot_df['vote_type'] == vote_type
        data = gpt4o_plot_df[mask]
        
        # Add jitter
        x = data['confidence'] + np.random.normal(0, 0.01, len(data))
        y = data['agrees'] + np.random.normal(0, 0.02, len(data))
        
        axes[0].scatter(x, y, alpha=0.6, s=100, 
                       color=colors.get(vote_type, 'black'),
                       label=vote_type)
    
    # Add regression line for GPT-4o
    if len(gpt4o_plot_df) > 0:
        z = np.polyfit(gpt4o_plot_df['confidence'], gpt4o_plot_df['agrees'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(gpt4o_plot_df['confidence'].min(), 
                            gpt4o_plot_df['confidence'].max(), 100)
        axes[0].plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=2)
        
        # Calculate correlation
        corr, p_val = stats.pearsonr(gpt4o_plot_df['confidence'], 
                                     gpt4o_plot_df['agrees'])
        axes[0].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}',
                    transform=axes[0].transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[0].set_xlabel('GPT-4o Confidence', fontsize=12)
    axes[0].set_ylabel('Agrees with Baseline (0/1)', fontsize=12)
    axes[0].set_title('GPT-4o: Confidence vs Agreement', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(True, alpha=0.3)
    
    # Plot o4-mini
    o4mini_plot_df = pd.DataFrame(o4mini_data)
    
    for vote_type in o4mini_plot_df['vote_type'].unique():
        mask = o4mini_plot_df['vote_type'] == vote_type
        data = o4mini_plot_df[mask]
        
        # Add jitter
        x = data['confidence'] + np.random.normal(0, 0.01, len(data))
        y = data['agrees'] + np.random.normal(0, 0.02, len(data))
        
        axes[1].scatter(x, y, alpha=0.6, s=100,
                       color=colors.get(vote_type, 'black'),
                       label=vote_type)
    
    # Add regression line for o4-mini
    if len(o4mini_plot_df) > 0:
        z = np.polyfit(o4mini_plot_df['confidence'], o4mini_plot_df['agrees'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(o4mini_plot_df['confidence'].min(),
                            o4mini_plot_df['confidence'].max(), 100)
        axes[1].plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=2)
        
        # Calculate correlation
        corr, p_val = stats.pearsonr(o4mini_plot_df['confidence'],
                                     o4mini_plot_df['agrees'])
        axes[1].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}',
                    transform=axes[1].transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[1].set_xlabel('o4-mini Confidence', fontsize=12)
    axes[1].set_ylabel('Agrees with Baseline (0/1)', fontsize=12)
    axes[1].set_title('o4-mini: Confidence vs Agreement', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Model Confidence vs Agreement with Baseline', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confidence_vs_agreement_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig('confidence_vs_agreement_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    return gpt4o_plot_df, o4mini_plot_df

# Figure 2: Confidence Calibration Curve
def create_calibration_curve(gpt4o_df, o4mini_df, baseline_map):
    """Create calibration curve showing expected vs actual accuracy"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Process data for both models
    models_data = {}
    
    for model_name, df, conf_col, agrees_col in [
        ('GPT-4o', gpt4o_df, 'gpt4o_confidence', 'agrees_with_baseline'),
        ('o4-mini', o4mini_df, 'o4mini_confidence', 'agrees_with_baseline')
    ]:
        confidences = df[conf_col].values
        agreements = df[agrees_col].values.astype(int)
        
        # Create bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        # Calculate actual accuracy in each bin
        actual_acc = []
        expected_acc = []
        counts = []
        
        for i in range(n_bins):
            mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
            if i == n_bins - 1:  # Include right boundary for last bin
                mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            
            if mask.sum() > 0:
                actual_acc.append(agreements[mask].mean())
                expected_acc.append(confidences[mask].mean())
                counts.append(mask.sum())
            else:
                actual_acc.append(np.nan)
                expected_acc.append(np.nan)
                counts.append(0)
        
        models_data[model_name] = {
            'actual': actual_acc,
            'expected': expected_acc,
            'counts': counts,
            'bin_centers': bin_centers,
            'confidences': confidences
        }
    
    # Plot calibration curves
    colors = {'GPT-4o': 'blue', 'o4-mini': 'orange'}
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Perfect Calibration')
    
    for model_name, data in models_data.items():
        # Remove NaN values
        valid_mask = ~np.isnan(data['actual'])
        actual = np.array(data['actual'])[valid_mask]
        expected = np.array(data['expected'])[valid_mask]
        
        ax1.plot(expected, actual, 'o-', 
                color=colors[model_name], 
                linewidth=2, markersize=8,
                label=f'{model_name}', alpha=0.8)
        
        # Calculate ECE (Expected Calibration Error)
        counts = np.array(data['counts'])[valid_mask]
        if len(counts) > 0:
            ece = np.sum(counts * np.abs(actual - expected)) / np.sum(counts)
            ax1.text(0.05, 0.85 - 0.05 * list(models_data.keys()).index(model_name),
                    f'{model_name} ECE: {ece:.3f}',
                    transform=ax1.transAxes, fontsize=11,
                    color=colors[model_name])
    
    ax1.set_xlabel('Expected Accuracy (Mean Confidence)', fontsize=12)
    ax1.set_ylabel('Actual Accuracy (Agreement Rate)', fontsize=12)
    ax1.set_title('Confidence Calibration: Expected vs Actual Accuracy', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Plot histogram of confidence distribution
    for model_name, data in models_data.items():
        ax2.hist(data['confidences'], bins=20, alpha=0.5, 
                color=colors[model_name], label=model_name,
                edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Confidence Scores', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Model Confidence Calibration Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confidence_calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig('confidence_calibration_curve.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 3: Disagreement Pattern Heatmap
def create_disagreement_heatmap(gpt4o_df, o4mini_df, v2_df, baseline_map):
    """Create heatmap showing confidence and agreement patterns by question"""
    
    # Prepare data
    questions = sorted(set(gpt4o_df['question']) & set(o4mini_df['question']))
    
    # Create matrix
    data_matrix = []
    question_labels = []
    
    for q in questions:
        gpt4o_row = gpt4o_df[gpt4o_df['question'] == q].iloc[0]
        o4mini_row = o4mini_df[o4mini_df['question'] == q].iloc[0]
        
        # Calculate human consensus strength
        consensus = calculate_human_consensus_strength(v2_df, q)
        if consensus is None:
            consensus = 0.5
        
        row = [
            gpt4o_row['gpt4o_confidence'],
            o4mini_row['o4mini_confidence'],
            int(gpt4o_row['agrees_with_baseline']),
            int(o4mini_row['agrees_with_baseline']),
            consensus
        ]
        data_matrix.append(row)
        question_labels.append(q)
    
    # Convert to DataFrame
    df_matrix = pd.DataFrame(data_matrix,
                            columns=['GPT-4o\nConf', 'o4-mini\nConf', 
                                   'GPT-4o\nAgrees', 'o4-mini\nAgrees',
                                   'Human\nConsensus'],
                            index=question_labels)
    
    # Sort by human consensus strength
    df_matrix = df_matrix.sort_values('Human\nConsensus')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 12))
    
    # Create custom colormap for different columns
    # Normalize confidences and consensus to 0-1, agreements are already 0/1
    plot_data = df_matrix.copy()
    
    # Plot heatmap
    sns.heatmap(plot_data, annot=True, fmt='.2f', cmap='RdYlGn',
               cbar_kws={'label': 'Value'}, ax=ax,
               vmin=0, vmax=1, linewidths=0.5, linecolor='gray')
    
    ax.set_title('Model Confidence and Agreement Patterns by Question\n(Sorted by Human Consensus Difficulty)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Questions', fontsize=12)
    
    # Rotate y-axis labels for better readability
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('disagreement_pattern_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('disagreement_pattern_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 4: Confidence Distribution by Agreement Outcome
def create_confidence_by_agreement(gpt4o_df, o4mini_df, baseline_map):
    """Create violin plots of confidence distribution by agreement patterns"""
    
    # Categorize each question by agreement pattern
    categories = []
    
    for q in set(gpt4o_df['question']) & set(o4mini_df['question']):
        gpt4o_row = gpt4o_df[gpt4o_df['question'] == q].iloc[0]
        o4mini_row = o4mini_df[o4mini_df['question'] == q].iloc[0]
        
        gpt4o_agrees = gpt4o_row['agrees_with_baseline']
        o4mini_agrees = o4mini_row['agrees_with_baseline']
        
        if gpt4o_agrees and o4mini_agrees:
            category = 'Both Agree'
        elif gpt4o_agrees and not o4mini_agrees:
            category = 'Only GPT-4o'
        elif not gpt4o_agrees and o4mini_agrees:
            category = 'Only o4-mini'
        else:
            category = 'Neither Agrees'
        
        categories.append({
            'question': q,
            'category': category,
            'gpt4o_conf': gpt4o_row['gpt4o_confidence'],
            'o4mini_conf': o4mini_row['o4mini_confidence']
        })
    
    df_cat = pd.DataFrame(categories)
    
    # Reshape for plotting
    plot_data = []
    for _, row in df_cat.iterrows():
        plot_data.append({
            'Category': row['category'],
            'Model': 'GPT-4o',
            'Confidence': row['gpt4o_conf']
        })
        plot_data.append({
            'Category': row['category'],
            'Model': 'o4-mini',
            'Confidence': row['o4mini_conf']
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define category order
    category_order = ['Both Agree', 'Only GPT-4o', 'Only o4-mini', 'Neither Agrees']
    
    # Create violin plot
    sns.violinplot(data=plot_df, x='Category', y='Confidence', hue='Model',
                  split=False, inner='box', ax=ax, order=category_order,
                  palette={'GPT-4o': 'blue', 'o4-mini': 'orange'})
    
    # Add strip plot for individual points
    sns.stripplot(data=plot_df, x='Category', y='Confidence', hue='Model',
                 dodge=True, alpha=0.3, ax=ax, order=category_order,
                 palette={'GPT-4o': 'blue', 'o4-mini': 'orange'},
                 legend=False)
    
    ax.set_title('Model Confidence Distribution by Agreement with Baseline',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Agreement Pattern', fontsize=12)
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count annotations
    for i, cat in enumerate(category_order):
        count = len(df_cat[df_cat['category'] == cat])
        ax.text(i, -0.15, f'n={count}', ha='center', transform=ax.get_xaxis_transform())
    
    # Statistical tests
    for cat in category_order:
        cat_data = df_cat[df_cat['category'] == cat]
        if len(cat_data) > 1:
            t_stat, p_val = stats.ttest_rel(cat_data['gpt4o_conf'], 
                                           cat_data['o4mini_conf'])
            print(f"{cat}: t={t_stat:.3f}, p={p_val:.3f}")
    
    plt.tight_layout()
    plt.savefig('confidence_by_agreement_outcome.png', dpi=300, bbox_inches='tight')
    plt.savefig('confidence_by_agreement_outcome.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 5: ROC-style Analysis
def create_roc_analysis(gpt4o_df, o4mini_df, baseline_map):
    """Create ROC curves using confidence as threshold for predicting agreement"""
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Process each model
    for model_name, df, conf_col, agrees_col, color in [
        ('GPT-4o', gpt4o_df, 'gpt4o_confidence', 'agrees_with_baseline', 'blue'),
        ('o4-mini', o4mini_df, 'o4mini_confidence', 'agrees_with_baseline', 'orange')
    ]:
        confidences = df[conf_col].values
        agreements = df[agrees_col].values.astype(int)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(agreements, confidences)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        ax.plot(fpr, tpr, color=color, linewidth=2,
               label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1,
           label='Random (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Analysis: Using Confidence to Predict Agreement with Baseline',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('roc_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('roc_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 6: Cross-Model Confidence Correlation
def create_cross_model_correlation(gpt4o_df, o4mini_df, baseline_map):
    """Create scatter plot of GPT-4o vs o4-mini confidence"""
    
    # Merge data on questions
    merged_data = []
    
    for q in set(gpt4o_df['question']) & set(o4mini_df['question']):
        gpt4o_row = gpt4o_df[gpt4o_df['question'] == q].iloc[0]
        o4mini_row = o4mini_df[o4mini_df['question'] == q].iloc[0]
        
        # Determine agreement pattern
        if gpt4o_row['agrees_with_baseline'] and o4mini_row['agrees_with_baseline']:
            pattern = 'Both Agree'
        elif gpt4o_row['agrees_with_baseline']:
            pattern = 'Only GPT-4o'
        elif o4mini_row['agrees_with_baseline']:
            pattern = 'Only o4-mini'
        else:
            pattern = 'Neither'
        
        merged_data.append({
            'question': q,
            'gpt4o_conf': gpt4o_row['gpt4o_confidence'],
            'o4mini_conf': o4mini_row['o4mini_confidence'],
            'agreement_pattern': pattern
        })
    
    df_merged = pd.DataFrame(merged_data)
    
    # Create figure with marginal distributions
    fig = plt.figure(figsize=(10, 10))
    
    # Main scatter plot
    ax_main = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
    
    # Marginal distributions
    ax_top = plt.subplot2grid((4, 4), (0, 0), colspan=3)
    ax_right = plt.subplot2grid((4, 4), (1, 3), rowspan=3)
    
    # Color map for agreement patterns
    colors = {
        'Both Agree': 'green',
        'Only GPT-4o': 'blue', 
        'Only o4-mini': 'orange',
        'Neither': 'red'
    }
    
    # Plot main scatter
    for pattern in colors.keys():
        mask = df_merged['agreement_pattern'] == pattern
        data = df_merged[mask]
        ax_main.scatter(data['gpt4o_conf'], data['o4mini_conf'],
                       color=colors[pattern], label=pattern,
                       alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line
    ax_main.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    # Calculate and display correlation
    corr, p_val = stats.pearsonr(df_merged['gpt4o_conf'], df_merged['o4mini_conf'])
    ax_main.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}',
                transform=ax_main.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_main.set_xlabel('GPT-4o Confidence', fontsize=12)
    ax_main.set_ylabel('o4-mini Confidence', fontsize=12)
    ax_main.legend(loc='lower right')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(0.4, 1.0)
    ax_main.set_ylim(0.4, 1.0)
    
    # Top marginal (GPT-4o)
    ax_top.hist(df_merged['gpt4o_conf'], bins=15, color='blue', alpha=0.5,
               edgecolor='black', linewidth=0.5)
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_ylabel('Count')
    ax_top.set_title('Cross-Model Confidence Correlation', fontsize=14, fontweight='bold')
    
    # Right marginal (o4-mini)
    ax_right.hist(df_merged['o4mini_conf'], bins=15, orientation='horizontal',
                 color='orange', alpha=0.5, edgecolor='black', linewidth=0.5)
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.set_xlabel('Count')
    
    plt.tight_layout()
    plt.savefig('cross_model_confidence_correlation.png', dpi=300, bbox_inches='tight')
    plt.savefig('cross_model_confidence_correlation.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_merged

# Figure 7: Confidence vs Human Consensus Strength
def create_confidence_vs_consensus(gpt4o_df, o4mini_df, v2_df):
    """Plot model confidence against human voting unanimity"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Collect data
    consensus_data = []
    
    for q in set(gpt4o_df['question']) & set(o4mini_df['question']):
        consensus = calculate_human_consensus_strength(v2_df, q)
        if consensus is None:
            continue
            
        gpt4o_row = gpt4o_df[gpt4o_df['question'] == q].iloc[0]
        o4mini_row = o4mini_df[o4mini_df['question'] == q].iloc[0]
        
        consensus_data.append({
            'question': q,
            'consensus': consensus,
            'gpt4o_conf': gpt4o_row['gpt4o_confidence'],
            'o4mini_conf': o4mini_row['o4mini_confidence'],
            'gpt4o_agrees': gpt4o_row['agrees_with_baseline'],
            'o4mini_agrees': o4mini_row['agrees_with_baseline']
        })
    
    df_consensus = pd.DataFrame(consensus_data)
    
    # Plot GPT-4o
    colors = df_consensus['gpt4o_agrees'].map({True: 'green', False: 'red'})
    axes[0].scatter(df_consensus['consensus'], df_consensus['gpt4o_conf'],
                   c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    # Add regression line
    z = np.polyfit(df_consensus['consensus'], df_consensus['gpt4o_conf'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_consensus['consensus'].min(),
                        df_consensus['consensus'].max(), 100)
    axes[0].plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=2)
    
    # Calculate correlation
    corr, p_val = stats.pearsonr(df_consensus['consensus'], df_consensus['gpt4o_conf'])
    axes[0].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}',
                transform=axes[0].transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[0].set_xlabel('Human Consensus Strength\n(% voting for majority)', fontsize=12)
    axes[0].set_ylabel('GPT-4o Confidence', fontsize=12)
    axes[0].set_title('GPT-4o Confidence vs Human Consensus', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.6, label='Agrees with Baseline'),
                      Patch(facecolor='red', alpha=0.6, label='Disagrees with Baseline')]
    axes[0].legend(handles=legend_elements, loc='lower right')
    
    # Plot o4-mini
    colors = df_consensus['o4mini_agrees'].map({True: 'green', False: 'red'})
    axes[1].scatter(df_consensus['consensus'], df_consensus['o4mini_conf'],
                   c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    # Add regression line
    z = np.polyfit(df_consensus['consensus'], df_consensus['o4mini_conf'], 1)
    p = np.poly1d(z)
    axes[1].plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=2)
    
    # Calculate correlation
    corr, p_val = stats.pearsonr(df_consensus['consensus'], df_consensus['o4mini_conf'])
    axes[1].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}',
                transform=axes[1].transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[1].set_xlabel('Human Consensus Strength\n(% voting for majority)', fontsize=12)
    axes[1].set_ylabel('o4-mini Confidence', fontsize=12)
    axes[1].set_title('o4-mini Confidence vs Human Consensus', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(handles=legend_elements, loc='lower right')
    
    plt.suptitle('Model Confidence vs Human Consensus Difficulty', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confidence_vs_human_consensus.png', dpi=300, bbox_inches='tight')
    plt.savefig('confidence_vs_human_consensus.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_consensus

# Main execution
def main():
    print("Loading data...")
    gpt4o_df, o4mini_df, v2_df, baseline_map = load_all_data()
    
    print("Creating Figure 1: Confidence vs Agreement Scatter Plot...")
    gpt4o_plot_df, o4mini_plot_df = create_confidence_agreement_scatter(
        gpt4o_df, o4mini_df, v2_df, baseline_map)
    
    print("Creating Figure 2: Confidence Calibration Curve...")
    create_calibration_curve(gpt4o_df, o4mini_df, baseline_map)
    
    print("Creating Figure 3: Disagreement Pattern Heatmap...")
    create_disagreement_heatmap(gpt4o_df, o4mini_df, v2_df, baseline_map)
    
    print("Creating Figure 4: Confidence by Agreement Outcome...")
    create_confidence_by_agreement(gpt4o_df, o4mini_df, baseline_map)
    
    print("Creating Figure 5: ROC Analysis...")
    create_roc_analysis(gpt4o_df, o4mini_df, baseline_map)
    
    print("Creating Figure 6: Cross-Model Confidence Correlation...")
    df_merged = create_cross_model_correlation(gpt4o_df, o4mini_df, baseline_map)
    
    print("Creating Figure 7: Confidence vs Human Consensus...")
    df_consensus = create_confidence_vs_consensus(gpt4o_df, o4mini_df, v2_df)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"\nGPT-4o:")
    print(f"  Mean confidence: {gpt4o_df['gpt4o_confidence'].mean():.3f}")
    print(f"  Std confidence: {gpt4o_df['gpt4o_confidence'].std():.3f}")
    print(f"  Agreement with baseline: {gpt4o_df['agrees_with_baseline'].mean():.3f}")
    
    print(f"\no4-mini:")
    print(f"  Mean confidence: {o4mini_df['o4mini_confidence'].mean():.3f}")
    print(f"  Std confidence: {o4mini_df['o4mini_confidence'].std():.3f}")
    print(f"  Agreement with baseline: {o4mini_df['agrees_with_baseline'].mean():.3f}")
    
    if len(df_merged) > 0:
        corr, p_val = stats.pearsonr(df_merged['gpt4o_conf'], df_merged['o4mini_conf'])
        print(f"\nCross-model confidence correlation: r={corr:.3f}, p={p_val:.3f}")
    
    print("\nAll figures created successfully!")

if __name__ == "__main__":
    main()