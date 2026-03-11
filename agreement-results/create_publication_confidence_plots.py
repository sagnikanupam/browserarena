#!/usr/bin/env python3
"""
Create publication-ready confidence vs human consensus plots.
Separate figures for GPT-4o and o4-mini with improved aesthetics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready style with even larger fonts
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 1.0

def load_data():
    """Load necessary data files"""
    # GPT-4o evaluations
    gpt4o_df = pd.read_csv('gpt4o_proper_evaluation_final.csv', 
                          quoting=1, on_bad_lines='skip')
    
    # o4-mini evaluations  
    o4mini_df = pd.read_csv('o4mini_evaluation_final.csv',
                           quoting=1, on_bad_lines='skip')
    
    # V2 human evaluations
    v2_df = pd.read_csv('BrowserArenaAgreementv2_July_18_2025_11.01.csv',
                       encoding='utf-8', on_bad_lines='skip')
    
    return gpt4o_df, o4mini_df, v2_df

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
        
    # Return percentage voting for majority as percentage (0-100)
    return (vote_counts.iloc[0] / len(votes)) * 100

def create_gpt4o_plot(gpt4o_df, v2_df):
    """Create publication-ready plot for GPT-4o"""
    
    # Collect data
    plot_data = []
    
    for _, row in gpt4o_df.iterrows():
        q = row['question']
        consensus = calculate_human_consensus_strength(v2_df, q)
        if consensus is None:
            continue
            
        plot_data.append({
            'question': q,
            'consensus': consensus,
            'confidence': row['gpt4o_confidence'] * 100,  # Convert to percentage
            'agrees': row['agrees_with_baseline']
        })
    
    df = pd.DataFrame(plot_data)
    
    # Create figure - larger for bigger fonts
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define colors - using colorblind-friendly palette
    colors = {True: '#2E7D32', False: '#C62828'}  # Green for agrees, Red for disagrees
    
    # Plot points with larger size
    for agrees_val, color in colors.items():
        mask = df['agrees'] == agrees_val
        data = df[mask]
        
        ax.scatter(data['consensus'], data['confidence'],
                  c=color, alpha=0.7, s=200,  # Increased point size
                  edgecolors='white', linewidth=2.0,
                  label='Agrees with baseline' if agrees_val else 'Disagrees with baseline')
    
    # Add regression line
    z = np.polyfit(df['consensus'], df['confidence'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['consensus'].min(), df['consensus'].max(), 100)
    ax.plot(x_line, p(x_line), color='#424242', linestyle='--', 
            linewidth=2, alpha=0.7, zorder=0)
    
    # Calculate correlation and r-squared
    corr, p_val = stats.pearsonr(df['consensus'], df['confidence'])
    r_squared = corr ** 2
    
    # Add r-squared text with even larger font
    text_str = f'r² = {r_squared:.3f}'
    ax.text(0.05, 0.95, text_str,
            transform=ax.transAxes, 
            fontsize=30,  # Even larger font for r-squared
            verticalalignment='top',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='white', 
                     edgecolor='gray',
                     alpha=0.9))
    
    # Labels and title with even larger fonts
    ax.set_xlabel('Human agreement (%)', fontsize=28, fontweight='bold')
    ax.set_ylabel('Model confidence (%)', fontsize=28, fontweight='bold')
    ax.set_title('GPT-4o', fontsize=32, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim(30, 105)
    ax.set_ylim(55, 105)
    
    # Improve ticks
    ax.set_xticks(range(40, 101, 20))
    ax.set_yticks(range(60, 101, 10))
    
    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, 
             shadow=False, framealpha=0.9, edgecolor='gray')
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color('#666666')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save
    plt.savefig('gpt4o_confidence_vs_human_consensus.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('gpt4o_confidence_vs_human_consensus.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return df, corr, p_val

def create_o4mini_plot(o4mini_df, v2_df):
    """Create publication-ready plot for o4-mini"""
    
    # Collect data
    plot_data = []
    
    for _, row in o4mini_df.iterrows():
        q = row['question']
        consensus = calculate_human_consensus_strength(v2_df, q)
        if consensus is None:
            continue
            
        plot_data.append({
            'question': q,
            'consensus': consensus,
            'confidence': row['o4mini_confidence'] * 100,  # Convert to percentage
            'agrees': row['agrees_with_baseline']
        })
    
    df = pd.DataFrame(plot_data)
    
    # Create figure - larger for bigger fonts
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define colors - using colorblind-friendly palette
    colors = {True: '#2E7D32', False: '#C62828'}  # Green for agrees, Red for disagrees
    
    # Plot points with larger size
    for agrees_val, color in colors.items():
        mask = df['agrees'] == agrees_val
        data = df[mask]
        
        ax.scatter(data['consensus'], data['confidence'],
                  c=color, alpha=0.7, s=200,  # Increased point size
                  edgecolors='white', linewidth=2.0,
                  label='Agrees with baseline' if agrees_val else 'Disagrees with baseline')
    
    # Add regression line
    z = np.polyfit(df['consensus'], df['confidence'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['consensus'].min(), df['consensus'].max(), 100)
    ax.plot(x_line, p(x_line), color='#424242', linestyle='--',
            linewidth=2, alpha=0.7, zorder=0)
    
    # Calculate correlation and r-squared
    corr, p_val = stats.pearsonr(df['consensus'], df['confidence'])
    r_squared = corr ** 2
    
    # Add r-squared text with even larger font
    text_str = f'r² = {r_squared:.3f}'
    ax.text(0.05, 0.95, text_str,
            transform=ax.transAxes,
            fontsize=30,  # Even larger font for r-squared
            verticalalignment='top',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5',
                     facecolor='white',
                     edgecolor='gray',
                     alpha=0.9))
    
    # Labels and title with even larger fonts
    ax.set_xlabel('Human agreement (%)', fontsize=28, fontweight='bold')
    ax.set_ylabel('Model confidence (%)', fontsize=28, fontweight='bold')
    ax.set_title('o4-mini', fontsize=32, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim(30, 105)
    ax.set_ylim(35, 105)
    
    # Improve ticks
    ax.set_xticks(range(40, 101, 20))
    ax.set_yticks(range(40, 101, 20))
    
    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=True,
             shadow=False, framealpha=0.9, edgecolor='gray')
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color('#666666')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save
    plt.savefig('o4mini_confidence_vs_human_consensus.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('o4mini_confidence_vs_human_consensus.pdf',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return df, corr, p_val

def create_combined_figure(gpt4o_df, o4mini_df, v2_df):
    """Create a combined figure with both models side by side"""
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Collect data for both models
    for idx, (model_df, conf_col, model_name) in enumerate([
        (gpt4o_df, 'gpt4o_confidence', 'GPT-4o'),
        (o4mini_df, 'o4mini_confidence', 'o4-mini')
    ]):
        plot_data = []
        
        for _, row in model_df.iterrows():
            q = row['question']
            consensus = calculate_human_consensus_strength(v2_df, q)
            if consensus is None:
                continue
                
            plot_data.append({
                'question': q,
                'consensus': consensus,
                'confidence': row[conf_col] * 100,
                'agrees': row['agrees_with_baseline']
            })
        
        df = pd.DataFrame(plot_data)
        ax = axes[idx]
        
        # Define colors
        colors = {True: '#2E7D32', False: '#C62828'}
        
        # Plot points
        for agrees_val, color in colors.items():
            mask = df['agrees'] == agrees_val
            data = df[mask]
            
            ax.scatter(data['consensus'], data['confidence'],
                      c=color, alpha=0.7, s=180,  # Larger points
                      edgecolors='white', linewidth=2.0,
                      label='Agrees with baseline' if agrees_val else 'Disagrees with baseline')
        
        # Add regression line
        z = np.polyfit(df['consensus'], df['confidence'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['consensus'].min(), df['consensus'].max(), 100)
        ax.plot(x_line, p(x_line), color='#424242', linestyle='--',
                linewidth=2.5, alpha=0.7, zorder=0)
        
        # Calculate correlation and r-squared
        corr, p_val = stats.pearsonr(df['consensus'], df['confidence'])
        r_squared = corr ** 2
        
        # Add r-squared text
        text_str = f'r² = {r_squared:.3f}'
        ax.text(0.05, 0.95, text_str,
                transform=ax.transAxes,
                fontsize=26,  # Larger font
                verticalalignment='top',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor='white',
                         edgecolor='gray',
                         alpha=0.9))
        
        # Labels and title with larger fonts
        ax.set_xlabel('Human agreement (%)', fontsize=24, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Model confidence (%)', fontsize=24, fontweight='bold')
        ax.set_title(model_name, fontsize=28, fontweight='bold', pad=15)
        
        # Set axis limits
        ax.set_xlim(30, 105)
        if model_name == 'GPT-4o':
            ax.set_ylim(55, 105)
            ax.set_yticks(range(60, 101, 10))
        else:
            ax.set_ylim(35, 105)
            ax.set_yticks(range(40, 101, 20))
        
        ax.set_xticks(range(40, 101, 20))
        
        # Legend
        if idx == 1:
            ax.legend(loc='lower right', frameon=True, fancybox=True,
                     shadow=False, framealpha=0.9, edgecolor='gray')
        
        # Grid styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Spine styling
        for spine in ax.spines.values():
            spine.set_color('#666666')
            spine.set_linewidth(1.5)
    
    plt.suptitle('Model Confidence vs Human Consensus', 
                fontsize=30, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    plt.savefig('combined_confidence_vs_human_consensus.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('combined_confidence_vs_human_consensus.pdf',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    print("Loading data...")
    gpt4o_df, o4mini_df, v2_df = load_data()
    
    print("\nCreating GPT-4o plot...")
    gpt4o_data, gpt4o_corr, gpt4o_pval = create_gpt4o_plot(gpt4o_df, v2_df)
    print(f"GPT-4o: r={gpt4o_corr:.3f}, r²={gpt4o_corr**2:.3f}, p={gpt4o_pval:.3f}")
    print(f"  Confidence range: {gpt4o_data['confidence'].min():.1f}% - {gpt4o_data['confidence'].max():.1f}%")
    print(f"  Points where agrees: {gpt4o_data['agrees'].sum()}/{len(gpt4o_data)}")
    
    print("\nCreating o4-mini plot...")
    o4mini_data, o4mini_corr, o4mini_pval = create_o4mini_plot(o4mini_df, v2_df)
    print(f"o4-mini: r={o4mini_corr:.3f}, r²={o4mini_corr**2:.3f}, p={o4mini_pval:.3f}")
    print(f"  Confidence range: {o4mini_data['confidence'].min():.1f}% - {o4mini_data['confidence'].max():.1f}%")
    print(f"  Points where agrees: {o4mini_data['agrees'].sum()}/{len(o4mini_data)}")
    
    print("\nCreating combined figure...")
    create_combined_figure(gpt4o_df, o4mini_df, v2_df)
    
    print("\nPublication-ready figures created successfully!")
    print("\nFiles generated:")
    print("  - gpt4o_confidence_vs_human_consensus.png/pdf")
    print("  - o4mini_confidence_vs_human_consensus.png/pdf")
    print("  - combined_confidence_vs_human_consensus.png/pdf")

if __name__ == "__main__":
    main()