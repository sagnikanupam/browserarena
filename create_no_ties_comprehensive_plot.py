#!/usr/bin/env python3
"""
Create comprehensive visualization for no-ties analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def create_comprehensive_no_ties_plot():
    """Create a comprehensive plot showing all no-ties analysis"""
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Pairwise Agreement Analysis: Impact of Tie Votes', fontsize=20, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Agreement rates comparison
    ax1 = fig.add_subplot(gs[0, :2])
    
    categories = ['GPT-4 vs Survey', 'GPT-4 vs Baseline', 'Survey vs Baseline', 'All Three Agree']
    all_interactions = [52.6, 73.7, 63.2, 52.6]
    no_ties_only = [90.9, 90.9, 100.0, 90.9]
    forced_binary = [77.8, 77.8, 100.0, None]  # No "all three" for forced
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax1.bar(x - width, all_interactions, width, label='All Interactions', color='#1f77b4', alpha=0.8)
    bars2 = ax1.bar(x, no_ties_only, width, label='No Ties Only (11 cases)', color='#ff7f0e', alpha=0.8)
    bars3 = ax1.bar(x + width, forced_binary[:3] + [0], width, label='Forced Binary (18 cases)', color='#2ca02c', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax1.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
    
    ax1.set_ylabel('Agreement Rate (%)', fontsize=12)
    ax1.set_title('Agreement Rates Across Different Tie Handling Methods', fontsize=14, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=15, ha='right')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 110)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Sample size impact
    ax2 = fig.add_subplot(gs[0, 2])
    
    labels = ['All\nInteractions', 'No Ties\nOnly', 'Forced\nBinary']
    sizes = [19, 11, 18]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax2.bar(labels, sizes, color=colors, alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Number of Interactions', fontsize=12)
    ax2.set_title('Sample Sizes', fontsize=14)
    ax2.set_ylim(0, 25)
    
    # 3. No-tie interaction details
    ax3 = fig.add_subplot(gs[1, :])
    
    # Create visualization of which interactions had no ties
    interactions = [f'{i:02d}' for i in range(1, 20)]
    no_tie_interactions = ['01', '02', '03', '04', '06', '08', '09', '10', '13', '17', '18']
    tie_interactions = ['05', '07', '11', '12', '14', '15', '16', '19']
    
    y_pos = 0.5
    for i, int_id in enumerate(interactions):
        x_pos = i * 0.05 + 0.05
        if int_id in no_tie_interactions:
            color = '#2ca02c' if int_id != '17' else '#ff7f0e'  # Highlight disagreement
            marker = 'o'
        else:
            color = '#cccccc'
            marker = 's'
        
        ax3.scatter(x_pos, y_pos, s=200, c=color, marker=marker, edgecolors='black', linewidth=1)
        ax3.text(x_pos, y_pos, int_id, ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.text(0.5, 0.8, 'Interaction Tie Status', ha='center', fontsize=14, fontweight='bold')
    ax3.text(0.5, 0.2, 'Green circles: No ties from any evaluator | Gray squares: At least one tie vote | Orange: Disagreement',
             ha='center', fontsize=10)
    
    # 4. Tie resolution analysis
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Survey tie resolutions
    tie_resolutions = {
        '05': ('Left', 3, 'L:8 R:5 T:11'),
        '07': ('Left', 9, 'L:11 R:2 T:14'),
        '11': ('Left', 2, 'L:4 R:2 T:18'),
        '12': ('Left', 2, 'L:7 R:5 T:12'),
        '14': ('Left', 5, 'L:8 R:3 T:13'),
        '15': ('Left', 4, 'L:8 R:4 T:12'),
        '16': ('Left', 4, 'L:8 R:4 T:8'),
        '19': ('Right', 5, 'L:4 R:9 T:15')
    }
    
    # All resolved to Left except one
    left_count = sum(1 for v in tie_resolutions.values() if v[0] == 'Left')
    right_count = sum(1 for v in tie_resolutions.values() if v[0] == 'Right')
    
    sizes = [left_count, right_count]
    labels = [f'Resolved to Left\n({left_count} cases)', f'Resolved to Right\n({right_count} case)']
    colors = ['#1f77b4', '#ff7f0e']
    
    wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                                       startangle=90, textprops={'fontsize': 10})
    ax4.set_title('Survey Tie Resolution Direction', fontsize=12)
    
    # 5. Agreement improvement
    ax5 = fig.add_subplot(gs[2, 1])
    
    improvements = {
        'GPT-4 vs Survey': (52.6, 90.9),
        'GPT-4 vs Baseline': (73.7, 90.9),
        'Survey vs Baseline': (63.2, 100.0)
    }
    
    labels = list(improvements.keys())
    original = [v[0] for v in improvements.values()]
    no_ties = [v[1] for v in improvements.values()]
    improvement = [no_ties[i] - original[i] for i in range(len(original))]
    
    x = np.arange(len(labels))
    ax5.bar(x, improvement, color=['#2ca02c' if imp > 0 else '#ff7f0e' for imp in improvement])
    
    for i, (orig, new, imp) in enumerate(zip(original, no_ties, improvement)):
        ax5.annotate(f'+{imp:.1f}%',
                    xy=(i, imp),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax5.text(i, -5, f'{orig:.1f}%→{new:.1f}%', ha='center', fontsize=8)
    
    ax5.set_ylabel('Agreement Rate Change (%)', fontsize=10)
    ax5.set_title('Agreement Improvement\n(No Ties Only)', fontsize=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels, rotation=15, ha='right')
    ax5.axhline(y=0, color='black', linewidth=0.5)
    ax5.set_ylim(-10, 50)
    
    # 6. Key findings
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    findings = [
        "Key Findings:",
        "",
        "• Without ties: 90.9% agreement",
        "  (vs 52-74% with ties)",
        "",
        "• Only 1 disagreement in 11 cases",
        "  (interaction_17)",
        "",
        "• 7/8 survey ties → Left when forced",
        "",
        "• Survey-Baseline: 100% agreement",
        "  when ties excluded"
    ]
    
    for i, finding in enumerate(findings):
        weight = 'bold' if i == 0 else 'normal'
        size = 12 if i == 0 else 10
        ax6.text(0.1, 0.9 - i*0.08, finding, fontsize=size, fontweight=weight, va='top')
    
    plt.tight_layout()
    plt.savefig('no_ties_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Comprehensive no-ties analysis plot saved: no_ties_comprehensive_analysis.png")

if __name__ == "__main__":
    create_comprehensive_no_ties_plot()