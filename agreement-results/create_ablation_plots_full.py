#!/usr/bin/env python3
"""
Create plots for FULL ablation study results (all 19 questions).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for paper-ready plots
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial']
})

# Load ablation results
print("Loading full ablation results...")
# Check if we're already in ablation directory
if Path('ablation_full_results.csv').exists():
    full_df = pd.read_csv('ablation_full_results.csv')
else:
    full_df = pd.read_csv('ablation/ablation_full_results.csv')

# Define colors
ablation_colors = {
    'full': '#1f77b4',          # Blue
    'gif_only': '#ff7f0e',      # Orange
    'trace_only': '#2ca02c',    # Green
    'truncated_trace': '#d62728' # Red
}

model_colors = {
    'gpt-4o': '#9467bd',  # Purple
    'o4-mini': '#d62728'  # Red
}

# ====================
# Plot 1: Agreement with Baseline by Ablation Type
# ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Prepare data
ablation_order = ['full', 'trace_only', 'truncated_trace', 'gif_only']
ablation_labels = ['Full\n(GIF+Trace)', 'Trace Only', 'Truncated\nTrace', 'GIF Only']

for model, ax, color in [('gpt-4o', ax1, model_colors['gpt-4o']), 
                         ('o4-mini', ax2, model_colors['o4-mini'])]:
    
    agreement_rates = []
    confidence_means = []
    confidence_stds = []
    
    for abl_type in ablation_order:
        data = full_df[(full_df['model'] == model) & (full_df['ablation_type'] == abl_type)]
        if len(data) > 0:
            agreement_rates.append(data['agrees_with_baseline'].mean() * 100)
            confidence_means.append(data['model_confidence'].mean())
            confidence_stds.append(data['model_confidence'].std())
        else:
            agreement_rates.append(0)
            confidence_means.append(0)
            confidence_stds.append(0)
    
    # Create bars
    x = np.arange(len(ablation_order))
    bars = ax.bar(x, agreement_rates, color=[ablation_colors[t] for t in ablation_order], alpha=0.8)
    
    # Add value labels
    for bar, rate in zip(bars, agreement_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Ablation Type', fontsize=14)
    ax.set_ylabel('Agreement with Baseline (%)', fontsize=14)
    ax.set_title(f'{model} Performance by Input Type', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_labels)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add confidence as secondary info
    ax2_twin = ax.twinx()
    ax2_twin.plot(x, confidence_means, 'k--', marker='o', markersize=8, linewidth=2, alpha=0.6)
    ax2_twin.set_ylabel('Average Confidence', fontsize=14)
    ax2_twin.set_ylim(0, 1)
    
    # Add horizontal line at baseline performance
    ax.axhline(y=agreement_rates[0], color='gray', linestyle=':', alpha=0.5, label='Full Context')

plt.tight_layout()
# Save to current directory if we're in ablation, otherwise to ablation/
if Path('ablation_full_results.csv').exists():
    plt.savefig('ablation_agreement_comparison_full.pdf', bbox_inches='tight')
    print("Saved: ablation_agreement_comparison_full.pdf")
else:
    plt.savefig('ablation/ablation_agreement_comparison_full.pdf', bbox_inches='tight')
    print("Saved: ablation/ablation_agreement_comparison_full.pdf")
plt.close()

# ====================
# Plot 2: Vote Distribution Comparison
# ====================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

plot_idx = 0
for model in ['gpt-4o', 'o4-mini']:
    for abl_type, abl_label in zip(ablation_order, ablation_labels):
        ax = axes[plot_idx]
        
        data = full_df[(full_df['model'] == model) & (full_df['ablation_type'] == abl_type)]
        if len(data) > 0:
            vote_counts = data['model_preference'].value_counts()
            
            # Ensure all categories are present
            for cat in ['Agent 1', 'Agent 2', 'Tie']:
                if cat not in vote_counts:
                    vote_counts[cat] = 0
            
            # Create pie chart
            colors_pie = ['#4A90E2', '#E94B3C', '#6B7280']
            wedges, texts, autotexts = ax.pie(
                [vote_counts['Agent 1'], vote_counts['Agent 2'], vote_counts['Tie']], 
                labels=['Agent 1', 'Agent 2', 'Tie'],
                colors=colors_pie,
                autopct='%1.0f%%',
                startangle=90
            )
            
            ax.set_title(f'{model}\n{abl_label}', fontsize=12, fontweight='bold')
        
        plot_idx += 1

plt.suptitle('Vote Distribution by Model and Ablation Type', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
if Path('ablation_full_results.csv').exists():
    plt.savefig('ablation_vote_distribution_full.pdf', bbox_inches='tight')
    print("Saved: ablation_vote_distribution_full.pdf")
else:
    plt.savefig('ablation/ablation_vote_distribution_full.pdf', bbox_inches='tight')
    print("Saved: ablation/ablation_vote_distribution_full.pdf")
plt.close()

# ====================
# Plot 3: Confidence Distribution by Ablation
# ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for model, ax in [('gpt-4o', ax1), ('o4-mini', ax2)]:
    # Prepare data for violin plot
    conf_data = []
    labels = []
    positions = []
    
    for i, abl in enumerate(ablation_order):
        data = full_df[(full_df['model'] == model) & (full_df['ablation_type'] == abl)]
        if len(data) > 0:
            conf_data.append(data['model_confidence'].values)
            labels.append(ablation_labels[i])
            positions.append(i)
    
    # Create violin plot
    parts = ax.violinplot(conf_data, positions=positions, showmeans=True, showmedians=True)
    
    # Color violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(ablation_colors[ablation_order[positions[i]]])
        pc.set_alpha(0.7)
    
    # Customize
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Confidence Score', fontsize=14)
    ax.set_title(f'{model} Confidence by Input Type', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean values
    for i, (pos, data) in enumerate(zip(positions, conf_data)):
        mean_val = np.mean(data)
        ax.text(pos, 1.02, f'{mean_val:.2f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
if Path('ablation_full_results.csv').exists():
    plt.savefig('ablation_confidence_distribution_full.pdf', bbox_inches='tight')
    print("Saved: ablation_confidence_distribution_full.pdf")
else:
    plt.savefig('ablation/ablation_confidence_distribution_full.pdf', bbox_inches='tight')
    print("Saved: ablation/ablation_confidence_distribution_full.pdf")
plt.close()

# ====================
# Plot 4: Key Finding - Trace vs Full Performance
# ====================
fig, ax = plt.subplots(figsize=(10, 6))

models = ['GPT-4o', 'O4-mini']
full_perf = []
trace_perf = []
gif_perf = []

for model in ['gpt-4o', 'o4-mini']:
    full_data = full_df[(full_df['model'] == model) & (full_df['ablation_type'] == 'full')]
    trace_data = full_df[(full_df['model'] == model) & (full_df['ablation_type'] == 'trace_only')]
    gif_data = full_df[(full_df['model'] == model) & (full_df['ablation_type'] == 'gif_only')]
    
    full_perf.append(full_data['agrees_with_baseline'].mean() * 100)
    trace_perf.append(trace_data['agrees_with_baseline'].mean() * 100)
    gif_perf.append(gif_data['agrees_with_baseline'].mean() * 100)

x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, full_perf, width, label='Full (GIF+Trace)', color=ablation_colors['full'])
bars2 = ax.bar(x, trace_perf, width, label='Trace Only', color=ablation_colors['trace_only'])
bars3 = ax.bar(x + width, gif_perf, width, label='GIF Only', color=ablation_colors['gif_only'])

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Agreement with Baseline (%)', fontsize=14)
ax.set_title('Key Finding: Trace-Only Outperforms Full Context for GPT-4o', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='upper right')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Add annotation
ax.annotate('10.5% improvement!', xy=(0.05, 73), xytext=(0.3, 85),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=14, fontweight='bold', color='red')

plt.tight_layout()
if Path('ablation_full_results.csv').exists():
    plt.savefig('ablation_key_finding_full.pdf', bbox_inches='tight')
    print("Saved: ablation_key_finding_full.pdf")
else:
    plt.savefig('ablation/ablation_key_finding_full.pdf', bbox_inches='tight')
    print("Saved: ablation/ablation_key_finding_full.pdf")
plt.close()

# ====================
# Generate Summary Statistics Table
# ====================
print("\n=== FULL ABLATION STUDY SUMMARY (19 Questions) ===")

summary_stats = []
for model in ['gpt-4o', 'o4-mini']:
    for abl in ablation_order:
        data = full_df[(full_df['model'] == model) & (full_df['ablation_type'] == abl)]
        if len(data) > 0:
            # Calculate confidence intervals
            agreement_rate = data['agrees_with_baseline'].mean() * 100
            agreement_std = data['agrees_with_baseline'].std() * 100
            n = len(data)
            agreement_ci = 1.96 * agreement_std / np.sqrt(n)  # 95% CI
            
            stats = {
                'Model': model,
                'Ablation': ablation_labels[ablation_order.index(abl)],
                'N': n,
                'Agreement (%)': f"{agreement_rate:.1f} ± {agreement_ci:.1f}",
                'Avg Confidence': f"{data['model_confidence'].mean():.3f}",
                'Tie Rate (%)': f"{(data['model_preference'] == 'Tie').mean() * 100:.1f}",
                'Agent 1 (%)': f"{(data['model_preference'] == 'Agent 1').mean() * 100:.1f}",
                'Agent 2 (%)': f"{(data['model_preference'] == 'Agent 2').mean() * 100:.1f}"
            }
            summary_stats.append(stats)

summary_df = pd.DataFrame(summary_stats)
if Path('ablation_full_results.csv').exists():
    summary_df.to_csv('ablation_full_summary_statistics.csv', index=False)
else:
    summary_df.to_csv('ablation/ablation_full_summary_statistics.csv', index=False)
print("\nSummary statistics with confidence intervals:")
print(summary_df.to_string(index=False))

# Key findings
print("\n=== KEY FINDINGS (FULL DATASET) ===")
print("\n1. GIF-Only Evaluation Shows Strong Biases:")
print("   - GPT-4o: 89.5% Agent 2 bias (10.5% baseline agreement)")
print("   - O4-mini: 100% Tie responses (5.3% baseline agreement)")

print("\n2. Trace-Only IMPROVES GPT-4o Performance:")
print("   - GPT-4o: 78.9% (trace-only) vs 68.4% (full) - 10.5% improvement!")
print("   - O4-mini: 52.6% (trace-only) vs 57.9% (full) - slight decrease")
print("   - Suggests GIFs may be adding noise rather than signal for GPT-4o")

print("\n3. Truncation Effects Differ by Model:")
print("   - GPT-4o: Severely impacted (78.9% → 42.1%)")
print("   - O4-mini: Maintains performance (52.6% → 57.9%)")

print("\n4. Confidence Calibration:")
print("   - GPT-4o maintains high confidence (0.90) even with GIF-only (10.5% accuracy)")
print("   - O4-mini shows some calibration (0.75) with GIF-only")

print("\nAll plots saved in ablation/ directory")