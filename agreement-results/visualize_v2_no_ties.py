import matplotlib.pyplot as plt
import json
import numpy as np

# Load the analysis results
with open('v2_no_ties_analysis.json', 'r') as f:
    data = json.load(f)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('V2 Inter-Annotator Agreement Analysis (Excluding Ties)', fontsize=16)

# 1. Overall Pie Chart - Tie vs No-Tie Comparisons
ax1 = axes[0, 0]
tie_data = [data['summary']['comparisons_with_ties'], 
            data['summary']['comparisons_without_ties']]
labels = [f"With Ties\n{tie_data[0]:,}\n({tie_data[0]/sum(tie_data)*100:.1f}%)",
          f"No Ties\n{tie_data[1]:,}\n({tie_data[1]/sum(tie_data)*100:.1f}%)"]
colors = ['#ff9999', '#66b3ff']
ax1.pie(tie_data, labels=labels, colors=colors, autopct='', startangle=90)
ax1.set_title('Distribution of Comparisons')

# 2. Agreement Rate for No-Tie Cases
ax2 = axes[0, 1]
agreement_data = [data['summary']['agreements_no_ties'],
                  data['summary']['disagreements_no_ties']]
labels = [f"Agree\n{agreement_data[0]:,}",
          f"Disagree\n{agreement_data[1]:,}"]
colors = ['#90EE90', '#FFB6C1']
ax2.pie(agreement_data, labels=labels, colors=colors, 
        autopct='%1.1f%%', startangle=90)
ax2.set_title(f'Agreement on Decisive Votes Only\n(Agreement Rate: {data["summary"]["agreement_rate_no_ties"]*100:.2f}%)')

# 3. Per-Question Agreement Rates
ax3 = axes[1, 0]
questions = sorted(data['per_question'].keys(), key=lambda x: int(x[1:]))
agreement_rates = []
for q in questions:
    if data['per_question'][q]['without_ties'] > 0:
        rate = data['per_question'][q]['agreements'] / data['per_question'][q]['without_ties'] * 100
    else:
        rate = 0
    agreement_rates.append(rate)

x_pos = np.arange(len(questions))
bars = ax3.bar(x_pos, agreement_rates)

# Color bars based on agreement rate
for i, (bar, rate) in enumerate(zip(bars, agreement_rates)):
    if rate >= 80:
        bar.set_color('#2ecc71')  # Green for high agreement
    elif rate >= 60:
        bar.set_color('#f39c12')  # Orange for medium agreement
    else:
        bar.set_color('#e74c3c')  # Red for low agreement

ax3.set_xlabel('Questions')
ax3.set_ylabel('Agreement Rate (%)')
ax3.set_title('Agreement Rates by Question (No Ties)')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(questions, rotation=45)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 105)

# Add overall average line
overall_rate = data['summary']['agreement_rate_no_ties'] * 100
ax3.axhline(y=overall_rate, color='red', linestyle='--', 
            label=f'Overall: {overall_rate:.2f}%')
ax3.legend()

# 4. Proportion of No-Tie Cases by Question
ax4 = axes[1, 1]
no_tie_proportions = []
for q in questions:
    total = data['per_question'][q]['total']
    no_ties = data['per_question'][q]['without_ties']
    if total > 0:
        proportion = no_ties / total * 100
    else:
        proportion = 0
    no_tie_proportions.append(proportion)

bars2 = ax4.bar(x_pos, no_tie_proportions)

# Color bars based on proportion
for bar, prop in zip(bars2, no_tie_proportions):
    if prop >= 80:
        bar.set_color('#3498db')  # Blue for high proportion
    elif prop >= 50:
        bar.set_color('#9b59b6')  # Purple for medium proportion
    else:
        bar.set_color('#95a5a6')  # Gray for low proportion

ax4.set_xlabel('Questions')
ax4.set_ylabel('Proportion of No-Tie Comparisons (%)')
ax4.set_title('Decisive Vote Proportion by Question')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(questions, rotation=45)
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim(0, 105)

# Add 50% reference line
ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
ax4.legend()

plt.tight_layout()
plt.savefig('v2_no_ties_agreement_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a detailed comparison plot
fig2, ax = plt.subplots(figsize=(12, 8))

# Prepare data for grouped bar chart
x = np.arange(len(questions))
width = 0.35

# Calculate values for each question
total_comparisons = []
decisive_comparisons = []
agreements = []

for q in questions:
    total_comparisons.append(data['per_question'][q]['total'])
    decisive_comparisons.append(data['per_question'][q]['without_ties'])
    agreements.append(data['per_question'][q]['agreements'])

# Create bars
bars1 = ax.bar(x - width/2, total_comparisons, width, label='Total Comparisons', alpha=0.7)
bars2 = ax.bar(x + width/2, decisive_comparisons, width, label='Decisive Comparisons', alpha=0.7)

# Add agreement numbers on top of decisive comparison bars
for i, (bar, agree) in enumerate(zip(bars2, agreements)):
    height = bar.get_height()
    if height > 0:
        agreement_pct = (agree / decisive_comparisons[i]) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{agreement_pct:.0f}%', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Questions')
ax.set_ylabel('Number of Comparisons')
ax.set_title('Total vs Decisive Comparisons by Question')
ax.set_xticks(x)
ax.set_xticklabels(questions, rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('v2_no_ties_comparison_detail.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations saved:")
print("- v2_no_ties_agreement_analysis.png")
print("- v2_no_ties_comparison_detail.png")