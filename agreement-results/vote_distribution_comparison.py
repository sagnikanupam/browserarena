#!/usr/bin/env python3
"""Create a detailed vote distribution comparison between original and new annotators"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the analysis results
results_df = pd.read_csv('agreement_analysis_results.csv')

# Read baseline for additional context
baseline_df = pd.read_csv('baseline.csv')

# Create the comparison table
print("VOTE DISTRIBUTION: ORIGINAL ANNOTATOR VS NEW ANNOTATORS")
print("=" * 80)
print(f"{'Question':<10} {'Original':<12} {'Agent 1':<10} {'Agent 2':<10} {'Tie':<10} {'Total':<10} {'Majority':<12}")
print("-" * 80)

for _, row in results_df.iterrows():
    question = row['question']
    original = row['original_label']
    agent1_votes = row['agent_1_votes']
    agent2_votes = row['agent_2_votes']
    tie_votes = row['tie_votes']
    total = row['n_responses']
    majority = row['majority_label']
    
    print(f"{question:<10} {original:<12} {agent1_votes:<10} {agent2_votes:<10} {tie_votes:<10} {total:<10} {majority:<12}")

print("-" * 80)

# Calculate summary statistics
print("\nSUMMARY STATISTICS:")
print(f"Total responses: {results_df['n_responses'].sum()}")
print(f"Average responses per question: {results_df['n_responses'].mean():.1f}")

# Count agreements/disagreements
agreements = (results_df['original_label'] == results_df['majority_label']).sum()
disagreements = len(results_df) - agreements
print(f"\nAgreements with original: {agreements}/{len(results_df)} ({agreements/len(results_df)*100:.1f}%)")
print(f"Disagreements with original: {disagreements}/{len(results_df)} ({disagreements/len(results_df)*100:.1f}%)")

# Analyze vote shifts
print("\nVOTE SHIFT ANALYSIS:")
print("Questions where new annotators shifted from original label:")
for _, row in results_df[results_df['original_label'] != results_df['majority_label']].iterrows():
    question = row['question']
    original = row['original_label']
    majority = row['majority_label']
    
    # Calculate percentage for each option
    total = row['n_responses']
    agent1_pct = (row['agent_1_votes'] / total * 100) if total > 0 else 0
    agent2_pct = (row['agent_2_votes'] / total * 100) if total > 0 else 0
    tie_pct = (row['tie_votes'] / total * 100) if total > 0 else 0
    
    print(f"\n{question}: {original} → {majority}")
    print(f"  Distribution: Agent 1: {row['agent_1_votes']} ({agent1_pct:.1f}%), "
          f"Agent 2: {row['agent_2_votes']} ({agent2_pct:.1f}%), "
          f"Tie: {row['tie_votes']} ({tie_pct:.1f}%)")

# Create visualization comparing original vs new majority
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Count original labels
original_counts = results_df['original_label'].value_counts()
colors = ['#3498db', '#e74c3c', '#95a5a6']

# Original annotator distribution
ax1.pie(original_counts.values, labels=original_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Original Annotator Distribution')

# New annotators' majority distribution
majority_counts = results_df['majority_label'].value_counts()
ax2.pie(majority_counts.values, labels=majority_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('New Annotators Majority Distribution')

plt.tight_layout()
plt.savefig('original_vs_new_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Create detailed per-question comparison chart
fig, ax = plt.subplots(figsize=(16, 10))

questions = results_df['question'].tolist()
x = np.arange(len(questions))
width = 0.35

# Prepare data for original annotator (single vote per question)
original_agent1 = []
original_agent2 = []
original_tie = []

for _, row in results_df.iterrows():
    if row['original_label'] == 'Agent 1':
        original_agent1.append(1)
        original_agent2.append(0)
        original_tie.append(0)
    elif row['original_label'] == 'Agent 2':
        original_agent1.append(0)
        original_agent2.append(1)
        original_tie.append(0)
    else:  # Tie
        original_agent1.append(0)
        original_agent2.append(0)
        original_tie.append(1)

# New annotators data (already in results_df)
new_agent1 = results_df['agent_1_votes'].tolist()
new_agent2 = results_df['agent_2_votes'].tolist()
new_tie = results_df['tie_votes'].tolist()

# Create grouped bar chart
bar_width = 0.25
r1 = np.arange(len(questions))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plot new annotators' votes
plt.bar(r1, new_agent1, color='#3498db', width=bar_width, label='Agent 1', alpha=0.8)
plt.bar(r2, new_agent2, color='#e74c3c', width=bar_width, label='Agent 2', alpha=0.8)
plt.bar(r3, new_tie, color='#95a5a6', width=bar_width, label='Tie', alpha=0.8)

# Add markers for original annotations
for i, q in enumerate(questions):
    if results_df.iloc[i]['original_label'] == 'Agent 1':
        plt.scatter(r1[i], new_agent1[i] + 1, color='black', s=100, marker='v', zorder=5)
    elif results_df.iloc[i]['original_label'] == 'Agent 2':
        plt.scatter(r2[i], new_agent2[i] + 1, color='black', s=100, marker='v', zorder=5)
    else:  # Tie
        plt.scatter(r3[i], new_tie[i] + 1, color='black', s=100, marker='v', zorder=5)

plt.xlabel('Question', fontsize=12)
plt.ylabel('Number of Votes', fontsize=12)
plt.title('Vote Distribution per Question: New Annotators\n(Black triangles indicate original annotator choice)', fontsize=14)
plt.xticks([r + bar_width for r in range(len(questions))], questions, rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('vote_distribution_detailed.png', dpi=300, bbox_inches='tight')
plt.close()

# Save detailed comparison to CSV
comparison_data = []
for _, row in results_df.iterrows():
    total = row['n_responses']
    comparison_data.append({
        'Question': row['question'],
        'Original_Annotator': row['original_label'],
        'New_Majority': row['majority_label'],
        'Agent1_Votes': f"{row['agent_1_votes']} ({row['agent_1_votes']/total*100:.1f}%)" if total > 0 else "0",
        'Agent2_Votes': f"{row['agent_2_votes']} ({row['agent_2_votes']/total*100:.1f}%)" if total > 0 else "0",
        'Tie_Votes': f"{row['tie_votes']} ({row['tie_votes']/total*100:.1f}%)" if total > 0 else "0",
        'Total_Responses': total,
        'Agreement': 'Yes' if row['original_label'] == row['majority_label'] else 'No'
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('vote_distribution_comparison.csv', index=False)

print("\n\nFiles created:")
print("- vote_distribution_comparison.csv")
print("- original_vs_new_distribution.png")
print("- vote_distribution_detailed.png")