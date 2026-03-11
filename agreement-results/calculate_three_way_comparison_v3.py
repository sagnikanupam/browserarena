#!/usr/bin/env python3
"""
Calculate three-way comparison including exact GPT-4o evaluations.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import krippendorff

print("=== THREE-WAY COMPARISON V3: BASELINE vs HUMAN V2 vs GPT-4O (EXACT) ===")

# Read all data sources
baseline_df = pd.read_csv('baseline.csv')
human_v2_df = pd.read_csv('BrowserArenaAgreementv2_July_18_2025_11.01.csv')
gpt4o_exact_df = pd.read_csv('survey_questions_gpt4o_exact_evaluations.csv')

# Process human V2 data
human_v2_df = human_v2_df.iloc[2:]  # Skip headers
human_v2_df = human_v2_df[human_v2_df['Status'] == '0']

# Get survey questions
survey_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 
                   'Q11', 'Q12', 'Q13', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q22']

# Process human V2 votes
print("\nProcessing Human V2 votes...")
human_v2_majority = {}
human_v2_stats = {}

for q in survey_questions:
    if q in human_v2_df.columns:
        responses = human_v2_df[q].dropna()
        
        # Count votes
        vote_counts = {
            'Agent 1': 0,
            'Agent 2': 0,
            'Tie': 0
        }
        
        for response in responses:
            response_str = str(response).strip()
            if response_str == '1':
                vote_counts['Agent 1'] += 1
            elif response_str == '2':
                vote_counts['Agent 2'] += 1
            elif response_str == '3':
                vote_counts['Tie'] += 1
        
        total_votes = sum(vote_counts.values())
        if total_votes > 0:
            # Get majority
            majority = max(vote_counts.items(), key=lambda x: x[1])
            human_v2_majority[q] = majority[0]
            
            human_v2_stats[q] = {
                'Agent1_votes': vote_counts['Agent 1'],
                'Agent2_votes': vote_counts['Agent 2'],
                'Tie_votes': vote_counts['Tie'],
                'Total': total_votes,
                'Agreement_rate': f"{(majority[1]/total_votes)*100:.1f}%"
            }

# Create comparison dataframe
comparison_data = []

for q in survey_questions:
    # Get baseline
    baseline_row = baseline_df[baseline_df['question'] == q]
    if baseline_row.empty:
        continue
    
    baseline_winner = baseline_row.iloc[0]['winner']
    
    # Get human V2
    human_v2_winner = human_v2_majority.get(q, 'N/A')
    human_stats = human_v2_stats.get(q, {})
    
    # Get GPT-4o exact
    gpt4o_row = gpt4o_exact_df[gpt4o_exact_df['question'] == q]
    if not gpt4o_row.empty:
        gpt4o_winner = gpt4o_row.iloc[0]['gpt4o_preference']
        gpt4o_confidence = gpt4o_row.iloc[0]['gpt4o_confidence']
    else:
        gpt4o_winner = 'N/A'
        gpt4o_confidence = 0
    
    comparison_data.append({
        'Question': q,
        'Original_Baseline': baseline_winner,
        'Human_V2_Majority': human_v2_winner,
        'GPT4o_Exact': gpt4o_winner,
        'GPT4o_Confidence': gpt4o_confidence,
        'V2_Agent1_Votes': human_stats.get('Agent1_votes', 0),
        'V2_Agent2_Votes': human_stats.get('Agent2_votes', 0),
        'V2_Tie_Votes': human_stats.get('Tie_votes', 0),
        'V2_Total': human_stats.get('Total', 0),
        'V2_Agreement_Rate': human_stats.get('Agreement_rate', 'N/A'),
        'All_Three_Agree': (baseline_winner == human_v2_winner == gpt4o_winner),
        'Baseline_Human_Agree': (baseline_winner == human_v2_winner),
        'Baseline_GPT4o_Agree': (baseline_winner == gpt4o_winner),
        'Human_GPT4o_Agree': (human_v2_winner == gpt4o_winner)
    })

comparison_df = pd.DataFrame(comparison_data)

# Save the comparison
comparison_df.to_csv('three_way_comparison_v3.csv', index=False)
print(f"\nSaved three-way comparison to three_way_comparison_v3.csv")

# Calculate agreement statistics
print("\n=== AGREEMENT STATISTICS ===")
total_questions = len(comparison_df)

all_agree = comparison_df['All_Three_Agree'].sum()
baseline_human = comparison_df['Baseline_Human_Agree'].sum()
baseline_gpt4o = comparison_df['Baseline_GPT4o_Agree'].sum()
human_gpt4o = comparison_df['Human_GPT4o_Agree'].sum()

print(f"Total questions: {total_questions}")
print(f"\nThree-way agreement: {all_agree}/{total_questions} ({all_agree/total_questions*100:.1f}%)")
print(f"\nPairwise agreements:")
print(f"  Baseline vs Human V2: {baseline_human}/{total_questions} ({baseline_human/total_questions*100:.1f}%)")
print(f"  Baseline vs GPT-4o: {baseline_gpt4o}/{total_questions} ({baseline_gpt4o/total_questions*100:.1f}%)")
print(f"  Human V2 vs GPT-4o: {human_gpt4o}/{total_questions} ({human_gpt4o/total_questions*100:.1f}%)")

# Calculate Cohen's Kappa for each pair
print("\n=== COHEN'S KAPPA (PAIRWISE) ===")

# Prepare labels for kappa calculation
labels_baseline = comparison_df['Original_Baseline'].tolist()
labels_human = comparison_df['Human_V2_Majority'].tolist()
labels_gpt4o = comparison_df['GPT4o_Exact'].tolist()

kappa_baseline_human = cohen_kappa_score(labels_baseline, labels_human)
kappa_baseline_gpt4o = cohen_kappa_score(labels_baseline, labels_gpt4o)
kappa_human_gpt4o = cohen_kappa_score(labels_human, labels_gpt4o)

print(f"Baseline vs Human V2: {kappa_baseline_human:.3f}")
print(f"Baseline vs GPT-4o: {kappa_baseline_gpt4o:.3f}")
print(f"Human V2 vs GPT-4o: {kappa_human_gpt4o:.3f}")

# Calculate Krippendorff's alpha for all three
print("\n=== KRIPPENDORFF'S ALPHA (ALL THREE) ===")

# Convert labels to numeric for Krippendorff
label_map = {'Agent 1': 1, 'Agent 2': 2, 'Tie': 3}
data_matrix = []
for labels in [labels_baseline, labels_human, labels_gpt4o]:
    numeric_labels = [label_map.get(label, 0) for label in labels]
    data_matrix.append(numeric_labels)

data_matrix = np.array(data_matrix)
alpha = krippendorff.alpha(reliability_data=data_matrix, level_of_measurement='nominal')
print(f"Krippendorff's alpha (all three): {alpha:.3f}")

# Create visualization
print("\n=== CREATING VISUALIZATIONS ===")

# 1. Agreement heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# Create agreement matrix
evaluators = ['Baseline', 'Human V2', 'GPT-4o']
agreement_matrix = np.array([
    [total_questions, baseline_human, baseline_gpt4o],
    [baseline_human, total_questions, human_gpt4o],
    [baseline_gpt4o, human_gpt4o, total_questions]
])

# Convert to percentages
agreement_pct = (agreement_matrix / total_questions) * 100

# Create heatmap
sns.heatmap(agreement_pct, annot=True, fmt='.1f', cmap='YlOrRd', 
            xticklabels=evaluators, yticklabels=evaluators,
            vmin=0, vmax=100, cbar_kws={'label': 'Agreement %'},
            square=True, linewidths=2, linecolor='white')

plt.title('Three-Way Agreement Heatmap\n(Baseline vs Human V2 vs GPT-4o Exact)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('three_way_agreement_heatmap_v3.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Vote distribution comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (source, ax) in enumerate(zip(['Baseline', 'Human V2', 'GPT-4o'], axes)):
    if source == 'Baseline':
        votes = comparison_df['Original_Baseline'].value_counts()
    elif source == 'Human V2':
        votes = comparison_df['Human_V2_Majority'].value_counts()
    else:
        votes = comparison_df['GPT4o_Exact'].value_counts()
    
    # Ensure all categories are present
    for cat in ['Agent 1', 'Agent 2', 'Tie']:
        if cat not in votes:
            votes[cat] = 0
    
    votes = votes.reindex(['Agent 1', 'Agent 2', 'Tie'])
    
    colors = ['#2E86AB', '#E63946', '#F77F00']
    bars = ax.bar(votes.index, votes.values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}\n({height/total_questions*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_title(f'{source} Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Questions', fontsize=12)
    ax.set_ylim(0, max(20, votes.max() + 3))
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Vote Distribution Across Three Evaluators', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('vote_distribution_comparison_v3.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Disagreement analysis
print("\n=== DISAGREEMENT ANALYSIS ===")

# Questions where all three disagree
all_different = comparison_df[
    (comparison_df['Original_Baseline'] != comparison_df['Human_V2_Majority']) &
    (comparison_df['Original_Baseline'] != comparison_df['GPT4o_Exact']) &
    (comparison_df['Human_V2_Majority'] != comparison_df['GPT4o_Exact'])
]

print(f"\nQuestions where all three evaluators chose differently: {len(all_different)}")
for _, row in all_different.iterrows():
    print(f"  {row['Question']}: Baseline={row['Original_Baseline']}, "
          f"Human={row['Human_V2_Majority']}, GPT-4o={row['GPT4o_Exact']}")

# Questions where GPT-4o disagrees with both
gpt4o_lone_wolf = comparison_df[
    (comparison_df['Original_Baseline'] == comparison_df['Human_V2_Majority']) &
    (comparison_df['Original_Baseline'] != comparison_df['GPT4o_Exact'])
]

print(f"\nQuestions where GPT-4o disagrees with both Baseline and Human: {len(gpt4o_lone_wolf)}")
for _, row in gpt4o_lone_wolf.iterrows():
    print(f"  {row['Question']}: Baseline/Human={row['Original_Baseline']}, "
          f"GPT-4o={row['GPT4o_Exact']} (conf: {row['GPT4o_Confidence']:.2f})")

# Average GPT-4o confidence by agreement pattern
print("\n=== GPT-4O CONFIDENCE BY AGREEMENT PATTERN ===")
patterns = {
    'All agree': comparison_df[comparison_df['All_Three_Agree']],
    'GPT-4o agrees with baseline only': comparison_df[
        (comparison_df['Baseline_GPT4o_Agree']) & 
        (~comparison_df['Human_GPT4o_Agree'])
    ],
    'GPT-4o agrees with human only': comparison_df[
        (~comparison_df['Baseline_GPT4o_Agree']) & 
        (comparison_df['Human_GPT4o_Agree'])
    ],
    'GPT-4o disagrees with both': comparison_df[
        (~comparison_df['Baseline_GPT4o_Agree']) & 
        (~comparison_df['Human_GPT4o_Agree'])
    ]
}

for pattern, data in patterns.items():
    if len(data) > 0:
        avg_conf = data['GPT4o_Confidence'].mean()
        print(f"{pattern}: {len(data)} questions, avg confidence = {avg_conf:.3f}")

print("\nAnalysis complete!")