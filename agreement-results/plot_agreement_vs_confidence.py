#!/usr/bin/env python3
"""Plot per-question human agreement vs GPT-4o confidence scores"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score

# Read the GPT-4o evaluation data
print("Reading GPT-4o evaluation data...")
gpt4o_df = pd.read_csv('../gpt4o_evaluation_analysis.csv')

# Read the per-question agreement data
print("Reading V2 per-question agreement data...")
agreement_df = pd.read_csv('v2_per_question_agreement.csv')

# Read the baseline data to map questions to case IDs
print("Reading baseline data...")
baseline_df = pd.read_csv('baseline.csv')

# Create a mapping from question numbers to case IDs
# The baseline.csv has case_id and question columns
question_to_case = {}
for _, row in baseline_df.iterrows():
    question_to_case[row['question']] = row['case_id']

print(f"\nFound {len(question_to_case)} question-to-case mappings")

# Calculate average confidence per question (case_id)
# Group by case_id and calculate mean confidence
confidence_by_case = gpt4o_df.groupby('case_id')['gpt4o_confidence'].mean().reset_index()
confidence_by_case.columns = ['case_id', 'avg_confidence']

print(f"Calculated average confidence for {len(confidence_by_case)} cases")

# Merge the data
# First, add case_id to agreement data
agreement_df['case_id'] = agreement_df['question'].map(question_to_case)

# Then merge with confidence data
merged_df = agreement_df.merge(confidence_by_case, on='case_id', how='inner')

print(f"\nMerged data has {len(merged_df)} rows")
print(merged_df[['question', 'pairwise_agreement', 'avg_confidence']].head())

# Filter out any rows with missing data
merged_df = merged_df.dropna(subset=['pairwise_agreement', 'avg_confidence'])

if len(merged_df) == 0:
    print("\nError: No data after merging. Checking data...")
    print("Sample case IDs from GPT4o data:", gpt4o_df['case_id'].head().tolist())
    print("Sample case IDs from baseline:", list(question_to_case.values())[:5])
    
    # Try alternative approach - use response_id matching
    print("\nTrying alternative approach using question numbers directly...")
    
    # Extract question numbers from GPT4o data if possible
    # Look for patterns in case_id that might contain question numbers
    
    # For now, let's create a synthetic mapping based on order
    # This assumes the questions in baseline.csv are in the same order as evaluations
    questions = sorted(agreement_df['question'].unique())
    unique_cases = confidence_by_case['case_id'].unique()
    
    print(f"Questions: {questions}")
    print(f"Number of unique cases in GPT4o data: {len(unique_cases)}")

# Create the scatter plot
plt.figure(figsize=(10, 8))

# Plot the scatter points
plt.scatter(merged_df['avg_confidence'], merged_df['pairwise_agreement'], 
           s=100, alpha=0.6, edgecolors='black', linewidth=1)

# Add labels for each point
for idx, row in merged_df.iterrows():
    plt.annotate(row['question'], 
                (row['avg_confidence'], row['pairwise_agreement']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

# Calculate correlation and R²
if len(merged_df) > 0:
    correlation = merged_df['avg_confidence'].corr(merged_df['pairwise_agreement'])
    r2 = r2_score(merged_df['pairwise_agreement'], 
                  np.poly1d(np.polyfit(merged_df['avg_confidence'], 
                                      merged_df['pairwise_agreement'], 1))(merged_df['avg_confidence']))
    
    # Add regression line
    z = np.polyfit(merged_df['avg_confidence'], merged_df['pairwise_agreement'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_df['avg_confidence'].min(), merged_df['avg_confidence'].max(), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Linear fit (R² = {r2:.3f})')
    
    # Calculate p-value
    _, p_value = stats.pearsonr(merged_df['avg_confidence'], merged_df['pairwise_agreement'])
    
    print(f"\n=== CORRELATION ANALYSIS ===")
    print(f"Pearson correlation: {correlation:.3f}")
    print(f"R² value: {r2:.3f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
else:
    print("\nNo data available for correlation analysis")

plt.xlabel('GPT-4o Average Confidence Score', fontsize=12)
plt.ylabel('Human Pairwise Agreement Rate', fontsize=12)
plt.title('Human Inter-Annotator Agreement vs GPT-4o Confidence\nby Question', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# Set axis limits
if len(merged_df) > 0:
    plt.xlim(merged_df['avg_confidence'].min() - 0.05, merged_df['avg_confidence'].max() + 0.05)
    plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('agreement_vs_confidence_scatter.png', dpi=300)
plt.close()

print("\nPlot saved as agreement_vs_confidence_scatter.png")

# Create a summary table
if len(merged_df) > 0:
    summary_df = merged_df[['question', 'pairwise_agreement', 'avg_confidence']].copy()
    summary_df = summary_df.sort_values('pairwise_agreement', ascending=False)
    summary_df['pairwise_agreement'] = summary_df['pairwise_agreement'].round(3)
    summary_df['avg_confidence'] = summary_df['avg_confidence'].round(3)
    summary_df.to_csv('agreement_confidence_summary.csv', index=False)
    print("\nSummary saved to agreement_confidence_summary.csv")
    
    print("\nTop 5 questions by agreement:")
    print(summary_df.head())
    
    print("\nBottom 5 questions by agreement:")
    print(summary_df.tail())