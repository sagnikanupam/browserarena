# BrowserArena Survey Agreement Analysis Report

## Overview
This report analyzes the inter-annotator agreement for the BrowserArena task evaluations. We collected responses from multiple annotators (average 26.8 per question) for 19 questions, comparing their judgments to the original baseline labels.

## Key Findings

### 1. Overall Agreement Statistics
- **Average Pairwise Agreement**: 51.6%
- **Average Majority Agreement**: 65.0%
- **Agreement with Original Baseline**: 63.2% (12/19 questions)

### 2. Inter-Annotator Reliability Metrics
- **Krippendorff's Alpha**: 0.193
  - This indicates "slight agreement" according to standard interpretation scales
  - Values < 0.20 are typically considered poor agreement
- **Cohen's Kappa** (original vs majority): 0.269
  - This indicates "fair agreement" between original and majority labels

### 3. Vote Distribution
- **Agent 1**: Selected in majority for 10 questions
- **Agent 2**: Selected in majority for 1 question
- **Tie**: Selected in majority for 8 questions

### 4. Questions with Disagreement from Baseline
Seven questions showed disagreement between the original label and majority vote:
- Q5: Original=Agent 1, Majority=Tie (45.8% agreement)
- Q7: Original=Agent 1, Majority=Tie (51.9% agreement)
- Q11: Original=Agent 1, Majority=Tie (75.0% agreement)
- Q12: Original=Agent 1, Majority=Tie (50.0% agreement)
- Q16: Original=Agent 1, Majority=Tie (54.2% agreement)
- Q17: Original=Agent 1, Majority=Tie (50.0% agreement)
- Q18: Original=Agent 1, Majority=Tie (40.0% agreement)

Notable pattern: All disagreements involved the original label being "Agent 1" but the majority choosing "Tie".

### 5. Questions with Highest Agreement
- Q20: 90.0% agreement (Agent 1)
- Q2: 88.5% agreement (Agent 1)
- Q13: 86.1% agreement (Agent 1)
- Q4: 83.9% agreement (Agent 1)
- Q6: 80.8% agreement (Agent 1)

### 6. Questions with Lowest Agreement
- Q18: 40.0% agreement (Tie)
- Q5: 45.8% agreement (Tie)
- Q12: 50.0% agreement (Tie)
- Q17: 50.0% agreement (Tie)

## Interpretation

### Agreement Patterns
1. **Strong bias toward Agent 1**: The original baseline heavily favored Agent 1 (17/19 questions), while annotators were more likely to see ties.

2. **Tie responses are common**: Annotators frequently chose "Tie" even when the original label was decisive, suggesting either:
   - The task completions were genuinely similar in quality
   - Annotators had difficulty distinguishing between agents
   - Annotators were more conservative in their judgments

3. **Low inter-rater reliability**: The Krippendorff's alpha of 0.193 suggests substantial disagreement among annotators, which is common in subjective evaluation tasks.

### Recommendations
1. **Clearer evaluation criteria**: Provide more specific guidelines for when to choose Agent 1, Agent 2, or Tie
2. **Training examples**: Show annotators examples of clear wins vs ties
3. **Task-specific rubrics**: Different task types may need different evaluation criteria
4. **Multiple annotations**: Continue collecting multiple annotations per question to identify consensus

## Statistical Summary
- Total questions analyzed: 19
- Total votes collected: 509
- Average responses per question: 26.8
- Questions where majority agrees with baseline: 12/19 (63.2%)

## Files Generated
1. `agreement_analysis_results.csv` - Detailed results for each question
2. `agreement_rates_by_question.png` - Bar chart of agreement rates
3. `response_distribution.png` - Pie chart of overall vote distribution
4. `confusion_matrix_heatmap.png` - Visual confusion matrix
5. `vote_distribution_by_question.png` - Stacked bar chart of votes per question