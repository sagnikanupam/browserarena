# V2 Inter-Annotator Agreement Analysis (Excluding Ties)

## Overview
This analysis examines inter-annotator agreement in the BrowserArena V2 dataset, specifically focusing on cases where both annotators made decisive votes (Agent 1 or Agent 2) and excluding any comparisons where either annotator voted "Tie" (vote = 3).

## Key Findings

### Overall Statistics
- **Total annotator pair comparisons**: 122,379
- **Comparisons with at least one tie vote**: 60,692 (49.6%)
- **Comparisons with only decisive votes**: 61,687 (50.4%)

### Agreement on Decisive Votes Only
- **Agreements**: 51,493
- **Disagreements**: 10,194
- **Agreement Rate**: **83.47%**

This high agreement rate on decisive votes suggests that when annotators are confident enough to pick a winner (rather than declaring a tie), they tend to agree on which agent performed better.

## Per-Question Analysis

### Questions with Highest Agreement (Decisive Votes Only)
1. **Q2**: 98.2% agreement (6,328 agreements out of 6,441 decisive comparisons)
2. **Q20**: 94.6% agreement (5,568 agreements out of 5,886 decisive comparisons)
3. **Q10**: 93.8% agreement (4,189 agreements out of 4,465 decisive comparisons)
4. **Q4**: 93.0% agreement (5,677 agreements out of 6,105 decisive comparisons)
5. **Q1**: 91.2% agreement (1,956 agreements out of 2,145 decisive comparisons)

### Questions with Lowest Agreement (Decisive Votes Only)
1. **Q22**: 49.3% agreement (604 agreements out of 1,225 decisive comparisons)
2. **Q5**: 52.3% agreement (276 agreements out of 528 decisive comparisons)
3. **Q12**: 52.6% agreement (618 agreements out of 1,176 decisive comparisons)
4. **Q7**: 63.4% agreement (685 agreements out of 1,081 decisive comparisons)
5. **Q11**: 64.4% agreement (406 agreements out of 630 decisive comparisons)

### Questions Most Likely to Have Decisive Votes
1. **Q4**: 94.8% of comparisons are decisive (6,105 out of 6,441)
2. **Q2**: 100% of comparisons are decisive (6,441 out of 6,441)
3. **Q20**: 91.4% of comparisons are decisive (5,886 out of 6,441)
4. **Q13**: 89.7% of comparisons are decisive (5,778 out of 6,441)
5. **Q8**: 80.0% of comparisons are decisive (5,151 out of 6,441)

### Questions Most Likely to Have Ties
1. **Q5**: Only 8.2% of comparisons are decisive (528 out of 6,441)
2. **Q11**: Only 9.8% of comparisons are decisive (630 out of 6,441)
3. **Q16**: Only 15.4% of comparisons are decisive (990 out of 6,441)
4. **Q17**: Only 16.1% of comparisons are decisive (1,035 out of 6,441)
5. **Q7**: Only 16.8% of comparisons are decisive (1,081 out of 6,441)

## Insights

1. **High Agreement on Clear Cases**: The 83.47% agreement rate on decisive votes indicates that annotators generally agree when there's a clear winner between agents.

2. **Question Difficulty Varies**: Some questions (like Q2, Q4, Q20) rarely result in ties and show high agreement, suggesting these tasks have clearer performance differences. Others (like Q5, Q11, Q16) frequently result in ties, indicating the agents perform similarly on these tasks.

3. **Tie Votes are Common**: Nearly 50% of all comparisons include at least one tie vote, suggesting that for many tasks, the performance difference between agents is not substantial enough for annotators to confidently pick a winner.

4. **Reliability of Decisive Votes**: The high agreement rate on decisive votes validates that when annotators do pick a winner, their judgments are reliable and consistent.

## Methodology
- Data source: BrowserArenaAgreementv2_July_18_2025_11.01.csv
- Analysis focused on questions Q1-Q22 (excluding Q14, Q15, Q21 which were not present)
- Inter-annotator agreement calculated by comparing all unique pairs of annotators for each question
- Ties (vote = 3) were excluded from the agreement calculation
- Agreement rate = (number of agreements) / (number of decisive comparisons)