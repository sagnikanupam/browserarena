# O4-mini vs GPT-4o Evaluation Comparison

## Summary

This report compares the performance of o4-mini and GPT-4o models in evaluating BrowserArena agent interactions using both execution traces and GIF recordings.

## Key Findings

### 1. Baseline Agreement
- **GPT-4o**: 68.4% (13/19) - Higher agreement with human baseline
- **o4-mini**: 57.9% (11/19) - Moderate agreement with baseline

GPT-4o shows better alignment with human judgments by ~10.5 percentage points.

### 2. Confidence Scores
- **GPT-4o**: Average confidence 0.853
- **o4-mini**: Average confidence 0.799

GPT-4o exhibits higher confidence in its evaluations overall.

### 3. Inter-Model Agreement
- Models agree on 68.4% (13/19) of cases
- 6 disagreements, primarily where o4-mini chose "Tie" while GPT-4o made a decisive choice

### 4. Preference Distribution

**o4-mini**:
- Agent 1: 52.6%
- Tie: 42.1% (notably high)
- Agent 2: 5.3%

**GPT-4o**:
- Agent 1: 63.2%
- Agent 2: 26.3%
- Tie: 10.5%

O4-mini shows a strong tendency toward "Tie" judgments (42.1% vs GPT-4o's 10.5%), suggesting it may be more conservative or less decisive in its evaluations.

## Disagreement Analysis

All 6 disagreements follow a pattern where o4-mini chose "Tie" while GPT-4o made a decisive choice:

1. **Q6** (BBC News headlines): o4-mini saw both agents fail equally, GPT-4o credited Agent 1's attempt
2. **Q8** (Boat rentals): o4-mini called tie, GPT-4o preferred Agent 2's extracted information
3. **Q9** (Nintendo games): o4-mini saw parsing errors on both sides, GPT-4o credited Agent 2's progress
4. **Q18** (HP laptop prices): o4-mini called tie, GPT-4o slightly preferred Agent 1
5. **Q19** (Samsung TV prices): o4-mini called tie, GPT-4o agreed with baseline (Agent 2)
6. **Q22** (BuzzFeed articles): Both baseline and o4-mini said tie, GPT-4o chose Agent 2

## Notable Patterns

1. **O4-mini's "Failed to parse model response"**: In 2 cases (Q6, Q9), o4-mini returned minimal reasoning, suggesting potential technical issues with response parsing.

2. **Conservative vs Decisive**: O4-mini appears more conservative, often calling ties when agents encounter similar types of failures. GPT-4o tends to find distinctions even in failure cases.

3. **Confidence Correlation**: When o4-mini chooses "Tie", it often has lower confidence (0.50-0.85), while GPT-4o maintains high confidence (0.80-0.90) even in these borderline cases.

## Recommendations

1. **For high-stakes evaluations**: GPT-4o may be preferred due to higher baseline agreement and more decisive judgments
2. **For conservative evaluations**: o4-mini's tendency to call ties might be valuable when avoiding false positives is important
3. **Response parsing**: O4-mini may need improvements in response generation/parsing to avoid the "Failed to parse" issues

## Technical Notes

Both models successfully:
- Processed GIF recordings (2 GIFs per evaluation)
- Utilized vLLM execution traces
- Completed all 19 evaluations

The evaluation used identical prompts and data for both models, ensuring a fair comparison.