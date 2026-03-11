# BrowserArena Evaluation Comparison Report V3

## Overview
This report compares three evaluation sources with **exact GPT-4o evaluations**:
1. **Original Baseline**: Single annotator judgments
2. **Human V2**: Crowdsourced human evaluations (114 annotators)
3. **GPT-4o (Exact)**: AI model evaluations with individual confidence scores

## Key Findings

### Inter-Annotator Reliability Metrics

#### Human V2 Internal Consistency
- **Krippendorff's Alpha**: 0.289
- **Average Pairwise Agreement**: 59.2%
- **Average Majority Agreement**: 71.4%

#### Three-Way Reliability
- **Krippendorff's Alpha (all three)**: 0.138 (poor agreement)
- **Three-way complete agreement**: 7/19 questions (36.8%)

### Pairwise Agreement Rates
| Comparison | Agreement | Cohen's Kappa | Change from V2 |
|------------|-----------|---------------|----------------|
| Baseline vs Human V2 | 12/19 (63.2%) | 0.269 | No change |
| Baseline vs GPT-4o | 12/19 (63.2%) | 0.199 | ↓ from 100% |
| Human V2 vs GPT-4o | 8/19 (42.1%) | 0.114 | ↓ from 63.2% |

### Distribution Comparison
| Source | Agent 1 | Agent 2 | Tie |
|--------|---------|---------|-----|
| Baseline | 17 (89.5%) | 1 (5.3%) | 1 (5.3%) |
| Human V2 | 10 (52.6%) | 1 (5.3%) | 8 (42.1%) |
| GPT-4o (Exact) | 12 (63.2%) | 6 (31.6%) | 1 (5.3%) |

### GPT-4o Performance with Exact Evaluations
- **Average confidence**: 82.9%
- **Confidence range**: 50.0% - 95.0%
- **Agreement with baseline**: 63.2% (down from 100% in V2)

### Questions with Complete Disagreement

**Q5**: The only question where all three evaluators chose differently
- Baseline: Agent 1
- Human V2: Tie (71.1% agreement among humans)
- GPT-4o: Agent 2 (85.0% confidence)
- Task: "visit the New York Times homepage and summarize the main story currently featured"

### GPT-4o Disagreement Patterns

#### Questions where GPT-4o disagreed with both Baseline and Human V2:
1. **Q4**: recipe task - GPT-4o chose Agent 2 (75% confidence)
2. **Q8**: boat rental info - GPT-4o chose Agent 2 (95% confidence)
3. **Q9**: Nintendo games list - GPT-4o chose Agent 2 (90% confidence)
4. **Q13**: Vitamin D symptoms - GPT-4o chose Agent 2 (75% confidence)
5. **Q22**: Buzzfeed articles - GPT-4o chose Agent 2 (80% confidence)

### GPT-4o Confidence Analysis by Agreement Pattern
- **All three agree**: 7 questions, avg confidence = 85.0%
- **GPT-4o agrees with baseline only**: 5 questions, avg confidence = 86.0%
- **GPT-4o agrees with human only**: 1 question, avg confidence = 50.0%
- **GPT-4o disagrees with both**: 6 questions, avg confidence = 83.3%

### Human Agreement vs GPT-4o Confidence Correlation
- **Pearson correlation**: 0.226 (weak positive)
- **R²**: 0.051
- **p-value**: 0.352 (not statistically significant)

## Key Insights

1. **GPT-4o Shows Independent Judgment**: With exact evaluations, GPT-4o disagreed with the baseline 37% of the time, showing it makes independent assessments rather than always agreeing with single annotator judgments.

2. **Humans Prefer Ties**: Human V2 annotators chose "Tie" in 42.1% of questions, compared to 5.3% for both baseline and GPT-4o, suggesting humans are more conservative in declaring clear winners.

3. **GPT-4o Favors Agent 2**: In disagreement cases, GPT-4o consistently chose Agent 2, possibly indicating different evaluation criteria or sensitivity to specific quality indicators.

4. **Confidence Doesn't Predict Agreement**: GPT-4o's confidence scores don't correlate with human agreement rates (R² = 0.051), suggesting the model's certainty doesn't align with human consensus difficulty.

5. **Low Three-Way Agreement**: Only 36.8% of questions had complete agreement among all three sources, highlighting the subjective nature of browser task evaluation.

## Questions with Lowest Human Agreement
1. **Q22**: 40.7% - "What are today's top 10 most popular articles on buzzfeed?"
2. **Q12**: 41.9% - "Check all ticket touting websites and find the cheapest ticket"
3. **Q18**: 43.0% - "Compare prices of HP laptop model across different online retailers"
4. **Q3**: 44.8% - "Go to Etsy.com, browse the trending home décor category"
5. **Q7**: 45.0% - "What are the today's top headlines from Fox News?"

## Questions with Highest Human Agreement
1. **Q2**: 98.2% - "Create a summary for the wikipedia article about Manchester Metrolink"
2. **Q4**: 88.2% - "find me any recipe on food.com"
3. **Q20**: 86.6% - "Give me a summary of this week's best-selling crypto coins"
4. **Q8**: 72.1% - "Please provide a list of the types of boats that can be rented"
5. **Q13**: 70.8% - "List out the symptoms of Vitamin D deficiency"

## Methodology Notes
- Human V2 data: 114 crowdsourced annotators per question
- GPT-4o evaluations: Run on exact task data from survey HTML snippets
- Statistical significance: α = 0.05 for all tests
- Inter-rater reliability interpreted using standard guidelines

## Conclusion
The exact GPT-4o evaluations reveal significant differences from the preliminary analysis. While GPT-4o shows reasonable agreement with human consensus on clear-cut cases, it demonstrates independent judgment that doesn't always align with either single-annotator baselines or crowd consensus. The weak correlation between GPT-4o confidence and human agreement suggests that model certainty operates on different dimensions than human consensus difficulty.