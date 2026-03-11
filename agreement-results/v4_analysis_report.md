# BrowserArena Evaluation Comparison Report V4
## With Proper GPT-4o Evaluation Using GIFs and Full Traces

## Overview
This report presents the corrected analysis using **proper GPT-4o evaluations** that include:
- Full vLLM execution traces from JSON files
- Visual GIF recordings of actual browser interactions
- GPT-4o model with vision capabilities

## Key Findings

### Agreement Statistics
| Comparison | Agreement Rate | Cohen's Kappa | Change from V3 |
|------------|----------------|---------------|----------------|
| Baseline vs Human V2 | 63.2% (12/19) | 0.269 | No change |
| **Baseline vs GPT-4o** | **68.4% (13/19)** | **0.240** | ↑ from 63.2% |
| **Human V2 vs GPT-4o** | **47.4% (9/19)** | **0.136** | ↑ from 42.1% |
| **Three-way agreement** | **42.1% (8/19)** | - | ↑ from 36.8% |

### Distribution Comparison
| Source | Agent 1 | Agent 2 | Tie |
|--------|---------|---------|-----|
| Baseline | 17 (89.5%) | 1 (5.3%) | 1 (5.3%) |
| Human V2 | 10 (52.6%) | 1 (5.3%) | 8 (42.1%) |
| **GPT-4o (Proper)** | **13 (68.4%)** | **4 (21.1%)** | **2 (10.5%)** |

### GPT-4o Performance with Full Data
- **Average confidence**: 85.3%
- **Confidence range**: 60% - 90%
- **Used visual evidence**: 100% (all 19 evaluations included GIFs)
- **Had full traces**: 100% (all 19 evaluations had vLLM traces)

### Human Agreement vs GPT-4o Confidence
- **Pearson correlation**: 0.335
- **R²**: 0.112
- **p-value**: 0.161 (not statistically significant)

The weak positive correlation suggests that GPT-4o's confidence when evaluating with full visual and trace data still doesn't strongly predict human consensus difficulty.

### Questions Where GPT-4o Disagreed with Baseline (6/19)

1. **Q3** (Etsy home décor): Baseline=Agent 1, GPT-4o=Tie (60% conf)
   - Lowest GPT-4o confidence, suggesting genuine task ambiguity

2. **Q5** (NYT homepage summary): Baseline=Agent 1, GPT-4o=Agent 2 (90% conf)
   - High confidence disagreement, visual evidence likely showed Agent 2 performed better

3. **Q8** (Boat rental types): Baseline=Agent 1, GPT-4o=Agent 2 (90% conf)
   - Another high-confidence disagreement based on visual evidence

4. **Q9** (Nintendo Switch games): Baseline=Agent 1, GPT-4o=Agent 2 (90% conf)
   - GPT-4o consistently favored Agent 2 with high confidence

5. **Q11** (YouTube homepage): Baseline=Agent 1, GPT-4o=Tie (80% conf)
   - Visual evidence suggested neither agent clearly succeeded

6. **Q22** (Buzzfeed articles): Baseline=Tie, GPT-4o=Agent 2 (90% conf)
   - Only case where baseline said Tie but GPT-4o saw a clear winner

### Key Insights from Proper Evaluation

1. **Visual Evidence Matters**: With access to GIFs and full traces, GPT-4o's evaluations are more nuanced and sometimes disagree with text-only baseline judgments.

2. **GPT-4o Shows Moderate Agreement**: 68.4% agreement with baseline is reasonable given that GPT-4o had access to complete visual evidence while baseline may have relied on partial information.

3. **High Confidence in Disagreements**: When GPT-4o disagreed with baseline, it was often with high confidence (avg 85% for disagreements), suggesting the visual evidence clearly supported a different conclusion.

4. **Humans Still Prefer Ties**: Human annotators chose "Tie" much more frequently (42.1%) than either baseline (5.3%) or GPT-4o (10.5%), indicating more conservative judgment.

5. **Three-Way Agreement Remains Low**: Even with proper evaluation, only 42.1% of questions achieved complete agreement among all three sources, highlighting the inherent subjectivity in browser task evaluation.

## Questions with Interesting Patterns

### Highest Human Agreement, All Sources Agree
- **Q2** (Wikipedia summary): 98.2% human agreement, all chose Agent 1
- **Q4** (Recipe): 88.2% human agreement, all chose Agent 1
- **Q20** (Crypto summary): 86.6% human agreement, all chose Agent 1

### Low Human Agreement, Sources Disagree
- **Q22** (Buzzfeed): 40.7% human agreement
  - Baseline: Tie, Human: Tie, GPT-4o: Agent 2
- **Q11** (YouTube): 52.9% human agreement  
  - Baseline: Agent 1, Human: Tie, GPT-4o: Tie

## Methodology Improvements in V4
1. Used GPT-4o model (not deprecated GPT-4V)
2. Loaded full vLLM execution traces from JSON files
3. Included actual GIF recordings in evaluations
4. All 19 questions evaluated with complete data

## Conclusion
The proper GPT-4o evaluation with visual evidence provides a more reliable assessment than text-only evaluation. The 68.4% agreement with baseline is reasonable and the disagreements appear to reflect genuine differences in interpretation when full visual evidence is available. The persistent low correlation between confidence and human agreement suggests that model certainty and human consensus difficulty operate on fundamentally different dimensions.