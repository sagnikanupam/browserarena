# BrowserArena Evaluation Final Report: GPT-4o vs Gemini
## Comprehensive Analysis of Browser Automation Agent Evaluation

## Overview
This report presents a comprehensive comparison of browser automation agent evaluations across four sources:
- **Baseline**: Original evaluations
- **Human V2**: Crowdsourced evaluations (114 annotators per question)
- **GPT-4o**: OpenAI's model with vision capabilities (evaluated with GIFs and traces)
- **Gemini**: Google's Gemini-1.5-Flash (evaluated with text-only due to API limitations)

## Key Findings

### Agreement Statistics
| Comparison | Agreement Rate | Cohen's Kappa |
|------------|----------------|---------------|
| Four-way agreement | 36.8% (7/19) | - |
| Baseline vs Human V2 | 63.2% (12/19) | 0.269 |
| Baseline vs GPT-4o | 68.4% (13/19) | 0.240 |
| Baseline vs Gemini | 68.4% (13/19) | 0.149 |
| Human V2 vs GPT-4o | 47.4% (9/19) | 0.136 |
| Human V2 vs Gemini | 42.1% (8/19) | 0.041 |
| **GPT-4o vs Gemini** | **84.2% (16/19)** | **0.680** |

### Model Characteristics
| Model | Avg Confidence | Std Dev | Correlation with Human Agreement | Input Data |
|-------|----------------|---------|----------------------------------|------------|
| GPT-4o | 85.3% | 7.7% | r = 0.342 (p = 0.152) | GIFs + Traces |
| Gemini | 89.2% | 11.3% | r = 0.222 (p = 0.360) | Text-only |

### Vote Distribution
| Source | Agent 1 | Agent 2 | Tie |
|--------|---------|---------|-----|
| Baseline | 17 (89.5%) | 1 (5.3%) | 1 (5.3%) |
| Human V2 | 10 (52.6%) | 1 (5.3%) | 8 (42.1%) |
| GPT-4o | 13 (68.4%) | 4 (21.1%) | 2 (10.5%) |
| Gemini | 14 (73.7%) | 4 (21.1%) | 1 (5.3%) |

## Key Insights

### 1. High Agreement Between AI Models
- GPT-4o and Gemini agree on 84.2% of cases (Cohen's κ = 0.680)
- Only 3 disagreements despite different input modalities:
  - Q3 (Etsy): GPT-4o chose Tie, Gemini chose Agent 1
  - Q11 (YouTube): GPT-4o chose Tie, Gemini chose Agent 2
  - Q19 (Samsung TV): GPT-4o chose Agent 2, Gemini chose Tie

### 2. Both AI Models Match Baseline Equally Well
- Both achieve 68.4% agreement with baseline
- GPT-4o had access to visual evidence (GIFs)
- Gemini used only text traces
- Suggests text traces contain sufficient information for evaluation

### 3. AI vs Human Disagreement Pattern
- AI models agree with humans ~45% of the time
- AI models agree with each other 84.2% of the time
- Suggests AI models apply consistent but different criteria than humans

### 4. Confidence Patterns
- Gemini shows higher average confidence (89.2%) despite text-only input
- GPT-4o more conservative (85.3%) even with visual evidence
- Neither model's confidence predicts human agreement difficulty well

### 5. Tie Preferences
- Humans strongly prefer Tie judgments (42.1%)
- Gemini rarely chooses Tie (5.3%)
- GPT-4o falls in between (10.5%)
- Baseline heavily biased toward Agent 1 (89.5%)

## Questions with Perfect Agreement
Seven questions achieved unanimous agreement across all four evaluators:
- Q1, Q2, Q4, Q6, Q10, Q13, Q20
- All chose Agent 1, suggesting clear performance differences

## Model Disagreements (GPT-4o vs Gemini)
1. **Q3** (Etsy home décor):
   - GPT-4o: Tie (60% confidence) - lowest confidence score
   - Gemini: Agent 1 (80% confidence)

2. **Q11** (YouTube homepage):
   - GPT-4o: Tie (80% confidence)
   - Gemini: Agent 2 (90% confidence)

3. **Q19** (Samsung TV prices):
   - GPT-4o: Agent 2 (80% confidence)
   - Gemini: Tie (50% confidence) - JSON parsing error

## Methodology Notes
1. **GPT-4o Evaluation**:
   - Used full vLLM execution traces
   - Included GIF recordings of browser interactions
   - 10,000 max tokens for detailed responses
   - Temperature = 0.1 for consistency

2. **Gemini Evaluation**:
   - Text-only evaluation (GIFs caused API errors)
   - Used Gemini-1.5-Flash model
   - Direct API calls with same evaluation prompt
   - Temperature = 0.1 for consistency

## Conclusions

1. **Visual Evidence May Not Be Essential**: Gemini achieved identical baseline agreement (68.4%) using only text traces, suggesting execution traces contain sufficient information for evaluation.

2. **AI Models Consistent with Each Other**: The high agreement between GPT-4o and Gemini (84.2%) indicates AI models apply similar evaluation criteria, regardless of input modality.

3. **Human-AI Evaluation Gap Persists**: The ~45% agreement between humans and AI models suggests fundamental differences in how task success is judged.

4. **Confidence Doesn't Predict Difficulty**: Neither model's confidence scores correlate well with human consensus difficulty, indicating these metrics capture different aspects of evaluation certainty.

5. **Evaluation Remains Subjective**: The low four-way agreement (36.8%) confirms that browser automation evaluation is inherently subjective, even with complete information.

## Recommendations
1. Consider text-only evaluation as a viable, cost-effective alternative
2. Use multiple AI evaluators for consistency checking
3. Recognize that AI and human evaluators may legitimately reach different conclusions
4. Focus on unanimous agreement cases for establishing ground truth
5. Investigate the source of baseline's Agent 1 bias (89.5% preference)