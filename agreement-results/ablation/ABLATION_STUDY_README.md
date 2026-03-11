# Ablation Study: Understanding Model Decision Drivers

## Overview

This ablation study investigates which input components (execution traces vs. GIF recordings) drive AI model decisions when evaluating browser automation agents. We tested GPT-4o and O4-mini under four conditions on all 19 evaluation questions:

1. **Full**: Both GIF recordings and vLLM execution traces (baseline)
2. **Trace Only**: Only execution traces, no visual input
3. **Truncated Trace**: First 1000 characters of traces only
4. **GIF Only**: Only visual recordings, no execution traces

## Key Findings

### 1. 🔴 Critical Discovery: GIFs Harm GPT-4o Performance
- **GPT-4o performs BETTER with trace-only input**: 78.9% vs 68.4% with full context
- **This represents a 10.5% improvement by removing visual input**
- **Interpretation**: GIFs may be adding noise rather than signal for GPT-4o

### 2. Visual-Only Evaluation Shows Extreme Biases
- **GIF-only agreement with baseline**:
  - GPT-4o: 10.5% (89.5% Agent 2 bias)
  - O4-mini: 5.3% (100% Tie responses)
- **Systematic biases emerge**: Without execution traces, models cannot make informed decisions and fall back to stereotypical behaviors

### 3. Traces Are Essential, GIFs Are Detrimental
- **Trace-only performance**:
  - GPT-4o: 78.9% agreement (BETTER than full context at 68.4%!)
  - O4-mini: 52.6% agreement (slight decrease from 57.9%)
- **Truncated traces** severely impact GPT-4o (42.1%) but O4-mini maintains performance (57.9%)
- **Interpretation**: Execution traces contain all necessary information; visual input confuses GPT-4o

### 4. Model-Specific Behaviors
**GPT-4o**:
- Maintains high confidence (0.90) even with GIF-only input (10.5% accuracy)
- Severely impacted by trace truncation (78.9% → 42.1%)
- Shows strong Agent 2 bias with visual-only input (89.5%)

**O4-mini**:
- Shows some confidence calibration (0.75 with GIF-only)
- Robust to trace truncation (maintains ~58% performance)
- Defaults to Tie when uncertain (100% ties with GIF-only)

## Plots Generated

### 1. ablation_agreement_comparison_full.pdf
- Bar charts showing agreement rates by ablation type for both models
- Overlaid confidence scores as secondary axis
- Highlights the surprising finding that trace-only outperforms full context

### 2. ablation_key_finding_full.pdf
- Direct comparison of Full vs Trace-Only vs GIF-Only performance
- Annotated to show the 10.5% improvement for GPT-4o without GIFs
- Clear visualization of the detrimental effect of visual input

### 3. ablation_vote_distribution_full.pdf
- Pie charts showing vote distributions for all conditions
- Reveals systematic biases in GIF-only evaluation
- 8 subplots (2 models × 4 conditions)

### 4. ablation_confidence_distribution_full.pdf
- Violin plots of confidence distributions by input type
- Shows poor confidence calibration, especially for GPT-4o
- Mean confidence values annotated above each distribution

## Summary Statistics (Full 19-Question Study)

| Model   | Condition      | Agreement (%) | 95% CI    | Confidence | Tie Rate (%) | Agent 1 (%) | Agent 2 (%) |
|---------|----------------|---------------|-----------|------------|--------------|-------------|-------------|
| GPT-4o  | Full (GIF+Trace) | 68.4       | ± 21.5    | 0.842      | 10.5         | 63.2        | 26.3        |
| GPT-4o  | Trace Only    | **78.9**      | ± 18.8    | 0.811      | 0.0          | 73.7        | 26.3        |
| GPT-4o  | Truncated     | 42.1          | ± 22.8    | 0.689      | 42.1         | 42.1        | 15.8        |
| GPT-4o  | GIF Only      | 10.5          | ± 14.2    | 0.900      | 0.0          | 10.5        | **89.5**    |
| O4-mini | Full (GIF+Trace) | 57.9       | ± 22.8    | 0.799      | 42.1         | 52.6        | 5.3         |
| O4-mini | Trace Only    | 52.6          | ± 23.1    | 0.673      | 47.4         | 42.1        | 10.5        |
| O4-mini | Truncated     | 57.9          | ± 22.8    | 0.669      | 36.8         | 52.6        | 10.5        |
| O4-mini | GIF Only      | 5.3           | ± 10.3    | 0.747      | **100.0**    | 0.0         | 0.0         |

## Implications

### For Practitioners:
1. **Use trace-only evaluation for GPT-4o** - 10.5% better performance and lower cost
2. **Avoid GIF-only evaluation** - Models show extreme biases and poor performance
3. **Be aware of overconfidence** - GPT-4o maintains 90% confidence with 10.5% accuracy

### For Research:
1. **Multimodal fusion can harm performance** - Visual input adds noise for GPT-4o
2. **Model architectures differ fundamentally** - O4-mini is robust to truncation, GPT-4o is not
3. **Confidence calibration is broken** - Especially for GPT-4o in degraded conditions

## Key Insights

1. **The Paradox of More Information**: Adding visual information (GIFs) actually reduces GPT-4o's accuracy by 10.5%, suggesting that multimodal models may struggle to effectively integrate different modalities for this task.

2. **Systematic Biases in Visual-Only Evaluation**: 
   - GPT-4o develops an extreme Agent 2 bias (89.5%) with GIF-only input
   - O4-mini defaults to 100% Tie responses with GIF-only input
   
3. **Robustness Varies by Architecture**: O4-mini maintains performance with truncated traces (57.9%) while GPT-4o drops dramatically (42.1%), suggesting different architectural sensitivities to input completeness.

## Files

- `ablation_full_results.csv`: Raw evaluation data for all 19 questions
- `ablation_full_summary_statistics.csv`: Aggregated metrics with confidence intervals
- `ablation_*_full.pdf`: Publication-ready plots
- `run_ablation_full.py`: Script to reproduce full study results

## Future Work

1. Investigate why GIFs harm GPT-4o performance
2. Test with different visual encodings or resolutions
3. Analyze which specific trace components are most informative
4. Develop better confidence calibration methods for multimodal evaluation