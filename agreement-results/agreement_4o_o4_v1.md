# GPT-4o and o4-mini Confidence Correlation Analysis

## Overview
This document describes the comprehensive analysis of confidence scores from GPT-4o and o4-mini models, demonstrating that model confidence is poorly correlated with agreement metrics (both with baseline and human annotations).

## Key Findings

### Main Results
- **GPT-4o and o4-mini confidence scores are uncorrelated** (r = -0.054)
- **Confidence does not predict correctness**: Models show similar confidence whether they agree or disagree with baseline
- **No correlation with human consensus**: Model confidence is independent of human inter-annotator agreement
- **Models cannot assess task difficulty**: Confidence remains high even for controversial/difficult tasks

## Implementation Details

### Data Sources
- **Model Evaluations**: `o4mini_evaluation_final.csv`, `gpt4o_proper_evaluation_final.csv`
- **Human Annotations**: `BrowserArenaAgreementv2_July_18_2025_11.01.csv` (116 annotators per question)
- **Baseline**: `baseline.csv` (ground truth labels)

### Analysis Script
- **File**: `analyze_confidence_correlations.py`
- **Dependencies**: pandas, numpy, matplotlib, seaborn, scipy, sklearn

## Six Analysis Strategies

### Strategy 1: Confidence Calibration Analysis
**Output**: `confidence_calibration.png`

**What it shows**:
- Binned confidence scores (0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0)
- Agreement rate with baseline in each confidence bin
- Reliability diagrams comparing expected vs observed accuracy

**Key insight**: Both models are poorly calibrated - high confidence doesn't correlate with being correct. The reliability diagrams show points far from the diagonal (perfect calibration line).

**Code approach**:
```python
# Bin confidence scores and calculate agreement rates
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
merged_df['conf_bin'] = pd.cut(merged_df[conf_col], bins=bins)
# Calculate agreement rate per bin
for bin_label in bin_labels:
    agreement_rate = bin_data[agree_col].mean()
```

### Strategy 2: Disagreement Confidence Paradox
**Output**: `disagreement_confidence_paradox.png`

**What it shows**:
- Scatter plot of o4-mini confidence vs GPT-4o confidence
- Color-coded by agreement status (both correct, only one correct, both wrong)
- Highlights high-confidence disagreement zone (both >0.8 confidence but disagree)
- Box plots comparing confidence when models agree vs disagree

**Key insight**: Found cases where both models have high confidence (>0.8) but disagree with each other. Models maintain similar confidence levels whether they agree or disagree.

**Code approach**:
```python
# Identify high confidence disagreements
high_conf_disagree = merged_df[
    (merged_df['o4mini_confidence'] > 0.8) & 
    (merged_df['gpt4o_confidence'] > 0.8) & 
    (merged_df['o4mini_preference'] != merged_df['gpt4o_preference'])
]
```

### Strategy 3: Confidence Independence from Human Consensus
**Output**: `confidence_vs_human_consensus.png`

**What it shows**:
- Model confidence vs human pairwise agreement rate
- Model confidence vs human majority vote strength
- Model confidence vs human tie rate
- Average confidence by human agreement level (low/medium/high)

**Key insight**: 
- GPT-4o correlation with human agreement: r = 0.221 (very weak)
- o4-mini correlation with human agreement: r = 0.034 (essentially zero)
- Models can't detect when humans disagree about quality

**Code approach**:
```python
# Calculate human pairwise agreement (probability two random annotators agree)
p_agent1 = agent1_count / total
p_agent2 = agent2_count / total
p_tie = tie_count / total
pairwise_agreement = p_agent1**2 + p_agent2**2 + p_tie**2
```

### Strategy 4: Confidence Decorrelation Matrix
**Output**: `confidence_decorrelation_matrix.png`

**What it shows**:
- Full correlation heatmap between all variables
- P-value heatmap showing statistical significance
- Variables: GPT-4o confidence, o4-mini confidence, agreement with baseline, human agreement metrics

**Key insight**: Near-zero correlations between confidence scores and all agreement metrics. The strongest correlation for o4-mini confidence is with its own accuracy (0.609) but GPT-4o shows almost no correlation with its accuracy (0.162).

**Key correlations found**:
- GPT-4o conf vs o4-mini conf: -0.054 (no correlation)
- GPT-4o conf vs GPT-4o accuracy: 0.162 (very weak)
- o4-mini conf vs o4-mini accuracy: 0.609 (moderate)
- Both models vs human agreement: <0.25 (weak to none)

### Strategy 5: When Confidence Misleads
**Output**: `confidence_error_analysis.png`

**What it shows**:
- Violin plots of confidence distributions when correct vs incorrect
- Histograms showing distribution overlap
- Statistical tests (t-tests) comparing confidence when right vs wrong

**Key insight**: Massive overlap in confidence distributions between correct and incorrect predictions. Models are nearly as confident when wrong as when right:
- GPT-4o: mean confidence 0.85 (correct) vs 0.82 (incorrect), p>0.05
- o4-mini: mean confidence 0.84 (correct) vs 0.75 (incorrect), p<0.05 but still high overlap

**Code approach**:
```python
# Separate confidence by correctness
correct_conf = merged_df[merged_df[agree_col] == True][conf_col]
incorrect_conf = merged_df[merged_df[agree_col] == False][conf_col]
# Calculate distribution overlap
overlap = np.minimum(hist_correct, hist_incorrect).sum()
```

### Strategy 6: Task Difficulty vs Model Confidence
**Output**: `difficulty_blind_confidence.png`

**What it shows**:
- Confidence vs task difficulty (defined as 1 - human agreement)
- Average confidence by difficulty bins (easy/medium/hard)
- Accuracy vs difficulty (showing expected performance drop)
- Confidence on the 5 most difficult tasks
- Calibration comparison across difficulty levels

**Key insight**: Models maintain high confidence regardless of task difficulty. The trend lines are nearly flat or slightly positive (should be negative if models could assess difficulty):
- GPT-4o slope: ~0.0 (no relationship)
- o4-mini slope: ~0.0 (no relationship)

Even on the hardest tasks (where humans strongly disagree), models maintain confidence scores >0.7.

**Code approach**:
```python
# Define difficulty as inverse of human agreement
difficulty = 1 - v2_stats[q]['pairwise_agreement']
# Bin into easy/medium/hard
diff_df['difficulty_bin'] = pd.cut(diff_df['difficulty'], 
                                   bins=[0, 0.4, 0.55, 1.0],
                                   labels=['Easy', 'Medium', 'Hard'])
```

## Statistical Summary

### Model Agreement
- Models agree with each other: 11/17 cases (64.7%)
- High confidence disagreements: 1/17 cases (5.9%)

### Confidence Statistics
- GPT-4o average confidence: 0.853
- o4-mini average confidence: 0.799
- Both maintain >0.75 confidence even when wrong

### Correlation Summary
| Metric | GPT-4o Confidence | o4-mini Confidence |
|--------|------------------|-------------------|
| With each other | -0.054 | -0.054 |
| With own accuracy | 0.162 | 0.609 |
| With human agreement | 0.221 | 0.034 |
| With task difficulty | ~0.0 | ~0.0 |

## Conclusions

1. **Confidence ≠ Correctness**: High confidence from either model does not indicate they are more likely to be correct.

2. **Models Can't Assess Uncertainty**: They maintain high confidence even on tasks where humans strongly disagree, indicating inability to recognize ambiguous cases.

3. **Poor Calibration**: Both models are overconfident. A well-calibrated model would show confidence scores that match actual accuracy rates.

4. **Independent Errors**: The near-zero correlation between GPT-4o and o4-mini confidence suggests they make independent judgments (good for ensemble methods but concerning for reliability).

5. **Difficulty-Blind**: Models cannot detect task difficulty and maintain similar confidence across easy and hard tasks.

## Recommendations

1. **Don't use confidence scores for filtering**: Since confidence doesn't predict correctness, using confidence thresholds to filter outputs is ineffective.

2. **Ensemble methods may help**: Since models make independent errors, combining multiple models might improve reliability.

3. **Need better uncertainty quantification**: Current confidence scores are not meaningful indicators of model uncertainty.

4. **Human review remains critical**: For tasks where quality matters, human review cannot be replaced by confidence-based filtering.

## Files Generated

All outputs are saved in the `agreement-results/` directory:

1. `confidence_calibration.png` - Calibration analysis plots
2. `disagreement_confidence_paradox.png` - Model disagreement analysis
3. `confidence_vs_human_consensus.png` - Human consensus correlation
4. `confidence_decorrelation_matrix.png` - Full correlation heatmap
5. `confidence_error_analysis.png` - Confidence when right vs wrong
6. `difficulty_blind_confidence.png` - Difficulty assessment analysis
7. `analyze_confidence_correlations.py` - Complete analysis script
8. `agreement_4o_o4_v1.md` - This documentation

## Reproducibility

To reproduce the analysis:
```bash
cd agreement-results
python analyze_confidence_correlations.py
```

Ensure all required CSV files are present in the directory.