# GPT-4o and o4-mini Confidence Correlation Analysis

## Overview
This document describes the comprehensive analysis of model confidence scores and their (lack of) correlation with agreement rates. The analysis demonstrates that both GPT-4o and o4-mini confidence scores are largely uncorrelated with whether they correctly agree with the baseline evaluation or V2 human majority consensus.

## Key Findings
- **GPT-4o confidence** shows weak positive correlation with agreement (r = 0.342, p = 0.152)
- **o4-mini confidence** shows virtually no correlation with agreement (r = -0.041, p = 0.876)
- **Cross-model confidence correlation** is negligible (r = -0.041, p = 0.876)
- Models maintain high confidence even when disagreeing with ground truth
- Neither model calibrates confidence to human consensus difficulty

## Script Information
**Script:** `create_confidence_correlation_figures.py`
**Dependencies:** pandas, numpy, matplotlib, seaborn, scipy, sklearn

## Figure Descriptions

### Figure 1: Confidence vs Agreement Accuracy Scatter Plot
**Files:** 
- `confidence_vs_agreement_scatter.png`
- `confidence_vs_agreement_scatter.pdf`

**Description:** Dual-panel scatter plot showing the relationship between model confidence (x-axis) and binary agreement with baseline (y-axis). Points are colored by vote type (Agent 1/2/Tie) with jitter added for visibility. Regression lines and correlation coefficients are displayed.

**Key Insight:** Both models show poor correlation between confidence and correctness. GPT-4o has a weak positive correlation (r=0.342) while o4-mini shows essentially no correlation.

**Code Implementation:**
- Loads evaluation data for both models
- Calculates agreement with baseline
- Creates scatter plots with vote type coloring
- Adds regression lines and correlation statistics
- Uses jitter to prevent overlapping points

### Figure 2: Confidence Calibration Curve
**Files:**
- `confidence_calibration_curve.png`
- `confidence_calibration_curve.pdf`

**Description:** Shows expected accuracy (mean confidence in bin) vs actual accuracy (agreement rate in bin) for both models. Perfect calibration would follow the diagonal line. Includes histogram of confidence distribution below.

**Key Insight:** Both models are poorly calibrated. They tend to be overconfident, with actual accuracy lower than expected based on confidence scores. Expected Calibration Error (ECE) quantifies this miscalibration.

**Code Implementation:**
- Bins confidence scores into deciles
- Calculates actual agreement rate per bin
- Plots calibration curves against perfect calibration line
- Computes Expected Calibration Error (ECE)
- Adds confidence distribution histogram

### Figure 3: Disagreement Pattern Heatmap
**Files:**
- `disagreement_pattern_heatmap.png`
- `disagreement_pattern_heatmap.pdf`

**Description:** Heatmap showing confidence scores and agreement patterns for each question. Rows are questions (sorted by human consensus strength), columns show GPT-4o confidence, o4-mini confidence, and whether each agrees with baseline, plus human consensus strength.

**Key Insight:** No clear pattern emerges between human consensus difficulty and model confidence. Models maintain high confidence regardless of task difficulty or agreement patterns.

**Code Implementation:**
- Creates matrix of confidence and agreement data
- Sorts questions by human consensus strength
- Uses color coding to show values (0-1 scale)
- Annotates cells with exact values

### Figure 4: Confidence Distribution by Agreement Outcome
**Files:**
- `confidence_by_agreement_outcome.png`
- `confidence_by_agreement_outcome.pdf`

**Description:** Violin plots showing confidence score distributions across four agreement patterns: Both Agree, Only GPT-4o, Only o4-mini, Neither Agrees. Includes individual data points and statistical comparisons.

**Key Insight:** Confidence distributions are similar across all agreement patterns. Models don't significantly lower confidence when they disagree with baseline or each other (t-tests show no significant differences).

**Code Implementation:**
- Categorizes questions by agreement pattern
- Creates violin plots with box plot overlays
- Adds strip plot for individual points
- Performs paired t-tests between models
- Annotates sample sizes per category

### Figure 5: ROC-style Analysis
**Files:**
- `roc_analysis.png`
- `roc_analysis.pdf`

**Description:** ROC curves using confidence as a threshold for predicting agreement with baseline. Shows how well confidence discriminates between correct and incorrect evaluations.

**Key Insight:** Both models perform barely better than random (AUC ≈ 0.5), confirming that confidence is a poor predictor of correctness. GPT-4o AUC: ~0.53, o4-mini AUC: ~0.51.

**Code Implementation:**
- Treats confidence as prediction score
- Calculates TPR/FPR at various thresholds
- Computes Area Under Curve (AUC)
- Compares to random baseline (diagonal)

### Figure 6: Cross-Model Confidence Correlation
**Files:**
- `cross_model_confidence_correlation.png`
- `cross_model_confidence_correlation.pdf`

**Description:** Scatter plot of GPT-4o confidence vs o4-mini confidence for the same questions. Points colored by agreement pattern with baseline. Includes marginal distributions.

**Key Insight:** Models show no correlation in confidence scores (r=-0.041, p=0.876). They assess difficulty independently, suggesting different internal calibration mechanisms.

**Code Implementation:**
- Merges data on shared questions
- Creates main scatter plot with agreement pattern coloring
- Adds marginal histograms for each model
- Calculates and displays correlation statistics
- Includes diagonal reference line

### Figure 7: Confidence vs Human Consensus Strength
**Files:**
- Original: `confidence_vs_human_consensus.png/pdf`
- Publication-ready separate plots:
  - `gpt4o_confidence_vs_human_consensus.png/pdf`
  - `o4mini_confidence_vs_human_consensus.png/pdf`
  - `combined_confidence_vs_human_consensus.png/pdf`

**Description:** Shows model confidence against human agreement percentage (% of humans voting for majority choice). Points colored green when model agrees with baseline, red when disagrees. Includes regression line and correlation statistics.

**Key Insight:** Neither model adjusts confidence based on human consensus difficulty. GPT-4o shows weak, non-significant positive correlation (r=0.35, p=0.163). o4-mini shows essentially no correlation (r=0.06, p=0.805).

**Code Implementation:**
- Calculates human consensus strength per question
- Plots confidence vs consensus for each model
- Colors points by agreement with baseline
- Adds regression lines and correlation statistics
- Shows that models don't calibrate to human difficulty

## Summary Statistics

### Model Performance
- **GPT-4o**: Mean confidence = 85.3%, Std = 8.0%, Baseline agreement = 70.6%
- **o4-mini**: Mean confidence = 79.9%, Std = 14.3%, Baseline agreement = 57.9%

### Correlation Results
- GPT-4o confidence vs agreement: r = 0.342, p = 0.152 (not significant)
- o4-mini confidence vs agreement: r ≈ 0 (no correlation)
- Cross-model confidence: r = -0.041, p = 0.876 (no correlation)
- Neither model's confidence correlates with human consensus strength

### Agreement Pattern Analysis
- When both models agree with baseline: Similar confidence levels
- When only one agrees: No significant confidence difference
- When neither agrees: Confidence remains high

## Conclusions

1. **Confidence is Uncalibrated**: Both models exhibit poor calibration, with confidence scores that don't reflect actual accuracy.

2. **Independence of Assessments**: The lack of correlation between GPT-4o and o4-mini confidence suggests they use different internal criteria for confidence.

3. **No Difficulty Awareness**: Models don't adjust confidence based on task difficulty as measured by human consensus.

4. **Overconfidence Bias**: Both models maintain high confidence even when incorrect, suggesting systematic overconfidence.

5. **Poor Discrimination**: ROC analysis shows confidence barely discriminates between correct and incorrect evaluations (AUC ≈ 0.5).

## Implications

These findings suggest that:
- Model confidence scores should not be used as reliability indicators
- Ensemble methods based on confidence weighting would be ineffective
- Models need better calibration mechanisms to align confidence with actual performance
- The lack of correlation with human consensus indicates models use different evaluation criteria than humans

## Reproducibility

To reproduce these figures:
```bash
cd agreement-results/
python create_confidence_correlation_figures.py  # All 7 figures
python create_publication_confidence_plots.py    # Publication-ready Figure 7 variants
```

Required data files:
- `gpt4o_proper_evaluation_final.csv`
- `o4mini_evaluation_final.csv`
- `BrowserArenaAgreementv2_July_18_2025_11.01.csv`

All figures are saved in both PNG (for viewing) and PDF (for publication) formats.

### Publication-Ready Figures
The `create_publication_confidence_plots.py` script generates enhanced versions of Figure 7 with:
- Separate plots for each model
- Larger text (16pt labels, 20pt titles)
- Colorblind-friendly colors (green/red)
- Professional styling with clean grids
- Simplified axis labels ("Human agreement (%)" and "Model confidence (%)")
- High DPI (300) for journal submission