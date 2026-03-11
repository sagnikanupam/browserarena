# Paper-Ready Plots Documentation

This directory contains publication-quality plots comparing human annotators and AI models (O4-mini and GPT-4o) in evaluating browser automation agents. All plots are saved as PDF files at 300 DPI.

## Data Sources
- **Baseline**: Ground truth evaluations
- **V1 Human**: First crowdsourced evaluation (24 annotators)
- **V2 Human**: Second crowdsourced evaluation (116 annotators per question)
- **O4-mini**: OpenAI's O4-mini model with vision capabilities
- **GPT-4o**: OpenAI's GPT-4o model with vision capabilities

Note: Gemini results were excluded due to technical issues (100% tie responses).

## Main Plots (create_separate_paper_plots.py)

### 1. agreement_matrix.pdf
- **Type**: Heatmap
- **Content**: Pairwise agreement rates between all evaluators
- **Key Finding**: GPT-4o shows highest agreement with baseline (68%), humans show perfect agreement with each other

### 2. vote_distribution.pdf
- **Type**: Grouped bar chart
- **Content**: Distribution of Agent 1, Agent 2, and Tie votes by evaluator
- **Key Finding**: Humans and O4-mini prefer ties (42%), GPT-4o more decisive (10% ties)

### 3. confidence_distribution.pdf
- **Type**: Violin plot
- **Content**: Distribution of confidence scores for O4-mini and GPT-4o
- **Key Finding**: GPT-4o has higher average confidence (0.853) than O4-mini (0.799)

### 4. baseline_agreement.pdf
- **Type**: Bar chart
- **Content**: Agreement rates with baseline for each evaluator
- **Key Finding**: GPT-4o (68.4%) > V1/V2 Human (63.2%) > O4-mini (57.9%)

### 5. per_question_votes.pdf
- **Type**: Heatmap
- **Content**: Voting patterns across all 19 questions for each evaluator
- **Key Finding**: Visual representation of where evaluators agree/disagree

### 6. confidence_vs_agreement.pdf
- **Type**: Scatter plot
- **Content**: Model confidence vs whether they agreed with baseline
- **Key Finding**: No strong correlation between confidence and correctness

### 7. summary_statistics.pdf
- **Type**: Table (as figure)
- **Content**: Summary statistics for all evaluators
- **Key Finding**: Comprehensive numerical summary of all metrics

## Additional Plots (create_additional_paper_plots.py)

### 8. human_inter_annotator_agreement.pdf
- **Type**: Bar chart
- **Content**: Per-question pairwise agreement rates for V2 human annotators
- **Key Finding**: Mean agreement ~59%, varies significantly by question

### 9. model_confidence_vs_human_agreement.pdf
- **Type**: Dual scatter plots
- **Content**: Model confidence vs human majority agreement strength
- **Key Finding**: Models don't calibrate confidence to human consensus

### 10. disagreement_patterns.pdf
- **Type**: Pie chart
- **Content**: Breakdown of agreement patterns with baseline
- **Key Finding**: Shows which models disagree and when

### 11. tie_analysis.pdf
- **Type**: Bar chart + horizontal bar chart
- **Content**: Tie rates by evaluator and questions with most ties
- **Key Finding**: Clear difference in tie preference between evaluators

### 12. confidence_by_vote_type.pdf
- **Type**: Box plots
- **Content**: Model confidence distribution by vote type (Agent 1/2/Tie)
- **Key Finding**: O4-mini has lower confidence when choosing ties

## Usage

All plots are publication-ready and can be directly included in papers. They use:
- Professional color schemes (colorblind-friendly)
- Clear labels and legends
- High resolution (300 DPI)
- Consistent styling
- Arial font family

## Key Insights

1. **Human Bias**: Both V1 and V2 humans show identical voting patterns with high tie rates (42%)
2. **Model Performance**: GPT-4o aligns best with baseline, making more decisive judgments
3. **Confidence Patterns**: GPT-4o maintains high confidence even in borderline cases
4. **Agreement Structure**: Humans agree perfectly with each other but less with baseline
5. **Task Difficulty**: Some questions (Q11, Q22) show high disagreement across all evaluators

## Regenerating Plots

To regenerate all plots:
```bash
python create_separate_paper_plots.py
python create_additional_paper_plots.py
```

Requirements:
- pandas
- matplotlib
- seaborn
- numpy
- scipy
- sklearn

All necessary data files must be present in the same directory.