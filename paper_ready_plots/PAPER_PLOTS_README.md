# Paper-Ready Plots Summary

This document lists all the publication-ready plots generated for the GPT-4o vs Human evaluation analysis.

## All Cases Analysis (126 cases, 55.6% agreement)

### Individual Plot Components:
1. **Agreement by Preference**: `paper_agreement_by_preference_all_cases.pdf/.png`
   - Shows agreement rates when humans preferred left (59.7%), right (56.9%), or tie (0.0%)
   - Bar chart with sample sizes labeled

2. **Vote Distribution**: `paper_vote_distribution_all_cases.pdf/.png`
   - Compares human vs GPT-4o vote distributions
   - Shows GPT-4o declares more ties (19%) vs humans (5%)

3. **Confidence Distribution**: `paper_confidence_distribution_all_cases.pdf/.png`
   - Histogram of GPT-4o confidence scores
   - Mean confidence: 0.75

4. **Confidence by Agreement**: `paper_confidence_by_agreement_all_cases.pdf/.png`
   - Box plots showing confidence when agreeing (0.85) vs disagreeing (0.63)

## Both Agents Completed Analysis (87 cases, 40.2% agreement)

### Individual Plot Components:
1. **Agreement by Preference**: `paper_agreement_by_preference_both_completed.pdf/.png`
   - Agreement rates when both agents completed: left (50.0%), right (34.2%), tie (0.0%)
   - Lower agreement rates when both agents succeed

2. **Vote Distribution**: `paper_vote_distribution_both_completed.pdf/.png`
   - When both complete, GPT-4o declares even more ties (26% vs 19% overall)
   - Humans still have clear preferences (51% left, 44% right, 6% tie)

3. **Confidence Distribution**: `paper_confidence_distribution_both_completed.pdf/.png`
   - Lower average confidence (0.70) when both agents complete tasks
   - More uncertainty about quality differences

4. **Confidence by Agreement**: `paper_confidence_by_agreement_both_completed.pdf/.png`
   - Similar pattern but lower overall confidence levels

## Comparison Plots

### Direct Comparisons:
1. **Overall Agreement**: `paper_overall_agreement_comparison.pdf/.png`
   - Side-by-side comparison: All cases (55.6%) vs Both completed (40.2%)
   - Shows task completion success makes evaluation harder

2. **Agreement by Preference**: `paper_preference_agreement_comparison.pdf/.png`
   - Compares agreement rates by preference type across datasets
   - Shows biggest drop for right-preference cases (57% → 34%)

## Original Analysis Separated

### From `improved_gpt4o_human_analysis.png`:
1. **Agreement by Preference**: `paper_original_agreement_by_preference.pdf/.png`
2. **Vote Distribution**: `paper_original_vote_distribution.pdf/.png` 
3. **Confidence Distribution**: `paper_original_confidence_distribution.pdf/.png`
4. **Confidence by Agreement**: `paper_original_confidence_by_agreement.pdf/.png`

## Key Findings Summary

### Main Results:
- **Overall agreement drops significantly when both agents complete tasks** (55.6% → 40.2%)
- **GPT-4o becomes more conservative**, declaring more ties when both succeed (26% vs 19%)
- **Humans maintain clear preferences** even when both agents work (only 6% ties)
- **Agreement varies by human preference type**, with right-preference showing largest drop

### Publication Notes:
- All plots available in both PDF (vector) and PNG (raster) formats
- Publication-ready styling with Times New Roman font
- Consistent color schemes and formatting
- Error bars and sample sizes included where appropriate
- All plots are 300 DPI for high-quality reproduction

## File Structure
```
paper_[plot_type]_[dataset].pdf/.png
├── agreement_by_preference_all_cases
├── agreement_by_preference_both_completed  
├── vote_distribution_all_cases
├── vote_distribution_both_completed
├── confidence_distribution_all_cases
├── confidence_distribution_both_completed
├── confidence_by_agreement_all_cases
├── confidence_by_agreement_both_completed
├── overall_agreement_comparison
├── preference_agreement_comparison
└── original_[plot_name] (separated from combined figure)
```

Total: **14 individual plot components** + **4 original separated plots** = **18 publication-ready figures**