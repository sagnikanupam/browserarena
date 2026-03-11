# BrowserArena Evaluation Comparison Report

## Overview
This report compares three evaluation sources:
1. **Original Baseline**: Single annotator judgments
2. **Human V2**: New crowdsourced human evaluations
3. **GPT-4.1**: AI model evaluations

## Key Findings

### Inter-Annotator Reliability (Human V2)
- Krippendorff's Alpha: 0.289
- Fleiss' Kappa: 0.289
- Average Majority Agreement: 71.4%

### Pairwise Agreement Rates
- Baseline vs Human V2: 12/19 (63.2%)
- Baseline vs GPT-4.1: 19/19 (100.0%)
- Human V2 vs GPT-4.1: 12/19 (63.2%)

### Distribution Comparison
| Source | Agent 1 | Agent 2 | Tie |
|--------|---------|---------|-----|
| Baseline | 17 (89.5%) | 1 (5.3%) | 1 (5.3%) |
| Human V2 | 10 (52.6%) | 1 (5.3%) | 8 (42.1%) |
| GPT-4.1 | 17 (89.5%) | 1 (5.3%) | 1 (5.3%) |

### Questions with Disagreement

Total questions with disagreement: 7/19

**Q11**:
- Original: Agent 1
- Human V2: Tie (68.4% agreement)
- GPT-4.1: Agent 1
- V2 Votes: Agent1=28, Agent2=8, Tie=78

**Q12**:
- Original: Agent 1
- Human V2: Tie (57.0% agreement)
- GPT-4.1: Agent 1
- V2 Votes: Agent1=31, Agent2=18, Tie=65

**Q16**:
- Original: Agent 1
- Human V2: Tie (60.5% agreement)
- GPT-4.1: Agent 1
- V2 Votes: Agent1=41, Agent2=4, Tie=69

**Q17**:
- Original: Agent 1
- Human V2: Tie (59.6% agreement)
- GPT-4.1: Agent 1
- V2 Votes: Agent1=37, Agent2=9, Tie=68

**Q18**:
- Original: Agent 1
- Human V2: Tie (50.9% agreement)
- GPT-4.1: Agent 1
- V2 Votes: Agent1=47, Agent2=9, Tie=58

**Q5**:
- Original: Agent 1
- Human V2: Tie (71.1% agreement)
- GPT-4.1: Agent 1
- V2 Votes: Agent1=21, Agent2=12, Tie=81

**Q7**:
- Original: Agent 1
- Human V2: Tie (58.8% agreement)
- GPT-4.1: Agent 1
- V2 Votes: Agent1=36, Agent2=11, Tie=67

