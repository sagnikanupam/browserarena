# Verified Interactions Extraction Report

## Summary
Successfully extracted all 126 verified interactions from the JSON file with their correct GIF IDs, Log IDs, and metadata.

## Processing Results

### Total Interactions: 126
- **Vote Type Distribution:**
  - Left votes: 59 (46.8%)
  - Right votes: 60 (47.6%)
  - Tie votes: 7 (5.6%)

### Models Tested
The following 6 models were evaluated:
1. `anthropic/claude-3.7-sonnet:thinking`
2. `deepseek/deepseek-r1`
3. `google/gemini-2.5-pro-preview-03-25`
4. `meta-llama/llama-4-maverick`
5. `openai/o4-mini`
6. `x-ai/grok-3-beta`

### Data Extraction Success
- **GIF IDs found:** 235 out of 252 total attempts (93.3%)
- **Log IDs found:** 235 out of 252 total attempts (93.3%)
- **Missing GIF/Log IDs:** 17 attempts (6.7%)

### File Verification
- **Referenced GIF files:** 227 unique GIF IDs
- **All referenced files exist:** ✅ 100% success rate
- **Total GIF files on disk:** 693 files
- **Unreferenced GIF files:** 466 files (likely from other experiments)

## Output Files Generated

1. **`extracted_verified_interactions.txt`** - Complete detailed listing of all 126 interactions
2. **`verified_interactions_summary.csv`** - Structured CSV format with 252 rows (126 interactions × 2 models each)
3. **`extraction_report.md`** - This summary report

## Data Structure Extracted for Each Interaction

For each of the 126 interactions, the following information was extracted:

### Interaction Metadata
- Interaction number (1-126)
- Vote type (leftvote/rightvote/tievote)
- User task/request

### For Each Model (Left & Right)
- Model name
- GIF ID (extracted from `<img src="/gradio_api/file=gifs/[ID].gif">`)
- Log ID (extracted from `Log ID: [ID]`)
- Task success status (SUCCESS/FAILED/UNCLEAR)

## Key Findings

1. **Data Quality**: 93.3% of interactions have complete GIF and Log ID information
2. **Model Coverage**: All 6 models appear across multiple interactions
3. **File Integrity**: All referenced GIF files exist and are accessible
4. **Vote Distribution**: Nearly equal distribution between left and right votes, with minimal tie votes

## Technical Implementation

The extraction process:
1. Parsed the 6.2MB JSON file containing all interaction data
2. Identified the correct message structure (array format with role/content pairs)
3. Used regex patterns to extract GIF IDs from HTML img tags
4. Used regex patterns to extract Log IDs from text content
5. Determined success/failure status based on content analysis
6. Verified all referenced files exist on disk

## Files Available for Analysis

All extraction outputs are available in `/Users/davisbrown/browserarena/`:
- `extracted_verified_interactions.txt` - Human-readable format
- `verified_interactions_summary.csv` - Machine-readable format
- Raw source: `FastChat/verified_conv_log/verified_interactions-clean.json`

The extraction successfully captured all 126 verified interactions with proper GIF ID mapping for further analysis.