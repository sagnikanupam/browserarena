#!/usr/bin/env python3
"""
Find the actual GIF files for each survey question.
"""

import pandas as pd
from pathlib import Path
import re

# Read the mapping
mapping_df = pd.read_csv('survey_questions_exact_mapping.csv')

# Read the verified interactions to get case IDs
verified_file = Path('data_updated/IntermediateFiles/verified_interactions-conv_initial.json')
gif_mapping = {}

# Check different potential sources
print("Searching for GIF files...")

# 1. Check verified-gifs-only directory
verified_gifs_dir = Path('../verified-gifs-only')
if verified_gifs_dir.exists():
    gif_files = list(verified_gifs_dir.glob('*.gif'))
    print(f"Found {len(gif_files)} GIFs in verified-gifs-only")
    
    # Sample files to understand naming
    for gif in gif_files[:5]:
        print(f"  Sample: {gif.name}")

# 2. Check FastChat/gifs directory  
fastchat_gifs = Path('../FastChat/gifs')
if fastchat_gifs.exists():
    gif_files = list(fastchat_gifs.glob('*.gif'))
    print(f"\nFound {len(gif_files)} GIFs in FastChat/gifs")
    
    # Look for patterns
    date_pattern = re.compile(r'(\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2}_\w{3})')
    
    for gif in gif_files[:10]:
        match = date_pattern.search(gif.name)
        if match:
            print(f"  Sample: {gif.name} -> ID: {match.group(1)}")

# 3. Try to match survey questions to GIFs
# The survey was conducted around May 2025, so look for GIFs from that period
may_gifs = []
for gif_dir in [verified_gifs_dir, fastchat_gifs]:
    if gif_dir.exists():
        may_gifs.extend(list(gif_dir.glob('*05_2025*.gif')))

print(f"\nFound {len(may_gifs)} GIFs from May 2025")

# Extract unique case IDs
case_ids = set()
for gif in may_gifs:
    match = date_pattern.search(gif.name)
    if match:
        case_ids.add(match.group(1))

print(f"Found {len(case_ids)} unique case IDs from May 2025")

# For the survey questions, we need to find which GIFs correspond to each question
# This would require matching the task descriptions or finding a mapping file

# Check if there's a mapping in the faithfulness data
faith_data = pd.read_csv('data_updated/faithfulness_data.csv')
print(f"\nFaithfulness data shape: {faith_data.shape}")
print(f"Columns with case info: {[col for col in faith_data.columns if 'Q7' in col or 'Q25' in col]}")

# Q7 and Q25 contain the case IDs
if 'Q7' in faith_data.columns and 'Q25' in faith_data.columns:
    print("\nSample case IDs from faithfulness data:")
    for i in range(min(5, len(faith_data))):
        left_case = faith_data.iloc[i]['Q7']
        right_case = faith_data.iloc[i]['Q25']
        task = faith_data.iloc[i]['Q2'] if 'Q2' in faith_data.columns else 'N/A'
        print(f"  Row {i}: {left_case} vs {right_case}")
        print(f"    Task: {str(task)[:60]}...")
        
        # Check if these GIFs exist
        for case in [left_case, right_case]:
            if pd.notna(case):
                for gif_dir in [verified_gifs_dir, fastchat_gifs]:
                    if gif_dir.exists():
                        matching_gifs = list(gif_dir.glob(f'*{case}*.gif'))
                        if matching_gifs:
                            print(f"    Found GIF: {matching_gifs[0].name}")

# Create a mapping file
print("\nCreating GIF mapping file...")
gif_mapping_data = []

# We need to map survey questions to the faithfulness data rows
# This is complex without the exact mapping, but we can try based on task similarity

for idx, row in mapping_df.iterrows():
    question = row['question']
    task = row['task']
    
    # Try to find matching task in faithfulness data
    best_match_idx = None
    best_match_score = 0
    
    if 'Q2' in faith_data.columns:
        for faith_idx, faith_row in faith_data.iterrows():
            faith_task = str(faith_row['Q2'])
            
            # Simple matching based on common words
            task_words = set(task.lower().split())
            faith_words = set(faith_task.lower().split())
            common_words = task_words.intersection(faith_words)
            
            if len(common_words) > best_match_score:
                best_match_score = len(common_words)
                best_match_idx = faith_idx
    
    # If we found a match, get the case IDs
    agent1_gif_path = None
    agent2_gif_path = None
    
    if best_match_idx is not None and best_match_score > 5:
        faith_row = faith_data.iloc[best_match_idx]
        left_case = faith_row.get('Q7')
        right_case = faith_row.get('Q25')
        
        # Find actual GIF files
        for case, agent_num in [(left_case, 1), (right_case, 2)]:
            if pd.notna(case):
                for gif_dir in [fastchat_gifs, verified_gifs_dir]:
                    if gif_dir.exists():
                        matching_gifs = list(gif_dir.glob(f'*{case}*.gif'))
                        if matching_gifs:
                            if agent_num == 1:
                                agent1_gif_path = str(matching_gifs[0])
                            else:
                                agent2_gif_path = str(matching_gifs[0])
                            break
    
    gif_mapping_data.append({
        'question': question,
        'task': task[:100],
        'agent1_gif_path': agent1_gif_path,
        'agent2_gif_path': agent2_gif_path,
        'match_score': best_match_score
    })

gif_mapping_df = pd.DataFrame(gif_mapping_data)
gif_mapping_df.to_csv('survey_question_gif_mapping.csv', index=False)

print(f"\nGIF mapping summary:")
print(f"Questions with both GIFs: {len(gif_mapping_df[(gif_mapping_df['agent1_gif_path'].notna()) & (gif_mapping_df['agent2_gif_path'].notna())])}")
print(f"Questions with at least one GIF: {len(gif_mapping_df[(gif_mapping_df['agent1_gif_path'].notna()) | (gif_mapping_df['agent2_gif_path'].notna())])}")
print(f"Questions with no GIFs: {len(gif_mapping_df[(gif_mapping_df['agent1_gif_path'].isna()) & (gif_mapping_df['agent2_gif_path'].isna())])}")

# Show sample mappings
print("\nSample mappings:")
for _, row in gif_mapping_df.head(5).iterrows():
    print(f"\n{row['question']}: {row['task']}")
    print(f"  Agent 1 GIF: {Path(row['agent1_gif_path']).name if row['agent1_gif_path'] else 'Not found'}")
    print(f"  Agent 2 GIF: {Path(row['agent2_gif_path']).name if row['agent2_gif_path'] else 'Not found'}")
    print(f"  Match score: {row['match_score']}")