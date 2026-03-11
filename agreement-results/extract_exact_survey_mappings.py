#!/usr/bin/env python3
"""
Extract exact mappings between survey questions and task data from HTML snippets.
"""

import json
import pandas as pd
from pathlib import Path
import re
from bs4 import BeautifulSoup

print("=== EXTRACTING EXACT SURVEY QUESTION MAPPINGS ===")

# Survey questions we need to map
survey_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 
                   'Q11', 'Q12', 'Q13', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q22']

# HTML snippets directory
snippets_dir = Path('../survey_html_snippets')
if not snippets_dir.exists():
    snippets_dir = Path('../survey_html_snippets_truncated')

print(f"Reading HTML snippets from {snippets_dir}")

def extract_from_html(html_file):
    """Extract task and agent outputs from HTML snippet"""
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse with BeautifulSoup for better extraction
    soup = BeautifulSoup(content, 'html.parser')
    
    # Extract task
    task = None
    task_div = soup.find('div', string=re.compile('Task:'))
    if task_div:
        task_text = task_div.get_text()
        task = task_text.replace('Task:', '').strip()
    else:
        # Try alternative extraction
        task_match = re.search(r'<strong>Task:</strong>\s*([^<]+)', content)
        if task_match:
            task = task_match.group(1).strip()
    
    # Extract agent outputs
    agent_outputs = {}
    
    # Find all agent sections
    agent_sections = soup.find_all('div', class_='agent-column')
    
    for i, section in enumerate(agent_sections):
        agent_num = i + 1
        
        # Extract output text
        output_div = section.find('div', class_='output-text')
        if output_div:
            output_text = output_div.get_text(separator='\n').strip()
            agent_outputs[f'agent{agent_num}_output'] = output_text
        
        # Extract GIF URL
        gif_img = section.find('img')
        if gif_img and 'src' in gif_img.attrs:
            agent_outputs[f'agent{agent_num}_gif'] = gif_img['src']
    
    # If BeautifulSoup didn't work, try regex
    if not agent_outputs:
        # Agent 1 output
        agent1_match = re.search(r'Agent 1.*?<div class="output-text">(.*?)</div>', content, re.DOTALL)
        if agent1_match:
            agent_outputs['agent1_output'] = agent1_match.group(1).replace('<br>', '\n').strip()
        
        # Agent 2 output
        agent2_match = re.search(r'Agent 2.*?<div class="output-text">(.*?)</div>', content, re.DOTALL)
        if agent2_match:
            agent_outputs['agent2_output'] = agent2_match.group(1).replace('<br>', '\n').strip()
    
    return task, agent_outputs

# Map interaction numbers to survey questions
# Based on the file naming (interaction_01.html, etc.), they likely correspond to Q1, Q2, etc.
# But we need to verify this

html_files = sorted(snippets_dir.glob('interaction_*.html'))
print(f"Found {len(html_files)} interaction HTML files")

# Extract data from each HTML file
interaction_data = {}
for html_file in html_files:
    # Extract interaction number
    match = re.search(r'interaction_(\d+)', html_file.name)
    if match:
        interaction_num = int(match.group(1))
        
        task, outputs = extract_from_html(html_file)
        
        interaction_data[interaction_num] = {
            'file': html_file.name,
            'task': task,
            **outputs
        }
        
        print(f"\nInteraction {interaction_num}:")
        print(f"  Task: {task[:80]}..." if task else "  Task: NOT FOUND")
        print(f"  Agent 1 output: {'Found' if outputs.get('agent1_output') else 'Not found'}")
        print(f"  Agent 2 output: {'Found' if outputs.get('agent2_output') else 'Not found'}")

# Now we need to map interaction numbers to survey questions
# The survey has 19 questions but we have 25 interactions
# Let's check if there's a pattern

print("\n=== MAPPING INTERACTIONS TO SURVEY QUESTIONS ===")

# Read baseline to get the expected questions
baseline_df = pd.read_csv('baseline.csv')

# If we have exactly 19 survey questions and interactions 1-19 exist,
# we can make a direct mapping
direct_mapping = {}
for i, question in enumerate(survey_questions, 1):
    if i in interaction_data:
        direct_mapping[question] = interaction_data[i]
        print(f"{question} -> interaction_{i:02d}.html")

# Some questions might be missing (Q14, Q15, Q21, Q23, Q24, Q25 are not in survey_questions)
# Let's handle this by creating a complete mapping

# Get all questions from baseline
all_baseline_questions = sorted(baseline_df['question'].unique())
print(f"\nAll baseline questions: {all_baseline_questions}")

# Create the final mapping
final_mapping = {}
question_to_interaction = {
    'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4, 'Q5': 5,
    'Q6': 6, 'Q7': 7, 'Q8': 8, 'Q9': 9, 'Q10': 10,
    'Q11': 11, 'Q12': 12, 'Q13': 13, 'Q14': 14, 'Q15': 15,
    'Q16': 16, 'Q17': 17, 'Q18': 18, 'Q19': 19, 'Q20': 20,
    'Q21': 21, 'Q22': 22, 'Q23': 23, 'Q24': 24, 'Q25': 25
}

# Build final mapping with all available data
results = []
for question in survey_questions:
    interaction_num = question_to_interaction.get(question)
    
    if interaction_num and interaction_num in interaction_data:
        data = interaction_data[interaction_num]
        
        # Get baseline info
        baseline_row = baseline_df[baseline_df['question'] == question]
        if not baseline_row.empty:
            results.append({
                'question': question,
                'interaction_number': interaction_num,
                'html_file': data['file'],
                'task': data.get('task', ''),
                'agent1_output': data.get('agent1_output', ''),
                'agent2_output': data.get('agent2_output', ''),
                'agent1_gif': data.get('agent1_gif', ''),
                'agent2_gif': data.get('agent2_gif', ''),
                'baseline_winner': baseline_row.iloc[0]['winner'],
                'agent1_model': baseline_row.iloc[0]['agent1_model'],
                'agent2_model': baseline_row.iloc[0]['agent2_model']
            })

# Save the exact mappings
if results:
    mapping_df = pd.DataFrame(results)
    
    # Save full mapping
    mapping_df.to_csv('survey_questions_exact_mapping.csv', index=False)
    print(f"\n=== SAVED EXACT MAPPINGS ===")
    print(f"Saved {len(mapping_df)} question mappings to survey_questions_exact_mapping.csv")
    
    # Save a simplified version for the GPT-4o evaluation
    simple_mapping = []
    for _, row in mapping_df.iterrows():
        simple_mapping.append({
            'question': row['question'],
            'task': row['task'][:200] + '...' if len(row['task']) > 200 else row['task'],
            'has_agent1_output': bool(row['agent1_output']),
            'has_agent2_output': bool(row['agent2_output']),
            'baseline_winner': row['baseline_winner']
        })
    
    simple_df = pd.DataFrame(simple_mapping)
    simple_df.to_csv('survey_questions_task_summary.csv', index=False)
    
    # Show summary
    print(f"\nSummary:")
    print(f"Questions with task data: {len(mapping_df[mapping_df['task'] != ''])}")
    print(f"Questions with both agent outputs: {len(mapping_df[(mapping_df['agent1_output'] != '') & (mapping_df['agent2_output'] != '')])}")
    
    # Show first few mappings
    print("\nFirst 5 mappings:")
    for i, row in mapping_df.head().iterrows():
        print(f"\n{row['question']}:")
        print(f"  Task: {row['task'][:80]}..." if row['task'] else "  Task: MISSING")
        print(f"  Baseline winner: {row['baseline_winner']}")
        print(f"  Models: {row['agent1_model']} vs {row['agent2_model']}")

    print("\nReady to run GPT-4o evaluation with exact task data!")
else:
    print("\nNo mappings could be created. Check the data sources.")