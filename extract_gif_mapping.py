#!/usr/bin/env python3
import re

# Read the survey interactions file
with open('verified-gifs-only/survey_interactions.txt', 'r') as f:
    content = f.read()

# Find all interactions and their GIF IDs
interactions = {}
current_interaction = None

lines = content.split('\n')
for i, line in enumerate(lines):
    # Check for interaction header
    if line.startswith('INTERACTION '):
        match = re.search(r'INTERACTION (\d+)', line)
        if match:
            current_interaction = int(match.group(1))
            interactions[current_interaction] = {'agent1': None, 'agent2': None}
    
    # Check for GIF IDs
    if 'GIF ID:' in line and current_interaction:
        gif_match = re.search(r'GIF ID: ([0-9]{2}_[0-9]{2}_[0-9]{4}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[A-Za-z]{3})\.gif', line)
        if gif_match:
            gif_id = gif_match.group(1)
            # Look for context to determine which agent
            context_start = max(0, i - 10)
            context = '\n'.join(lines[context_start:i])
            
            if 'MODEL ATTEMPT 1:' in context or 'Agent 1' in context:
                if not interactions[current_interaction]['agent1']:
                    interactions[current_interaction]['agent1'] = gif_id
            elif 'MODEL ATTEMPT 2:' in context or 'Agent 2' in context:
                if not interactions[current_interaction]['agent2']:
                    interactions[current_interaction]['agent2'] = gif_id

# Print the mapping
for i in sorted(interactions.keys()):
    print(f"Interaction {i}:")
    print(f"  Agent 1 GIF: {interactions[i]['agent1']}")
    print(f"  Agent 2 GIF: {interactions[i]['agent2']}")
    print()