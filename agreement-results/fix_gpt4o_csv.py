#!/usr/bin/env python3
"""
Fix the GPT-4o CSV file that has corrupted Q7 row.
"""

import pandas as pd
import csv

# Read the file line by line and fix
rows = []
with open('gpt4o_proper_evaluation_final.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    rows.append(headers)
    
    for row in reader:
        if len(row) > 12:
            # This is the corrupted Q7 row
            # Extract the relevant fields
            question = row[0]
            task = row[1]
            baseline = row[2]
            agent1 = row[3]
            agent2 = row[4]
            preference = row[5]
            confidence = row[6]
            # The reasoning is repeated 3 times, take just the first occurrence
            reasoning_parts = ' '.join(row[7:])
            # Find where the reasoning starts repeating
            base_reasoning = "Agent 1 successfully navigated to the Fox News homepage and extracted the top headlines"
            first_end = reasoning_parts.find(", as evidenced by")
            if first_end > 0:
                reasoning = reasoning_parts[:first_end] + "."
            else:
                reasoning = base_reasoning + "."
            
            # Reconstruct the row
            fixed_row = [question, task, baseline, agent1, agent2, preference, confidence, 
                        reasoning, 'True', '2', 'True', '2025-07-26T12:18:00.282062']
            rows.append(fixed_row)
        else:
            rows.append(row)

# Write the fixed file
with open('gpt4o_proper_evaluation_fixed.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("Fixed GPT-4o CSV saved to gpt4o_proper_evaluation_fixed.csv")

# Verify it can be read
df = pd.read_csv('gpt4o_proper_evaluation_fixed.csv')
print(f"Successfully loaded {len(df)} rows")
print(f"Q7 reasoning length: {len(df[df['question'] == 'Q7']['gpt4o_reasoning'].iloc[0])}")