#!/usr/bin/env python3
"""
Run GPT-4o evaluation on the exact survey questions with proper mappings.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import os
from openai import OpenAI

# Set up OpenAI client with correct API key
api_key = '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'
client = OpenAI(api_key=api_key)

print("=== RUNNING GPT-4O EXACT EVALUATION ===")

# Read the exact mappings we extracted
mapping_df = pd.read_csv('survey_questions_exact_mapping.csv')
print(f"Loaded {len(mapping_df)} question mappings")

def evaluate_with_gpt4o(task, agent1_output, agent2_output, agent1_model, agent2_model):
    """Evaluate two agent outputs using GPT-4o"""
    
    # Truncate outputs if too long
    agent1_output_truncated = agent1_output[:2000] + '...' if len(agent1_output) > 2000 else agent1_output
    agent2_output_truncated = agent2_output[:2000] + '...' if len(agent2_output) > 2000 else agent2_output
    
    prompt = f"""You are evaluating two browser automation agents on the following task:

Task: {task}

Agent 1 ({agent1_model}) Output:
{agent1_output_truncated}

Agent 2 ({agent2_model}) Output:
{agent2_output_truncated}

Evaluate which agent performed better based on:
1. Task completion success
2. Efficiency (fewer unnecessary steps)
3. Correctness of the final result
4. Overall quality of execution

Provide your evaluation in JSON format:
{{
    "preference": "Agent 1" or "Agent 2" or "Tie",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your decision"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert evaluator of browser automation agents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                'preference': result.get('preference', 'Tie'),
                'confidence': float(result.get('confidence', 0.5)),
                'reasoning': result.get('reasoning', 'No reasoning provided')
            }
        else:
            return {
                'preference': 'Tie',
                'confidence': 0.5,
                'reasoning': 'Could not parse response'
            }
            
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {
            'preference': 'Tie',
            'confidence': 0.5,
            'reasoning': f'Error: {str(e)}'
        }

# Run evaluations
results = []
print("\nEvaluating each question...")

for idx, row in mapping_df.iterrows():
    question = row['question']
    task = row['task']
    agent1_output = row['agent1_output']
    agent2_output = row['agent2_output']
    baseline_winner = row['baseline_winner']
    agent1_model = row['agent1_model']
    agent2_model = row['agent2_model']
    
    print(f"\n{question}: {task[:60]}...")
    
    # Skip if we don't have both outputs
    if not agent1_output or not agent2_output:
        print(f"  Skipping - missing agent outputs")
        continue
    
    # Run GPT-4o evaluation
    eval_result = evaluate_with_gpt4o(task, agent1_output, agent2_output, agent1_model, agent2_model)
    
    results.append({
        'question': question,
        'task': task,
        'baseline_winner': baseline_winner,
        'agent1_model': agent1_model,
        'agent2_model': agent2_model,
        'gpt4o_preference': eval_result['preference'],
        'gpt4o_confidence': eval_result['confidence'],
        'gpt4o_reasoning': eval_result['reasoning'],
        'agrees_with_baseline': eval_result['preference'] == baseline_winner,
        'timestamp': datetime.now().isoformat()
    })
    
    print(f"  Baseline: {baseline_winner}")
    print(f"  GPT-4o: {eval_result['preference']} (confidence: {eval_result['confidence']:.3f})")
    print(f"  Agrees: {'Yes' if eval_result['preference'] == baseline_winner else 'No'}")
    
    # Rate limiting
    time.sleep(0.5)
    
    # Save intermediate results every 5 evaluations
    if len(results) % 5 == 0:
        temp_df = pd.DataFrame(results)
        temp_df.to_csv('survey_questions_gpt4o_exact_temp.csv', index=False)
        print(f"  Saved {len(results)} results so far...")

# Save final results
results_df = pd.DataFrame(results)
results_df.to_csv('survey_questions_gpt4o_exact_evaluations.csv', index=False)
print(f"\n=== EVALUATION COMPLETE ===")
print(f"Saved {len(results)} evaluations to survey_questions_gpt4o_exact_evaluations.csv")

# Save question->confidence mapping for scatter plot
mapping_df = results_df[['question', 'gpt4o_confidence']].copy()
mapping_df.to_csv('question_to_gpt4o_confidence_exact.csv', index=False)

# Save as JSON for easy lookup
mapping_dict = {row['question']: row['gpt4o_confidence'] for _, row in mapping_df.iterrows()}
with open('question_to_gpt4o_confidence_exact.json', 'w') as f:
    json.dump(mapping_dict, f, indent=2)

# Summary statistics
print(f"\nSummary:")
print(f"Average confidence: {results_df['gpt4o_confidence'].mean():.3f}")
print(f"Min confidence: {results_df['gpt4o_confidence'].min():.3f}")
print(f"Max confidence: {results_df['gpt4o_confidence'].max():.3f}")
print(f"Agreement with baseline: {results_df['agrees_with_baseline'].sum()}/{len(results_df)} ({results_df['agrees_with_baseline'].mean()*100:.1f}%)")

# Show questions with lowest confidence
print(f"\nQuestions with lowest GPT-4o confidence:")
low_conf = results_df.nsmallest(5, 'gpt4o_confidence')[['question', 'gpt4o_confidence', 'gpt4o_preference']]
for _, row in low_conf.iterrows():
    print(f"  {row['question']}: {row['gpt4o_confidence']:.3f} ({row['gpt4o_preference']})")

# Show questions with highest confidence
print(f"\nQuestions with highest GPT-4o confidence:")
high_conf = results_df.nlargest(5, 'gpt4o_confidence')[['question', 'gpt4o_confidence', 'gpt4o_preference']]
for _, row in high_conf.iterrows():
    print(f"  {row['question']}: {row['gpt4o_confidence']:.3f} ({row['gpt4o_preference']})")