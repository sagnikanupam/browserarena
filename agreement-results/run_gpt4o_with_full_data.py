#!/usr/bin/env python3
"""
Run GPT-4o evaluation with full vLLM traces and GIF files.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import os
import base64
from openai import OpenAI

# Set up OpenAI client
api_key = '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'
client = OpenAI(api_key=api_key)

print("=== RUNNING GPT-4O EVALUATION WITH FULL DATA ===")

# Read the mapping data
mapping_df = pd.read_csv('survey_questions_exact_mapping.csv')
print(f"Loaded {len(mapping_df)} question mappings")

# Read baseline for comparison
baseline_df = pd.read_csv('baseline.csv')

def find_agent_data(task_description, agent_model):
    """Find the JSON file with full vLLM trace for a given task and model"""
    prompts_dir = Path('../FastChat/prompts_and_outputs')
    
    # Search for files that might match this task
    # This is a simplified approach - in reality we'd need better matching
    json_files = list(prompts_dir.glob('*.json'))
    
    # Try to find files from around the survey date
    # The survey questions are from specific dates
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if prompt matches
            if 'prompt' in data:
                prompt_text = data['prompt'].lower()
                task_lower = task_description.lower()
                
                # Simple matching - check if key words from task are in prompt
                task_words = task_lower.split()[:5]  # First 5 words
                if all(word in prompt_text for word in task_words if len(word) > 3):
                    return data
        except:
            continue
    
    return None

def encode_image_to_base64(image_path):
    """Encode image to base64 for GPT-4 vision"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except:
        return None

def evaluate_with_gpt4_vision(task, agent1_data, agent2_data, agent1_gif_path, agent2_gif_path, 
                              agent1_model, agent2_model):
    """Evaluate using GPT-4 with vision capabilities"""
    
    # Prepare the full traces
    agent1_trace = ""
    agent2_trace = ""
    
    if agent1_data and 'output' in agent1_data:
        if isinstance(agent1_data['output'], list):
            for step in agent1_data['output']:
                if isinstance(step, dict):
                    agent1_trace += f"Step {step.get('step', '?')}: {step.get('action', '')}\n"
                    if 'observation' in step:
                        agent1_trace += f"  Observation: {step['observation'][:200]}...\n"
        else:
            agent1_trace = str(agent1_data['output'])[:3000]
    
    if agent2_data and 'output' in agent2_data:
        if isinstance(agent2_data['output'], list):
            for step in agent2_data['output']:
                if isinstance(step, dict):
                    agent2_trace += f"Step {step.get('step', '?')}: {step.get('action', '')}\n"
                    if 'observation' in step:
                        agent2_trace += f"  Observation: {step['observation'][:200]}...\n"
        else:
            agent2_trace = str(agent2_data['output'])[:3000]
    
    # If we don't have full traces, use what we have from HTML
    if not agent1_trace:
        agent1_trace = "No full trace available - using HTML snippet data"
    if not agent2_trace:
        agent2_trace = "No full trace available - using HTML snippet data"
    
    # Prepare messages for GPT-4V
    messages = [
        {
            "role": "system",
            "content": """You are an expert evaluator of browser automation agents. 
You will be shown the execution traces and visual recordings (GIFs) of two agents attempting the same task.
Evaluate based on:
1. Task completion success (most important)
2. Efficiency (fewer unnecessary steps)
3. Correctness of the final result
4. Overall quality of execution

Pay special attention to the GIF recordings as they show the actual browser interactions."""
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Task: {task}

Agent 1 ({agent1_model}) Execution Trace:
{agent1_trace[:2000]}{'...' if len(agent1_trace) > 2000 else ''}

Agent 2 ({agent2_model}) Execution Trace:
{agent2_trace[:2000]}{'...' if len(agent2_trace) > 2000 else ''}

Please analyze both the execution traces and the GIF recordings below to determine which agent performed better."""
                }
            ]
        }
    ]
    
    # Add GIFs if available
    if agent1_gif_path and Path(agent1_gif_path).exists():
        base64_image = encode_image_to_base64(agent1_gif_path)
        if base64_image:
            messages[1]["content"].append({
                "type": "text",
                "text": "Agent 1 GIF recording:"
            })
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/gif;base64,{base64_image}",
                    "detail": "high"
                }
            })
    
    if agent2_gif_path and Path(agent2_gif_path).exists():
        base64_image = encode_image_to_base64(agent2_gif_path)
        if base64_image:
            messages[1]["content"].append({
                "type": "text",
                "text": "Agent 2 GIF recording:"
            })
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/gif;base64,{base64_image}",
                    "detail": "high"
                }
            })
    
    # Add final prompt
    messages[1]["content"].append({
        "type": "text",
        "text": """Based on the execution traces and GIF recordings, provide your evaluation in JSON format:
{
    "preference": "Agent 1" or "Agent 2" or "Tie",
    "confidence": 0.0 to 1.0,
    "reasoning": "Detailed explanation of your decision based on both traces and visual evidence"
}"""
    })
    
    try:
        # Use GPT-4 Vision model
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            temperature=0.1,
            max_tokens=500
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

# Process each question
results = []
print("\nEvaluating each question with full data...")

for idx, row in mapping_df.iterrows():
    question = row['question']
    task = row['task']
    baseline_winner = row['baseline_winner']
    agent1_model = row['agent1_model']
    agent2_model = row['agent2_model']
    
    print(f"\n{question}: {task[:60]}...")
    
    # Find GIF files
    agent1_gif = None
    agent2_gif = None
    
    # Try to find GIFs based on the HTML file names
    # The GIFs should be in FastChat/gifs or browserarena-gifs
    gif_dirs = [
        Path('../FastChat/gifs'),
        Path('../browserarena-gifs'),
        Path('../verified-gifs-only')
    ]
    
    # Extract GIF URLs from the mapping if available
    if 'agent1_gif' in row and row['agent1_gif']:
        # Convert URL to file path
        gif_name = Path(row['agent1_gif']).name
        for gif_dir in gif_dirs:
            potential_path = gif_dir / gif_name
            if potential_path.exists():
                agent1_gif = str(potential_path)
                break
    
    if 'agent2_gif' in row and row['agent2_gif']:
        gif_name = Path(row['agent2_gif']).name
        for gif_dir in gif_dirs:
            potential_path = gif_dir / gif_name
            if potential_path.exists():
                agent2_gif = str(potential_path)
                break
    
    # Find full vLLM traces
    agent1_data = find_agent_data(task, agent1_model)
    agent2_data = find_agent_data(task, agent2_model)
    
    # If we don't have full data, use what we have from the mapping
    if not agent1_data:
        agent1_data = {'output': row.get('agent1_output', '')}
    if not agent2_data:
        agent2_data = {'output': row.get('agent2_output', '')}
    
    print(f"  Agent 1 GIF: {'Found' if agent1_gif else 'Not found'}")
    print(f"  Agent 2 GIF: {'Found' if agent2_gif else 'Not found'}")
    print(f"  Agent 1 trace: {'Found' if agent1_data else 'Not found'}")
    print(f"  Agent 2 trace: {'Found' if agent2_data else 'Not found'}")
    
    # Run evaluation
    eval_result = evaluate_with_gpt4_vision(
        task, agent1_data, agent2_data, 
        agent1_gif, agent2_gif,
        agent1_model, agent2_model
    )
    
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
        'had_agent1_gif': bool(agent1_gif),
        'had_agent2_gif': bool(agent2_gif),
        'had_full_traces': bool(agent1_data and agent2_data),
        'timestamp': datetime.now().isoformat()
    })
    
    print(f"  Baseline: {baseline_winner}")
    print(f"  GPT-4o: {eval_result['preference']} (confidence: {eval_result['confidence']:.3f})")
    print(f"  Agrees: {'Yes' if eval_result['preference'] == baseline_winner else 'No'}")
    
    # Rate limiting
    time.sleep(2)
    
    # Save intermediate results
    if len(results) % 5 == 0:
        temp_df = pd.DataFrame(results)
        temp_df.to_csv('gpt4o_full_data_evaluation_temp.csv', index=False)
        print(f"  Saved {len(results)} results so far...")

# Save final results
results_df = pd.DataFrame(results)
results_df.to_csv('gpt4o_full_data_evaluation_final.csv', index=False)
print(f"\n=== EVALUATION COMPLETE ===")
print(f"Saved {len(results)} evaluations to gpt4o_full_data_evaluation_final.csv")

# Summary statistics
print(f"\nSummary:")
print(f"Questions with both GIFs: {results_df['had_agent1_gif'].sum()}")
print(f"Questions with full traces: {results_df['had_full_traces'].sum()}")
print(f"Average confidence: {results_df['gpt4o_confidence'].mean():.3f}")
print(f"Agreement with baseline: {results_df['agrees_with_baseline'].sum()}/{len(results_df)} ({results_df['agrees_with_baseline'].mean()*100:.1f}%)")

# Save confidence mapping
mapping_df = results_df[['question', 'gpt4o_confidence']].copy()
mapping_df.to_csv('question_to_gpt4o_confidence_full_data.csv', index=False)