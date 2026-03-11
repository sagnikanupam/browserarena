#!/usr/bin/env python3
"""
Run o4-mini evaluation with GIFs and full vLLM traces.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import base64
from openai import OpenAI

# Set up OpenAI client
api_key = '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'
client = OpenAI(api_key=api_key)

print("=== RUNNING O4-MINI EVALUATION WITH GIFS ===")

# Read the mappings
mapping_df = pd.read_csv('survey_questions_exact_mapping.csv')
gif_mapping_df = pd.read_csv('survey_question_gif_mapping.csv')
baseline_df = pd.read_csv('baseline.csv')

# Merge the mappings
full_mapping = mapping_df.merge(gif_mapping_df[['question', 'agent1_gif_path', 'agent2_gif_path']], on='question')
print(f"Loaded {len(full_mapping)} question mappings with GIF paths")

def load_vllm_trace(gif_path):
    """Load the full vLLM trace from the corresponding JSON file"""
    if not gif_path or pd.isna(gif_path):
        return None
    
    # Extract case ID from GIF path
    gif_name = Path(gif_path).stem
    
    # Look for corresponding JSON in prompts_and_outputs
    json_path = Path(f'../FastChat/prompts_and_outputs/{gif_name}.json')
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except:
            pass
    
    return None

def encode_image_to_base64(image_path):
    """Encode image to base64 for GPT-4V"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def format_vllm_trace(trace_data):
    """Format vLLM trace for presentation to o4-mini"""
    if not trace_data:
        return "No trace data available"
    
    formatted = []
    
    # Add prompt
    if 'prompt' in trace_data:
        formatted.append(f"TASK: {trace_data['prompt']}")
        formatted.append("")
    
    # Add output steps
    if 'output' in trace_data:
        output = trace_data['output']
        
        if isinstance(output, list):
            for i, step in enumerate(output):
                if isinstance(step, dict):
                    formatted.append(f"Step {i+1}:")
                    if 'action' in step:
                        formatted.append(f"  Action: {step['action']}")
                    if 'observation' in step:
                        obs = str(step['observation'])
                        if len(obs) > 500:
                            obs = obs[:500] + "..."
                        formatted.append(f"  Observation: {obs}")
                    formatted.append("")
        elif isinstance(output, str):
            formatted.append("Output:")
            formatted.append(output[:3000] + "..." if len(output) > 3000 else output)
    
    return "\n".join(formatted)

def evaluate_with_o4mini(task, agent1_trace, agent2_trace, agent1_gif, agent2_gif, agent1_model, agent2_model):
    """Evaluate using o4-mini with both traces and GIFs"""
    
    # Format traces
    agent1_trace_formatted = format_vllm_trace(agent1_trace)
    agent2_trace_formatted = format_vllm_trace(agent2_trace)
    
    # Build messages
    messages = [
        {
            "role": "system",
            "content": """You are an expert evaluator of browser automation agents. 
You will analyze both the execution traces and visual GIF recordings to determine which agent better completed the task.

Evaluation criteria (in order of importance):
1. Task completion success - Did the agent successfully complete what was asked?
2. Correctness - Is the information/result accurate?
3. Efficiency - Did the agent take a reasonable path without too many unnecessary steps?
4. Error recovery - If errors occurred, did the agent handle them well?

The GIF recordings show the actual browser interactions and are the ground truth for what happened."""
        }
    ]
    
    # Build user message content
    user_content = [
        {
            "type": "text",
            "text": f"""Task: {task}

AGENT 1 ({agent1_model}) EXECUTION TRACE:
{agent1_trace_formatted[:2000]}{'...' if len(agent1_trace_formatted) > 2000 else ''}

AGENT 2 ({agent2_model}) EXECUTION TRACE:
{agent2_trace_formatted[:2000]}{'...' if len(agent2_trace_formatted) > 2000 else ''}

Now I'll show you the GIF recordings of both agents' actual browser interactions:"""
        }
    ]
    
    # Add GIFs if available
    gifs_added = 0
    
    if agent1_gif and Path(agent1_gif).exists():
        base64_gif = encode_image_to_base64(agent1_gif)
        if base64_gif:
            user_content.append({
                "type": "text",
                "text": "\nAGENT 1 GIF RECORDING:"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/gif;base64,{base64_gif}",
                    "detail": "high"
                }
            })
            gifs_added += 1
    
    if agent2_gif and Path(agent2_gif).exists():
        base64_gif = encode_image_to_base64(agent2_gif)
        if base64_gif:
            user_content.append({
                "type": "text", 
                "text": "\nAGENT 2 GIF RECORDING:"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/gif;base64,{base64_gif}",
                    "detail": "high"
                }
            })
            gifs_added += 1
    
    # Add final instruction
    user_content.append({
        "type": "text",
        "text": f"""\n{'Based on the execution traces and GIF recordings above' if gifs_added > 0 else 'Based on the execution traces above'}, evaluate which agent performed better.

Provide your evaluation in JSON format:
{{
    "preference": "Agent 1" or "Agent 2" or "Tie",
    "confidence": 0.0 to 1.0,
    "reasoning": "Detailed explanation referencing specific observations from {'both traces and GIFs' if gifs_added > 0 else 'the traces'}"
}}"""
    })
    
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    try:
        # Use o4-mini model
        response = client.chat.completions.create(
            model="o4-mini",
            messages=messages,
            max_completion_tokens=1000  # o4-mini uses max_completion_tokens
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                'preference': result.get('preference', 'Tie'),
                'confidence': float(result.get('confidence', 0.5)),
                'reasoning': result.get('reasoning', 'No reasoning provided'),
                'used_gifs': gifs_added
            }
    except Exception as e:
        print(f"Error in o4-mini evaluation: {e}")
        return {
            'preference': 'Tie',
            'confidence': 0.5,
            'reasoning': f'Error: {str(e)}',
            'used_gifs': gifs_added
        }
    
    # Default return if JSON parsing fails
    return {
        'preference': 'Tie',
        'confidence': 0.5,
        'reasoning': 'Failed to parse model response',
        'used_gifs': gifs_added
    }

# Run evaluations
results = []
print("\nEvaluating each question...")

# Check if we have existing results to continue from
existing_results = []
if Path('o4mini_evaluation_temp.csv').exists():
    existing_df = pd.read_csv('o4mini_evaluation_temp.csv')
    existing_results = existing_df.to_dict('records')
    results = existing_results
    print(f"Resuming from {len(existing_results)} existing results...")

for idx, row in full_mapping.iterrows():
    question = row['question']
    task = row['task']
    baseline_winner = row['baseline_winner']
    agent1_model = row['agent1_model']
    agent2_model = row['agent2_model']
    agent1_gif = row['agent1_gif_path']
    agent2_gif = row['agent2_gif_path']
    
    # Skip if already processed
    if any(r['question'] == question for r in results):
        print(f"\nSkipping {question} (already processed)")
        continue
    
    print(f"\n{question}: {task[:60]}...")
    
    # Load vLLM traces
    agent1_trace = load_vllm_trace(agent1_gif)
    agent2_trace = load_vllm_trace(agent2_gif)
    
    print(f"  Agent 1: Trace={'Found' if agent1_trace else 'Not found'}, GIF={'Found' if agent1_gif else 'Not found'}")
    print(f"  Agent 2: Trace={'Found' if agent2_trace else 'Not found'}, GIF={'Found' if agent2_gif else 'Not found'}")
    
    # Run evaluation
    eval_result = evaluate_with_o4mini(
        task, agent1_trace, agent2_trace,
        agent1_gif, agent2_gif,
        agent1_model, agent2_model
    )
    
    results.append({
        'question': question,
        'task': task,
        'baseline_winner': baseline_winner,
        'agent1_model': agent1_model,
        'agent2_model': agent2_model,
        'o4mini_preference': eval_result['preference'],
        'o4mini_confidence': eval_result['confidence'],
        'o4mini_reasoning': eval_result['reasoning'],
        'agrees_with_baseline': eval_result['preference'] == baseline_winner,
        'used_gifs': eval_result['used_gifs'],
        'had_traces': bool(agent1_trace or agent2_trace),
        'timestamp': datetime.now().isoformat()
    })
    
    print(f"  Baseline: {baseline_winner}")
    print(f"  o4-mini: {eval_result['preference']} (confidence: {eval_result['confidence']:.3f})")
    print(f"  Used {eval_result['used_gifs']} GIFs in evaluation")
    print(f"  Agrees with baseline: {'Yes' if eval_result['preference'] == baseline_winner else 'No'}")
    
    # Rate limiting
    time.sleep(2)
    
    # Save intermediate results
    if len(results) % 5 == 0:
        temp_df = pd.DataFrame(results)
        temp_df.to_csv('o4mini_evaluation_temp.csv', index=False)
        print(f"  Saved {len(results)} results...")

# Save final results
results_df = pd.DataFrame(results)
results_df.to_csv('o4mini_evaluation_final.csv', index=False)

print(f"\n=== EVALUATION COMPLETE ===")
print(f"Total evaluations: {len(results_df)}")
print(f"Used GIFs: {results_df[results_df['used_gifs'] > 0].shape[0]} evaluations")
print(f"Average GIFs per evaluation: {results_df['used_gifs'].mean():.1f}")
print(f"Had vLLM traces: {results_df['had_traces'].sum()} evaluations")
print(f"Average confidence: {results_df['o4mini_confidence'].mean():.3f}")
print(f"Agreement with baseline: {results_df['agrees_with_baseline'].sum()}/{len(results_df)} ({results_df['agrees_with_baseline'].mean()*100:.1f}%)")

# Save mapping for scatter plot
mapping_df = results_df[['question', 'o4mini_confidence']].copy()
mapping_df.to_csv('question_to_o4mini_confidence.csv', index=False)