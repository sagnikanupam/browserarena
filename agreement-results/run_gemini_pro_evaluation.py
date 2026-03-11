#!/usr/bin/env python3
"""
Run Gemini Pro evaluation with GIFs and full vLLM traces.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import base64
import google.generativeai as genai
import os

# Set up Gemini client
# First check if API key is in env.sh
api_key = None
env_file = Path('../env.sh')
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            if 'GOOGLE_API_KEY' in line or 'GEMINI_API_KEY' in line:
                # Extract key from export GOOGLE_API_KEY=xxx format
                parts = line.strip().split('=', 1)
                if len(parts) == 2:
                    api_key = parts[1].strip().strip('"').strip("'")
                    break

if not api_key:
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')

if not api_key:
    raise ValueError("No Google/Gemini API key found. Please set GOOGLE_API_KEY or GEMINI_API_KEY")

genai.configure(api_key=api_key)

print("=== RUNNING GEMINI PRO EVALUATION WITH GIFS ===")

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
    """Encode image to base64 for Gemini"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def format_vllm_trace(trace_data):
    """Format vLLM trace for presentation to Gemini"""
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

def evaluate_with_gemini(task, agent1_trace, agent2_trace, agent1_gif, agent2_gif, agent1_model, agent2_model):
    """Evaluate using Gemini with both traces and GIFs"""
    
    # Format traces
    agent1_trace_formatted = format_vllm_trace(agent1_trace)
    agent2_trace_formatted = format_vllm_trace(agent2_trace)
    
    # Build full prompt
    prompt = f"""You are an expert evaluator of browser automation agents. 
You will analyze the execution traces to determine which agent better completed the task.

Evaluation criteria (in order of importance):
1. Task completion success - Did the agent successfully complete what was asked?
2. Correctness - Is the information/result accurate?
3. Efficiency - Did the agent take a reasonable path without too many unnecessary steps?
4. Error recovery - If errors occurred, did the agent handle them well?

Task: {task}

AGENT 1 ({agent1_model}) EXECUTION TRACE:
{agent1_trace_formatted[:2000]}{'...' if len(agent1_trace_formatted) > 2000 else ''}

AGENT 2 ({agent2_model}) EXECUTION TRACE:
{agent2_trace_formatted[:2000]}{'...' if len(agent2_trace_formatted) > 2000 else ''}

Based on the execution traces above, evaluate which agent performed better.

Provide your evaluation in JSON format:
{{
    "preference": "Agent 1" or "Agent 2" or "Tie",
    "confidence": 0.0 to 1.0,
    "reasoning": "Detailed explanation referencing specific observations from the traces"
}}"""
    
    try:
        # Use the simpler text generation API (without images for now)
        response = genai.generate_text(
            model='models/text-bison-001',
            prompt=prompt,
            temperature=0.1,
            max_output_tokens=1000,
        )
        
        content = response.result
        
        # Parse JSON
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                'preference': result.get('preference', 'Tie'),
                'confidence': float(result.get('confidence', 0.5)),
                'reasoning': result.get('reasoning', 'No reasoning provided'),
                'used_gifs': 0  # Text-only for now
            }
    except Exception as e:
        print(f"Error in Gemini evaluation: {e}")
        return {
            'preference': 'Tie',
            'confidence': 0.5,
            'reasoning': f'Error: {str(e)}',
            'used_gifs': 0
        }

# Run evaluations
results = []
print("\nEvaluating each question...")

for idx, row in full_mapping.iterrows():
    question = row['question']
    task = row['task']
    baseline_winner = row['baseline_winner']
    agent1_model = row['agent1_model']
    agent2_model = row['agent2_model']
    agent1_gif = row['agent1_gif_path']
    agent2_gif = row['agent2_gif_path']
    
    print(f"\n{question}: {task[:60]}...")
    
    # Load vLLM traces
    agent1_trace = load_vllm_trace(agent1_gif)
    agent2_trace = load_vllm_trace(agent2_gif)
    
    print(f"  Agent 1: Trace={'Found' if agent1_trace else 'Not found'}, GIF={'Found' if agent1_gif else 'Not found'}")
    print(f"  Agent 2: Trace={'Found' if agent2_trace else 'Not found'}, GIF={'Found' if agent2_gif else 'Not found'}")
    
    # Run evaluation
    eval_result = evaluate_with_gemini(
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
        'gemini_preference': eval_result['preference'],
        'gemini_confidence': eval_result['confidence'],
        'gemini_reasoning': eval_result['reasoning'],
        'agrees_with_baseline': eval_result['preference'] == baseline_winner,
        'used_gifs': eval_result['used_gifs'],
        'had_traces': bool(agent1_trace or agent2_trace),
        'timestamp': datetime.now().isoformat()
    })
    
    print(f"  Baseline: {baseline_winner}")
    print(f"  Gemini: {eval_result['preference']} (confidence: {eval_result['confidence']:.3f})")
    print(f"  Used {eval_result['used_gifs']} GIFs in evaluation")
    print(f"  Agrees with baseline: {'Yes' if eval_result['preference'] == baseline_winner else 'No'}")
    
    # Rate limiting
    time.sleep(2)
    
    # Save intermediate results
    if len(results) % 5 == 0:
        temp_df = pd.DataFrame(results)
        temp_df.to_csv('gemini_pro_evaluation_temp.csv', index=False)
        print(f"  Saved {len(results)} results...")

# Save final results
results_df = pd.DataFrame(results)
results_df.to_csv('gemini_pro_evaluation_final.csv', index=False)

print(f"\n=== EVALUATION COMPLETE ===")
print(f"Total evaluations: {len(results_df)}")
print(f"Used GIFs: {results_df[results_df['used_gifs'] > 0].shape[0]} evaluations")
print(f"Average GIFs per evaluation: {results_df['used_gifs'].mean():.1f}")
print(f"Had vLLM traces: {results_df['had_traces'].sum()} evaluations")
print(f"Average confidence: {results_df['gemini_confidence'].mean():.3f}")
print(f"Agreement with baseline: {results_df['agrees_with_baseline'].sum()}/{len(results_df)} ({results_df['agrees_with_baseline'].mean()*100:.1f}%)")

# Save mapping for scatter plot
mapping_df = results_df[['question', 'gemini_confidence']].copy()
mapping_df.to_csv('question_to_gemini_confidence.csv', index=False)