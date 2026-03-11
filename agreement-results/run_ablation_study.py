#!/usr/bin/env python3
"""
Run ablation study for GPT-4o and O4-mini models.
Tests different input combinations:
1. Full (GIFs + traces) - baseline
2. GIF-only (no traces)
3. Trace-only (no GIFs)
4. Truncated traces (first 1000 chars)
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import base64
from openai import OpenAI
import os

# Set up OpenAI client
api_key = os.getenv('OPENAI_API_KEY', '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')
client = OpenAI(api_key=api_key)

print("=== RUNNING ABLATION STUDY ===")

# Read the mappings
mapping_df = pd.read_csv('survey_questions_exact_mapping.csv')
gif_mapping_df = pd.read_csv('survey_question_gif_mapping.csv')

# Merge the mappings
full_mapping = mapping_df.merge(gif_mapping_df[['question', 'agent1_gif_path', 'agent2_gif_path']], on='question')
print(f"Loaded {len(full_mapping)} question mappings")

def load_vllm_trace(gif_path):
    """Load the full vLLM trace from the corresponding JSON file"""
    if not gif_path or pd.isna(gif_path):
        return None
    
    gif_name = Path(gif_path).stem
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
    """Encode image to base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def format_vllm_trace(trace_data, truncate=False):
    """Format vLLM trace for presentation"""
    if not trace_data:
        return "No trace data available"
    
    formatted = []
    
    if 'prompt' in trace_data:
        formatted.append(f"TASK: {trace_data['prompt']}")
        formatted.append("")
    
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
    
    result = "\n".join(formatted)
    
    if truncate and len(result) > 1000:
        result = result[:1000] + "\n\n[TRUNCATED]"
    
    return result

def evaluate_with_model(model_name, task, agent1_trace, agent2_trace, agent1_gif, agent2_gif, 
                       agent1_model, agent2_model, ablation_type="full"):
    """Evaluate using specified model with ablation configuration"""
    
    # Build system message
    system_message = """You are an expert evaluator of browser automation agents. 
Evaluation criteria (in order of importance):
1. Task completion success - Did the agent successfully complete what was asked?
2. Correctness - Is the information/result accurate?
3. Efficiency - Did the agent take a reasonable path without too many unnecessary steps?
4. Error recovery - If errors occurred, did the agent handle them well?"""
    
    # Build user message based on ablation type
    user_content = []
    
    if ablation_type in ["full", "trace_only", "truncated_trace"]:
        # Include traces
        truncate = (ablation_type == "truncated_trace")
        agent1_trace_formatted = format_vllm_trace(agent1_trace, truncate=truncate)
        agent2_trace_formatted = format_vllm_trace(agent2_trace, truncate=truncate)
        
        trace_text = f"""Task: {task}

AGENT 1 ({agent1_model}) EXECUTION TRACE:
{agent1_trace_formatted[:2000]}{'...' if len(agent1_trace_formatted) > 2000 else ''}

AGENT 2 ({agent2_model}) EXECUTION TRACE:
{agent2_trace_formatted[:2000]}{'...' if len(agent2_trace_formatted) > 2000 else ''}"""
        
        user_content.append({
            "type": "text",
            "text": trace_text
        })
    else:
        # GIF-only - just task description
        user_content.append({
            "type": "text",
            "text": f"Task: {task}\n\nAgent 1 Model: {agent1_model}\nAgent 2 Model: {agent2_model}"
        })
    
    if ablation_type in ["full", "gif_only"]:
        # Add GIFs
        user_content.append({
            "type": "text",
            "text": "\n\nNow I'll show you the GIF recordings of both agents' actual browser interactions:"
        })
        
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
    
    # Add evaluation instruction
    context_desc = {
        "full": "the execution traces and GIF recordings",
        "gif_only": "the GIF recordings",
        "trace_only": "the execution traces",
        "truncated_trace": "the truncated execution traces"
    }[ablation_type]
    
    user_content.append({
        "type": "text",
        "text": f"""\nBased on {context_desc} above, evaluate which agent performed better.

Provide your evaluation in JSON format:
{{
    "preference": "Agent 1" or "Agent 2" or "Tie",
    "confidence": 0.0 to 1.0,
    "reasoning": "Detailed explanation"
}}"""
    })
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]
    
    try:
        if model_name == "gpt-4o":
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
        else:  # o4-mini
            response = client.chat.completions.create(
                model="o4-mini",
                messages=messages,
                max_completion_tokens=500
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
                'reasoning': result.get('reasoning', 'No reasoning provided')
            }
    except Exception as e:
        print(f"Error in {model_name} evaluation: {e}")
    
    return {
        'preference': 'Tie',
        'confidence': 0.5,
        'reasoning': f'Error during evaluation'
    }

# Run ablation experiments
ablation_configs = [
    ("full", "Full (GIF + Trace)"),
    ("gif_only", "GIF Only"),
    ("trace_only", "Trace Only"),
    ("truncated_trace", "Truncated Trace")
]

models = ["gpt-4o", "o4-mini"]

# Store results for each configuration
all_results = {}

for ablation_type, ablation_name in ablation_configs:
    print(f"\n=== Running {ablation_name} ablation ===")
    
    for model in models:
        print(f"\nEvaluating with {model}...")
        results = []
        
        for idx, row in full_mapping.iterrows():
            question = row['question']
            task = row['task']
            baseline_winner = row['baseline_winner']
            agent1_model = row['agent1_model']
            agent2_model = row['agent2_model']
            agent1_gif = row['agent1_gif_path']
            agent2_gif = row['agent2_gif_path']
            
            print(f"\n{question}: {task[:60]}...")
            
            # Load traces
            agent1_trace = load_vllm_trace(agent1_gif)
            agent2_trace = load_vllm_trace(agent2_gif)
            
            # Run evaluation
            eval_result = evaluate_with_model(
                model, task, agent1_trace, agent2_trace,
                agent1_gif, agent2_gif,
                agent1_model, agent2_model,
                ablation_type=ablation_type
            )
            
            results.append({
                'question': question,
                'task': task,
                'baseline_winner': baseline_winner,
                'model_preference': eval_result['preference'],
                'model_confidence': eval_result['confidence'],
                'model_reasoning': eval_result['reasoning'],
                'agrees_with_baseline': eval_result['preference'] == baseline_winner,
                'ablation_type': ablation_type,
                'model': model
            })
            
            print(f"  {model}: {eval_result['preference']} (conf: {eval_result['confidence']:.3f})")
            print(f"  Agrees with baseline: {'Yes' if eval_result['preference'] == baseline_winner else 'No'}")
            
            # Rate limiting
            time.sleep(2)
            
            # Save intermediate results
            if len(results) % 5 == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(f'ablation/ablation_{model}_{ablation_type}_temp.csv', index=False)
        
        # Save final results for this configuration
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'ablation/ablation_{model}_{ablation_type}_final.csv', index=False)
        
        # Store in memory
        key = f"{model}_{ablation_type}"
        all_results[key] = results_df
        
        print(f"\nCompleted {model} {ablation_name}:")
        print(f"  Agreement with baseline: {results_df['agrees_with_baseline'].mean()*100:.1f}%")
        print(f"  Average confidence: {results_df['model_confidence'].mean():.3f}")

# Save combined results
combined_results = []
for key, df in all_results.items():
    combined_results.append(df)

combined_df = pd.concat(combined_results, ignore_index=True)
combined_df.to_csv('ablation/ablation_all_results.csv', index=False)

print("\n=== ABLATION STUDY COMPLETE ===")
print(f"Total evaluations: {len(combined_df)}")
print("\nResults saved in ablation/ directory")