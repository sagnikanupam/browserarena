#!/usr/bin/env python3
"""
Run GPT-4o evaluations on browser task completions where both agents succeeded.
This script evaluates pairs of agent trajectories using GPT-4o vision capabilities.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime
import base64
import requests
from typing import Dict, List, Tuple
import pandas as pd

# OpenAI API configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

def encode_image(image_path: str) -> str:
    """Encode image to base64 for API calls"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def evaluate_with_gpt4o(left_gif: str, right_gif: str, task_description: str = "") -> Dict:
    """
    Evaluate two agent trajectories using GPT-4o
    Returns evaluation result with reasoning and preference
    """
    
    # Prepare the prompt
    prompt = f"""You are evaluating two browser automation agents completing the same task.

Task: {task_description if task_description else "Complete the given browser task"}

Please analyze both trajectories and determine which agent performed better based on:
1. Task completion success
2. Efficiency (fewer unnecessary steps)
3. Correctness of the final result
4. Overall quality of execution

Provide your evaluation in the following JSON format:
{{
    "preference": "left" or "right" or "tie",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your decision",
    "left_analysis": "Analysis of left agent's performance",
    "right_analysis": "Analysis of right agent's performance"
}}
"""

    # API call to GPT-4o
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/gif;base64,{encode_image(left_gif)}"
                        }
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/gif;base64,{encode_image(right_gif)}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500,
        "temperature": 0.1
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    result = response.json()
    
    try:
        # Parse the JSON response
        content = result['choices'][0]['message']['content']
        # Extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            evaluation = json.loads(json_match.group())
        else:
            evaluation = {
                "preference": "error",
                "reasoning": "Failed to parse response",
                "raw_response": content
            }
    except Exception as e:
        evaluation = {
            "preference": "error",
            "reasoning": f"Error parsing response: {str(e)}",
            "raw_response": content if 'content' in locals() else str(result)
        }
    
    return evaluation

def find_gif_files(task_id: str, base_dir: Path) -> Tuple[str, str]:
    """Find the GIF files for a given task ID"""
    # Look for GIF files in various possible locations
    gif_dirs = [
        base_dir / "FastChat" / "gifs",
        base_dir / "browserarena-gifs",
        base_dir / "verified-gifs-only",
        base_dir
    ]
    
    left_gif = None
    right_gif = None
    
    for gif_dir in gif_dirs:
        if gif_dir.exists():
            # Search for files matching the task ID
            for f in gif_dir.glob(f"*{task_id}*.gif"):
                return str(f), None
    
    # If exact match not found, try to find by date pattern
    date_pattern = task_id.split('_')[0:3]  # Extract date part
    date_str = '_'.join(date_pattern)
    
    for gif_dir in gif_dirs:
        if gif_dir.exists():
            matching_files = list(gif_dir.glob(f"{date_str}*.gif"))
            if len(matching_files) >= 2:
                return str(matching_files[0]), str(matching_files[1])
    
    return None, None

def main():
    """Main evaluation pipeline"""
    
    # Load cases where both agents succeeded
    with open('both_agents_successful_cases.json', 'r') as f:
        data = json.load(f)
    
    successful_cases = data['both_agents_successful']
    print(f"Found {len(successful_cases)} cases where both agents succeeded")
    
    # Directory setup
    base_dir = Path('/Users/davisbrown/browserarena')
    output_dir = base_dir / 'gpt4o_evaluations'
    output_dir.mkdir(exist_ok=True)
    
    # Check existing evaluations
    existing_evals = set()
    eval_results_file = output_dir / 'evaluation_results.json'
    if eval_results_file.exists():
        with open(eval_results_file, 'r') as f:
            existing_results = json.load(f)
            existing_evals = set(r['case_id'] for r in existing_results)
    else:
        existing_results = []
    
    # Process each case
    new_evaluations = 0
    for i, case in enumerate(successful_cases):
        case_id = f"{case['left_task']}_{case['right_task']}"
        
        if case_id in existing_evals:
            print(f"Skipping {i+1}/{len(successful_cases)}: {case_id} (already evaluated)")
            continue
        
        print(f"\nProcessing {i+1}/{len(successful_cases)}: {case_id}")
        
        # Find GIF files
        left_gif, _ = find_gif_files(case['left_task'], base_dir)
        right_gif, _ = find_gif_files(case['right_task'], base_dir)
        
        if not left_gif:
            left_gif, _ = find_gif_files(case['left_task'], base_dir)
        if not right_gif:
            right_gif, _ = find_gif_files(case['right_task'], base_dir)
        
        if not left_gif or not right_gif:
            print(f"  Warning: Could not find GIF files for {case_id}")
            continue
        
        try:
            # Run GPT-4o evaluation
            print(f"  Evaluating with GPT-4o...")
            evaluation = evaluate_with_gpt4o(left_gif, right_gif)
            
            # Store result
            result = {
                'case_id': case_id,
                'left_task': case['left_task'],
                'right_task': case['right_task'],
                'human_vote': case['vote'],
                'gpt4o_preference': evaluation.get('preference', 'error'),
                'gpt4o_confidence': evaluation.get('confidence', 0),
                'gpt4o_reasoning': evaluation.get('reasoning', ''),
                'left_analysis': evaluation.get('left_analysis', ''),
                'right_analysis': evaluation.get('right_analysis', ''),
                'timestamp': datetime.now().isoformat()
            }
            
            existing_results.append(result)
            new_evaluations += 1
            
            # Save results incrementally
            with open(eval_results_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
            
            print(f"  GPT-4o: {evaluation.get('preference', 'error')} (confidence: {evaluation.get('confidence', 0):.2f})")
            print(f"  Human: {case['vote']}")
            print(f"  Agreement: {'✓' if evaluation.get('preference', '').lower() == case['vote'].lower() else '✗'}")
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"  Error evaluating {case_id}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"New evaluations: {new_evaluations}")
    print(f"Total evaluations: {len(existing_results)}")
    
    # Calculate agreement statistics
    if existing_results:
        agreements = sum(1 for r in existing_results 
                        if r['gpt4o_preference'].lower() == r['human_vote'].lower())
        agreement_rate = agreements / len(existing_results)
        print(f"GPT-4o vs Human agreement rate: {agreement_rate:.2%}")

if __name__ == "__main__":
    main()