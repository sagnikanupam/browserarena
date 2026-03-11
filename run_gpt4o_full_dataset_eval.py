#!/usr/bin/env python3
"""
Run GPT-4o evaluations on all browser task completions where both agents succeeded
from the full dataset. This script evaluates pairs of agent trajectories using GPT-4o.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key='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')
from typing import Dict, List, Tuple
import pandas as pd
import glob

# Set up OpenAI client
# openai.api_key = os.environ.get('OPENAI_API_KEY')
# if not openai.api_key:
#     raise ValueError("Please set OPENAI_API_KEY environment variable")

def find_gif_files(left_task: str, right_task: str) -> Tuple[str, str]:
    """Find the GIF files for given task IDs"""

    # Search directories
    search_dirs = [
        Path('FastChat/gifs'),
        Path('browserarena-gifs'),
        Path('verified-gifs-only'),
        Path('FastChat')
    ]

    left_gif = None
    right_gif = None

    for search_dir in search_dirs:
        if search_dir.exists():
            # Look for exact matches
            left_matches = list(search_dir.glob(f'*{left_task}*.gif'))
            right_matches = list(search_dir.glob(f'*{right_task}*.gif'))

            if left_matches:
                left_gif = str(left_matches[0])
            if right_matches:
                right_gif = str(right_matches[0])

            if left_gif and right_gif:
                break

    return left_gif, right_gif

def find_prompts_and_outputs(task_id: str) -> Dict:
    """Find prompt and output data for a task"""

    # Search in prompts_and_outputs directory
    prompt_dir = Path('FastChat/prompts_and_outputs')
    if prompt_dir.exists():
        matches = list(prompt_dir.glob(f'*{task_id}*.json'))
        if matches:
            with open(matches[0], 'r') as f:
                return json.load(f)

    return None

def evaluate_with_gpt4o(left_task: str, right_task: str, task_data: Dict = None) -> Dict:
    """
    Evaluate two agent trajectories using GPT-4o
    Returns evaluation result with reasoning and preference
    """

    # Extract task description if available
    task_description = ""
    if task_data:
        if 'task' in task_data:
            task_description = task_data['task']
        elif 'prompt' in task_data:
            task_description = task_data['prompt']

    # Find GIF files
    left_gif, right_gif = find_gif_files(left_task, right_task)

    if not left_gif or not right_gif:
        return {
            'preference': 'error',
            'reasoning': f'Could not find GIF files for {left_task} and/or {right_task}',
            'error': 'missing_files'
        }

    # Load prompt/output data if available
    left_data = find_prompts_and_outputs(left_task)
    right_data = find_prompts_and_outputs(right_task)

    # Prepare the evaluation prompt
    prompt = f"""You are evaluating two browser automation agents completing the same task.

Task: {task_description if task_description else "Complete the given browser task"}

Please analyze both trajectories based on:
1. Task completion success
2. Efficiency (fewer unnecessary steps)
3. Correctness of the final result
4. Overall quality of execution

Important: Both agents have been marked as successfully completing the task, so focus on quality and efficiency differences.

Provide your evaluation in the following JSON format:
{{
    "preference": "left" or "right" or "tie",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your decision",
    "left_analysis": "Analysis of left agent's performance",
    "right_analysis": "Analysis of right agent's performance"
}}"""

    try:
        # Use the Chat Completions API with GPT-4o
        messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator of browser automation agents. Analyze the provided information and make a fair judgment."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]

        # Add trajectory information if available
        if left_data:
            messages.append({
                "role": "user",
                "content": f"Left agent output:\n{json.dumps(left_data.get('output', 'No output data'), indent=2)[:1000]}..."
            })

        if right_data:
            messages.append({
                "role": "user",
                "content": f"Right agent output:\n{json.dumps(right_data.get('output', 'No output data'), indent=2)[:1000]}..."
            })

        response = client.chat.completions.create(model="gpt-4-1106-preview",  # GPT-4o
        messages=messages,
        temperature=0.1,
        max_tokens=500)

        # Parse the response
        content = response.choices[0].message.content

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
            "reasoning": f"Error during evaluation: {str(e)}",
            "error": str(e)
        }

    return evaluation

def main():
    """Main evaluation pipeline"""

    print("Loading cases from full dataset...")

    # Load all cases
    with open('full_dataset_cases.json', 'r') as f:
        data = json.load(f)

    # Filter for cases where both succeeded
    successful_cases = [c for c in data['all_cases'] if c['both_succeeded']]

    print(f"Found {len(successful_cases)} cases where both agents succeeded")

    # Prepare output
    output_dir = Path('gpt4o_full_dataset_evaluations')
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / 'evaluation_results.json'

    # Load existing results if any
    if results_file.exists():
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []

    # Track evaluated cases
    evaluated_cases = set(r['case_id'] for r in all_results)

    # Process each case
    new_evaluations = 0
    errors = 0

    for i, case in enumerate(successful_cases):
        case_id = f"{case['left_task']}_{case['right_task']}"

        if case_id in evaluated_cases:
            print(f"Skipping {i+1}/{len(successful_cases)}: Already evaluated")
            continue

        print(f"\nEvaluating {i+1}/{len(successful_cases)}: {case_id}")

        try:
            # Get task data
            task_data = find_prompts_and_outputs(case['left_task'])

            # Run GPT-4o evaluation
            evaluation = evaluate_with_gpt4o(case['left_task'], case['right_task'], task_data)

            # Prepare result
            result = {
                'case_id': case_id,
                'response_id': case['response_id'],
                'left_task': case['left_task'],
                'right_task': case['right_task'],
                'original_vote': case['vote'],
                'gpt4o_preference': evaluation.get('preference', 'error'),
                'gpt4o_confidence': evaluation.get('confidence', 0),
                'gpt4o_reasoning': evaluation.get('reasoning', ''),
                'left_analysis': evaluation.get('left_analysis', ''),
                'right_analysis': evaluation.get('right_analysis', ''),
                'timestamp': datetime.now().isoformat(),
                'both_agents_successful': True
            }

            all_results.append(result)
            new_evaluations += 1

            # Save incrementally
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

            # Print result
            preference = evaluation.get('preference', 'error')
            if preference != 'error':
                agreement = 'Yes' if preference.lower() == case['vote'].lower() else 'No'
                print(f"  GPT-4o: {preference}, Original: {case['vote']}, Agreement: {agreement}")
            else:
                print(f"  Error: {evaluation.get('reasoning', 'Unknown error')}")
                errors += 1

            # Rate limiting
            time.sleep(2)  # Be conservative with API calls

        except Exception as e:
            print(f"  Error processing case: {str(e)}")
            errors += 1
            continue

        # Stop after a batch to avoid rate limits
        if new_evaluations >= 50:
            print(f"\nCompleted batch of {new_evaluations} evaluations. Run again to continue.")
            break

    # Final summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary:")
    print(f"  New evaluations: {new_evaluations}")
    print(f"  Total evaluations: {len(all_results)}")
    print(f"  Errors: {errors}")

    # Calculate agreement statistics
    valid_results = [r for r in all_results if r['gpt4o_preference'] not in ['error', '']]
    if valid_results:
        agreements = sum(1 for r in valid_results 
                        if r['gpt4o_preference'].lower() == r['original_vote'].lower())
        print(f"  Agreement with original votes: {agreements}/{len(valid_results)} ({agreements/len(valid_results)*100:.1f}%)")

    # Save summary CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_file = output_dir / 'evaluation_summary.csv'
        df.to_csv(csv_file, index=False)
        print(f"\nSummary saved to: {csv_file}")

if __name__ == "__main__":
    main()