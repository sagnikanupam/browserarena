#!/usr/bin/env python3
"""
Evaluate browser agent interactions using GPT-4.1 to compare with human survey results.
"""

import os
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from openai import OpenAI
from datetime import datetime

# Set up OpenAI client
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

# Model to use
MODEL = "gpt-4-turbo-preview"  # GPT-4.1

def load_survey_interactions():
    """Load the survey interaction data from HTML snippets."""
    interactions = []
    snippets_dir = Path("survey_html_snippets")
    
    for html_file in sorted(snippets_dir.glob("interaction_*.html")):
        with open(html_file, 'r') as f:
            content = f.read()
            
        # Extract task and agent outputs from HTML
        task_start = content.find('<strong>Task:</strong>') + len('<strong>Task:</strong>')
        task_end = content.find('</div>', task_start)
        task = content[task_start:task_end].strip()
        
        # Extract Agent 1 output
        agent1_start = content.find('<div class="output-text">', content.find('Agent 1'))
        agent1_start += len('<div class="output-text">')
        agent1_end = content.find('</div>', agent1_start)
        agent1_output = content[agent1_start:agent1_end].replace('<br>', '\n')
        
        # Extract Agent 2 output
        agent2_start = content.find('<div class="output-text">', content.find('Agent 2'))
        agent2_start += len('<div class="output-text">')
        agent2_end = content.find('</div>', agent2_start)
        agent2_output = content[agent2_start:agent2_end].replace('<br>', '\n')
        
        # Extract GIF URLs
        gif1_start = content.find('src="', content.find('Agent 1'))
        gif1_start += len('src="')
        gif1_end = content.find('"', gif1_start)
        gif1_url = content[gif1_start:gif1_end]
        
        gif2_start = content.find('src="', content.find('Agent 2'))
        gif2_start += len('src="')
        gif2_end = content.find('"', gif2_start)
        gif2_url = content[gif2_start:gif2_end]
        
        interaction_id = html_file.stem
        
        interactions.append({
            'id': interaction_id,
            'task': task,
            'agent1_output': agent1_output,
            'agent2_output': agent2_output,
            'agent1_gif': gif1_url,
            'agent2_gif': gif2_url
        })
    
    return interactions

def create_evaluation_prompt(task: str, agent1_output: str, agent2_output: str) -> str:
    """Create the evaluation prompt for GPT-4.1."""
    prompt = f"""You are evaluating AI agents performing web browsing tasks. Given a task and the outputs from two agents, determine which agent performed better.

TASK: {task}

AGENT 1 OUTPUT:
{agent1_output}

AGENT 2 OUTPUT:
{agent2_output}

EVALUATION CRITERIA:
- Task completion: Did the agent successfully complete the requested task?
- Accuracy: Is the information provided correct and complete?
- Efficiency: How well did the agent navigate to find the information?
- Error handling: Did the agent recover from any errors encountered?

If both agents failed, choose the one that made more progress toward completing the task.

Based on the outputs above, which agent performed better?

Respond with ONLY one of these three options:
- "Left" if Agent 1 performed better
- "Right" if Agent 2 performed better  
- "Tie" if both performed equally well

Your response:"""
    
    return prompt

def evaluate_interaction(interaction: Dict) -> str:
    """Evaluate a single interaction using GPT-4.1."""
    prompt = create_evaluation_prompt(
        interaction['task'],
        interaction['agent1_output'],
        interaction['agent2_output']
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert evaluator of AI agent performance on web browsing tasks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=10
        )
        
        vote = response.choices[0].message.content.strip()
        
        # Remove quotes if present
        vote = vote.strip('"').strip("'")
        
        # Extract just the vote if it's part of a longer response
        if "Left" in vote and "Right" not in vote and "Tie" not in vote:
            vote = "Left"
        elif "Right" in vote and "Left" not in vote and "Tie" not in vote:
            vote = "Right"
        elif "Tie" in vote and "Left" not in vote and "Right" not in vote:
            vote = "Tie"
        
        # Validate response
        if vote not in ["Left", "Right", "Tie"]:
            print(f"Invalid response: {vote}")
            return "Tie"  # Default to tie if invalid
            
        return vote
        
    except Exception as e:
        print(f"Error evaluating interaction: {e}")
        return "Tie"  # Default to tie on error

def load_human_results():
    """Load human survey results for comparison."""
    results_file = Path("agreement-results/agreement_analysis_results.csv")
    
    human_votes = {}
    question_map = {
        'Q1': 'interaction_01',
        'Q2': 'interaction_02', 
        'Q3': 'interaction_03',
        'Q4': 'interaction_04',
        'Q5': 'interaction_05',
        'Q6': 'interaction_06',
        'Q7': 'interaction_07',
        'Q8': 'interaction_08',
        'Q9': 'interaction_09',
        'Q10': 'interaction_10',
        'Q11': 'interaction_11',
        'Q12': 'interaction_12',
        'Q13': 'interaction_13',
        'Q16': 'interaction_14',
        'Q17': 'interaction_15',
        'Q18': 'interaction_16',
        'Q19': 'interaction_17',
        'Q20': 'interaction_18',
        'Q22': 'interaction_19'
    }
    
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row['question']
            if question in question_map:
                interaction_id = question_map[question]
                
                # Convert majority label to our format
                majority = row['majority_label']
                if majority == 'Agent 1':
                    majority = 'Left'
                elif majority == 'Agent 2':
                    majority = 'Right'
                else:
                    majority = 'Tie'
                
                human_votes[interaction_id] = {
                    'majority': majority,
                    'counts': {
                        'Left': int(row['agent_1_votes']),
                        'Right': int(row['agent_2_votes']),
                        'Tie': int(row['tie_votes'])
                    },
                    'total': int(row['n_responses'])
                }
    
    return human_votes

def main():
    """Run the GPT-4.1 evaluation and compare with human results."""
    print("Loading survey interactions...")
    interactions = load_survey_interactions()
    print(f"Loaded {len(interactions)} interactions")
    
    print("\nLoading human survey results...")
    human_results = load_human_results()
    print(f"Loaded human results for {len(human_results)} interactions")
    
    print(f"\nEvaluating interactions with {MODEL}...")
    gpt_results = {}
    
    for i, interaction in enumerate(interactions):
        print(f"Evaluating {interaction['id']} ({i+1}/{len(interactions)})...")
        vote = evaluate_interaction(interaction)
        gpt_results[interaction['id']] = vote
        
        # Rate limiting
        time.sleep(1)
    
    # Save GPT results
    results_data = {
        'model': MODEL,
        'timestamp': datetime.now().isoformat(),
        'evaluations': gpt_results
    }
    
    with open('gpt4_evaluation_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Compare with human results
    print("\n\nComparison with Human Results:")
    print("=" * 60)
    
    agreement_count = 0
    total_compared = 0
    
    comparison_results = []
    
    for interaction_id, gpt_vote in gpt_results.items():
        if interaction_id in human_results:
            human_vote = human_results[interaction_id]['majority']
            agrees = gpt_vote == human_vote
            if agrees:
                agreement_count += 1
            total_compared += 1
            
            comparison_results.append({
                'interaction_id': interaction_id,
                'gpt_vote': gpt_vote,
                'human_majority': human_vote,
                'human_counts': human_results[interaction_id]['counts'],
                'agrees': agrees
            })
            
            print(f"{interaction_id}: GPT-4={gpt_vote}, Human={human_vote} {'✓' if agrees else '✗'}")
    
    # Calculate agreement rate
    if total_compared > 0:
        agreement_rate = agreement_count / total_compared
        print(f"\nOverall Agreement Rate: {agreement_rate:.2%} ({agreement_count}/{total_compared})")
    
    # Save comparison results
    with open('gpt4_human_comparison.json', 'w') as f:
        json.dump({
            'comparison_results': comparison_results,
            'agreement_rate': agreement_rate if total_compared > 0 else 0,
            'agreement_count': agreement_count,
            'total_compared': total_compared
        }, f, indent=2)
    
    # Create summary CSV
    with open('gpt4_evaluation_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Interaction_ID', 'Task', 'GPT4_Vote', 'Human_Majority', 'Human_Left', 'Human_Right', 'Human_Tie', 'Agrees'])
        
        for interaction in interactions:
            int_id = interaction['id']
            if int_id in gpt_results and int_id in human_results:
                writer.writerow([
                    int_id,
                    interaction['task'][:100] + '...' if len(interaction['task']) > 100 else interaction['task'],
                    gpt_results[int_id],
                    human_results[int_id]['majority'],
                    human_results[int_id]['counts']['Left'],
                    human_results[int_id]['counts']['Right'],
                    human_results[int_id]['counts']['Tie'],
                    'Yes' if gpt_results[int_id] == human_results[int_id]['majority'] else 'No'
                ])

if __name__ == "__main__":
    main()