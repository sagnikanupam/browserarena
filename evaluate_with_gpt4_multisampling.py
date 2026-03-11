#!/usr/bin/env python3
"""
Evaluate browser agent interactions using GPT-4.1 with multiple samples
and compare with both human survey results and baseline results.
"""

import os
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
from openai import OpenAI
from datetime import datetime

# Set up OpenAI client
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

# Model to use
MODEL = "gpt-4-turbo-preview"  # GPT-4.1
NUM_SAMPLES = 5  # Number of times to sample each interaction

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

def evaluate_interaction(interaction: Dict, sample_num: int) -> str:
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
            temperature=0.7,  # Higher temperature for sampling diversity
            max_tokens=10,
            seed=sample_num  # Use sample number as seed for reproducibility
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
            print(f"Invalid response in sample {sample_num}: {vote}")
            return "Tie"  # Default to tie if invalid
            
        return vote
        
    except Exception as e:
        print(f"Error evaluating interaction (sample {sample_num}): {e}")
        return "Tie"  # Default to tie on error

def load_human_survey_results():
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

def load_baseline_results():
    """Load baseline results for comparison."""
    results_file = Path("agreement-results/baseline.csv")
    
    baseline_votes = {}
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
                
                # Convert winner to our format
                winner = row['winner']
                if winner == 'Agent 1':
                    vote = 'Left'
                elif winner == 'Agent 2':
                    vote = 'Right'
                else:
                    vote = 'Tie'
                
                baseline_votes[interaction_id] = {
                    'vote': vote,
                    'agent1_model': row['agent1_model'],
                    'agent2_model': row['agent2_model']
                }
    
    return baseline_votes

def calculate_consistency(votes: List[str]) -> Dict:
    """Calculate consistency metrics for a list of votes."""
    vote_counts = Counter(votes)
    total_votes = len(votes)
    
    # Calculate entropy as a measure of disagreement
    entropy = 0
    for count in vote_counts.values():
        if count > 0:
            p = count / total_votes
            entropy -= p * np.log2(p)
    
    # Normalize entropy to 0-1 range (max entropy for 3 choices is log2(3))
    max_entropy = np.log2(3)
    normalized_entropy = entropy / max_entropy
    
    # Calculate agreement rate (how often votes agree with majority)
    majority_vote = vote_counts.most_common(1)[0][0]
    agreement_rate = vote_counts[majority_vote] / total_votes
    
    return {
        'vote_counts': dict(vote_counts),
        'majority_vote': majority_vote,
        'agreement_rate': agreement_rate,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'consistency_score': 1 - normalized_entropy  # Higher is more consistent
    }

def main():
    """Run the GPT-4.1 evaluation with multiple samples and compare with both human datasets."""
    print("Loading survey interactions...")
    interactions = load_survey_interactions()
    print(f"Loaded {len(interactions)} interactions")
    
    print("\nLoading human survey results...")
    survey_results = load_human_survey_results()
    print(f"Loaded survey results for {len(survey_results)} interactions")
    
    print("\nLoading baseline results...")
    baseline_results = load_baseline_results()
    print(f"Loaded baseline results for {len(baseline_results)} interactions")
    
    print(f"\nEvaluating interactions with {MODEL} ({NUM_SAMPLES} samples each)...")
    gpt_results = {}
    all_samples = {}
    
    for i, interaction in enumerate(interactions):
        int_id = interaction['id']
        print(f"\nEvaluating {int_id} ({i+1}/{len(interactions)})...")
        
        samples = []
        for sample_num in range(NUM_SAMPLES):
            print(f"  Sample {sample_num + 1}/{NUM_SAMPLES}...", end='')
            vote = evaluate_interaction(interaction, sample_num)
            samples.append(vote)
            print(f" {vote}")
            time.sleep(0.5)  # Rate limiting
        
        all_samples[int_id] = samples
        consistency = calculate_consistency(samples)
        gpt_results[int_id] = {
            'samples': samples,
            'consistency': consistency
        }
    
    # Save detailed results
    results_data = {
        'model': MODEL,
        'num_samples': NUM_SAMPLES,
        'timestamp': datetime.now().isoformat(),
        'evaluations': gpt_results
    }
    
    with open('gpt4_multisampling_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Compare with both human datasets
    print("\n\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    comparison_results = []
    
    # Agreement counters
    survey_agreement_by_sample = [0] * NUM_SAMPLES
    baseline_agreement_by_sample = [0] * NUM_SAMPLES
    survey_majority_agreement = 0
    baseline_majority_agreement = 0
    
    total_survey_compared = 0
    total_baseline_compared = 0
    
    for int_id in sorted(gpt_results.keys()):
        result = gpt_results[int_id]
        samples = result['samples']
        majority = result['consistency']['majority_vote']
        consistency_score = result['consistency']['consistency_score']
        
        row = {
            'interaction_id': int_id,
            'gpt_samples': samples,
            'gpt_majority': majority,
            'consistency_score': consistency_score
        }
        
        # Compare with survey results
        if int_id in survey_results:
            survey_majority = survey_results[int_id]['majority']
            row['survey_majority'] = survey_majority
            row['survey_counts'] = survey_results[int_id]['counts']
            
            # Check agreement for each sample
            for j, sample_vote in enumerate(samples):
                if sample_vote == survey_majority:
                    survey_agreement_by_sample[j] += 1
            
            # Check majority agreement
            if majority == survey_majority:
                survey_majority_agreement += 1
            
            total_survey_compared += 1
        
        # Compare with baseline results
        if int_id in baseline_results:
            baseline_vote = baseline_results[int_id]['vote']
            row['baseline_vote'] = baseline_vote
            row['baseline_models'] = f"{baseline_results[int_id]['agent1_model']} vs {baseline_results[int_id]['agent2_model']}"
            
            # Check agreement for each sample
            for j, sample_vote in enumerate(samples):
                if sample_vote == baseline_vote:
                    baseline_agreement_by_sample[j] += 1
            
            # Check majority agreement
            if majority == baseline_vote:
                baseline_majority_agreement += 1
            
            total_baseline_compared += 1
        
        comparison_results.append(row)
        
        # Print detailed comparison
        print(f"\n{int_id}:")
        print(f"  GPT-4 samples: {samples}")
        print(f"  GPT-4 majority: {majority} (consistency: {consistency_score:.2f})")
        
        if int_id in survey_results:
            print(f"  Survey majority: {survey_majority} {'✓' if majority == survey_majority else '✗'}")
        
        if int_id in baseline_results:
            print(f"  Baseline vote: {baseline_vote} {'✓' if majority == baseline_vote else '✗'}")
    
    # Calculate and display agreement rates
    print("\n\n" + "="*80)
    print("AGREEMENT ANALYSIS")
    print("="*80)
    
    print("\nAgreement with Human Survey:")
    print(f"  Majority vote agreement: {survey_majority_agreement}/{total_survey_compared} = {survey_majority_agreement/total_survey_compared:.2%}")
    print("  Per-sample agreement rates:")
    for i in range(NUM_SAMPLES):
        rate = survey_agreement_by_sample[i] / total_survey_compared
        print(f"    Sample {i+1}: {survey_agreement_by_sample[i]}/{total_survey_compared} = {rate:.2%}")
    
    print("\nAgreement with Baseline:")
    print(f"  Majority vote agreement: {baseline_majority_agreement}/{total_baseline_compared} = {baseline_majority_agreement/total_baseline_compared:.2%}")
    print("  Per-sample agreement rates:")
    for i in range(NUM_SAMPLES):
        rate = baseline_agreement_by_sample[i] / total_baseline_compared
        print(f"    Sample {i+1}: {baseline_agreement_by_sample[i]}/{total_baseline_compared} = {rate:.2%}")
    
    # Analyze GPT-4 consistency
    print("\n\nGPT-4 Self-Consistency Analysis:")
    consistency_scores = [r['consistency']['consistency_score'] for r in gpt_results.values()]
    print(f"  Average consistency score: {np.mean(consistency_scores):.3f}")
    print(f"  Min consistency: {np.min(consistency_scores):.3f}")
    print(f"  Max consistency: {np.max(consistency_scores):.3f}")
    
    # Count how often GPT-4 was unanimous
    unanimous_count = sum(1 for r in gpt_results.values() if r['consistency']['agreement_rate'] == 1.0)
    print(f"  Unanimous decisions: {unanimous_count}/{len(gpt_results)} = {unanimous_count/len(gpt_results):.2%}")
    
    # Save comparison summary
    with open('gpt4_multisampling_comparison.json', 'w') as f:
        json.dump({
            'comparison_results': comparison_results,
            'survey_agreement': {
                'majority_agreement_rate': survey_majority_agreement/total_survey_compared if total_survey_compared > 0 else 0,
                'majority_agreement_count': survey_majority_agreement,
                'per_sample_agreement': [survey_agreement_by_sample[i]/total_survey_compared for i in range(NUM_SAMPLES)],
                'total_compared': total_survey_compared
            },
            'baseline_agreement': {
                'majority_agreement_rate': baseline_majority_agreement/total_baseline_compared if total_baseline_compared > 0 else 0,
                'majority_agreement_count': baseline_majority_agreement,
                'per_sample_agreement': [baseline_agreement_by_sample[i]/total_baseline_compared for i in range(NUM_SAMPLES)],
                'total_compared': total_baseline_compared
            },
            'consistency_analysis': {
                'mean_consistency': float(np.mean(consistency_scores)),
                'min_consistency': float(np.min(consistency_scores)),
                'max_consistency': float(np.max(consistency_scores)),
                'unanimous_rate': unanimous_count/len(gpt_results)
            }
        }, f, indent=2)
    
    # Create detailed CSV
    with open('gpt4_multisampling_detailed.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Interaction_ID', 'Task', 'GPT4_Sample1', 'GPT4_Sample2', 'GPT4_Sample3', 'GPT4_Sample4', 'GPT4_Sample5',
                        'GPT4_Majority', 'Consistency_Score', 'Survey_Majority', 'Survey_Agreement',
                        'Baseline_Vote', 'Baseline_Agreement'])
        
        for interaction in interactions:
            int_id = interaction['id']
            if int_id in gpt_results:
                samples = gpt_results[int_id]['samples']
                majority = gpt_results[int_id]['consistency']['majority_vote']
                consistency = gpt_results[int_id]['consistency']['consistency_score']
                
                survey_maj = survey_results.get(int_id, {}).get('majority', 'N/A')
                survey_agree = 'Yes' if survey_maj == majority else 'No' if survey_maj != 'N/A' else 'N/A'
                
                baseline_vote = baseline_results.get(int_id, {}).get('vote', 'N/A')
                baseline_agree = 'Yes' if baseline_vote == majority else 'No' if baseline_vote != 'N/A' else 'N/A'
                
                writer.writerow([
                    int_id,
                    interaction['task'][:80] + '...' if len(interaction['task']) > 80 else interaction['task'],
                    *samples,
                    majority,
                    f"{consistency:.3f}",
                    survey_maj,
                    survey_agree,
                    baseline_vote,
                    baseline_agree
                ])

if __name__ == "__main__":
    main()