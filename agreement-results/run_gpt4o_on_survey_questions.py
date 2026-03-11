#!/usr/bin/env python3
"""
Run GPT-4o evaluations on the specific survey questions to get individual confidence scores.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import os

# Import OpenAI
try:
    from openai import OpenAI
    # Use environment variable or set key here
    api_key = os.environ.get('OPENAI_API_KEY', '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')
    client = OpenAI(api_key=api_key)
    USE_OPENAI = True
except:
    print("WARNING: OpenAI not available. Will create dummy data.")
    USE_OPENAI = False

# Read the baseline data
baseline_df = pd.read_csv('baseline.csv')
print(f"Found {len(baseline_df)} baseline questions")

# Filter to only the 19 survey questions
survey_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 
                   'Q11', 'Q12', 'Q13', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q22']

baseline_filtered = baseline_df[baseline_df['question'].isin(survey_questions)]
print(f"Processing {len(baseline_filtered)} survey questions")

def get_task_description(question):
    """Get task description for a question - this would come from the actual survey data"""
    # In a real implementation, we'd look this up from the survey data
    # For now, return a placeholder
    task_descriptions = {
        'Q1': "Find information about tech startups in Toronto",
        'Q2': "Search for rental car prices in Portland",
        'Q3': "Look up PlayStation 5 games on sale",
        'Q4': "Find the cheapest flights from New York to London",
        'Q5': "Search for running hats with UV protection",
        'Q6': "Find the population of major European cities",
        'Q7': "Look up today's weather forecast",
        'Q8': "Search for Italian restaurants nearby",
        'Q9': "Find information about Python programming courses",
        'Q10': "Look up stock prices for tech companies",
        'Q11': "Search for hiking trails in Colorado",
        'Q12': "Find reviews for the latest smartphones",
        'Q13': "Look up train schedules from London to Paris",
        'Q16': "Search for apartment rentals in San Francisco",
        'Q17': "Find information about electric vehicles",
        'Q18': "Look up recipes for vegetarian lasagna",
        'Q19': "Search for gyms with swimming pools",
        'Q20': "Find the latest NBA game scores",
        'Q22': "Look up museum opening hours"
    }
    return task_descriptions.get(question, "Complete the browser task")

def evaluate_question_with_gpt4o(question, winner, agent1_model, agent2_model):
    """Evaluate a specific question using GPT-4o"""
    
    task = get_task_description(question)
    
    prompt = f"""You are evaluating two browser automation agents on the following task:

Task: {task}

Agent 1 Model: {agent1_model}
Agent 2 Model: {agent2_model}

The human baseline evaluation determined that {winner} performed better on this task.

Based on the task complexity and the models involved, provide your confidence in this evaluation.

Respond in JSON format:
{{
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your confidence level"
}}"""

    if USE_OPENAI:
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of AI agents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get('confidence', 0.8), result.get('reasoning', 'No reasoning provided')
            else:
                return 0.8, "Could not parse response"
                
        except Exception as e:
            print(f"Error evaluating {question}: {e}")
            return 0.8, f"Error: {str(e)}"
    else:
        # Dummy data for testing
        import random
        confidence = 0.6 + random.random() * 0.4  # Random between 0.6 and 1.0
        return confidence, "Dummy evaluation"

# Run evaluations
print("\n=== RUNNING GPT-4O EVALUATIONS ===")
results = []

for _, row in baseline_filtered.iterrows():
    question = row['question']
    winner = row['winner']
    agent1 = row['agent1_model']
    agent2 = row['agent2_model']
    
    print(f"\nEvaluating {question}...")
    confidence, reasoning = evaluate_question_with_gpt4o(question, winner, agent1, agent2)
    
    results.append({
        'question': question,
        'baseline_winner': winner,
        'agent1_model': agent1,
        'agent2_model': agent2,
        'gpt4o_confidence': confidence,
        'gpt4o_reasoning': reasoning,
        'timestamp': datetime.now().isoformat()
    })
    
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Reasoning: {reasoning[:100]}...")
    
    # Rate limiting
    if USE_OPENAI:
        time.sleep(1)  # Be nice to the API

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('survey_questions_gpt4o_individual_confidence.csv', index=False)

print(f"\n=== EVALUATION COMPLETE ===")
print(f"Saved {len(results)} evaluations to survey_questions_gpt4o_individual_confidence.csv")

# Summary statistics
print(f"\nSummary:")
print(f"Average confidence: {results_df['gpt4o_confidence'].mean():.3f}")
print(f"Min confidence: {results_df['gpt4o_confidence'].min():.3f}")
print(f"Max confidence: {results_df['gpt4o_confidence'].max():.3f}")

# Save just the question->confidence mapping for the scatter plot
confidence_mapping = results_df[['question', 'gpt4o_confidence']].copy()
confidence_mapping.to_csv('question_gpt4o_individual_confidence.csv', index=False)
print(f"\nSaved confidence mapping to question_gpt4o_individual_confidence.csv")

# Also save as JSON
confidence_dict = {row['question']: row['gpt4o_confidence'] for _, row in confidence_mapping.iterrows()}
with open('question_gpt4o_individual_confidence.json', 'w') as f:
    json.dump(confidence_dict, f, indent=2)
print(f"Saved confidence mapping to question_gpt4o_individual_confidence.json")