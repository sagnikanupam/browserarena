#!/usr/bin/env python3
"""
Create a CSV summary of the verified interactions
"""

import json
import csv
import re
from typing import Dict, List, Any

def extract_gif_id(message_content: str) -> str:
    """Extract GIF ID from message content using regex"""
    gif_pattern = r'<img src="/gradio_api/file=gifs/([^"]+\.gif)"'
    match = re.search(gif_pattern, message_content)
    return match.group(1) if match else "NOT_FOUND"

def extract_log_id(message_content: str) -> str:
    """Extract Log ID from message content"""
    log_pattern = r'Log ID:\s*([^\s\n]+)'
    match = re.search(log_pattern, message_content)
    return match.group(1) if match else "NOT_FOUND"

def determine_success_status(message_content: str) -> str:
    """Determine if the task succeeded or failed based on message content"""
    failure_indicators = [
        "failed", "error", "couldn't", "unable", "unsuccessful", 
        "not working", "didn't work", "exception", "timeout"
    ]
    
    success_indicators = [
        "successfully", "completed", "done", "finished", "success"
    ]
    
    content_lower = message_content.lower()
    
    for indicator in failure_indicators:
        if indicator in content_lower:
            return "FAILED"
    
    for indicator in success_indicators:
        if indicator in content_lower:
            return "SUCCESS"
    
    return "UNCLEAR"

def extract_user_task(messages: List[List]) -> str:
    """Extract the user task from the messages"""
    for message in messages:
        if isinstance(message, list) and len(message) >= 2:
            role = message[0]
            content = message[1]
            if role.lower() in ["human", "user"]:
                return content.strip()
    return "NOT_FOUND"

def main():
    """Main function to process the JSON file and create a CSV summary"""
    input_file = "/Users/davisbrown/browserarena/FastChat/verified_conv_log/verified_interactions-clean.json"
    output_file = "/Users/davisbrown/browserarena/verified_interactions_summary.csv"
    
    print("Loading JSON file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} interactions")
    
    # Prepare CSV data
    csv_data = []
    
    for i, interaction in enumerate(data, 1):
        vote_type = interaction.get("type", "UNKNOWN")
        states = interaction.get("states", [])
        
        if len(states) != 2:
            continue
        
        # Extract user task
        user_task = extract_user_task(states[0].get("messages", []))
        
        # Process both models
        for side, state in enumerate(states):
            model_name = state.get("model_name", "UNKNOWN")
            gif_id = "NOT_FOUND"
            log_id = "NOT_FOUND"
            status = "UNCLEAR"
            
            # Find the last assistant message
            for message in reversed(state.get("messages", [])):
                if isinstance(message, list) and len(message) >= 2:
                    role = message[0]
                    content = message[1]
                    if role.lower() in ["assistant", "model"]:
                        gif_id = extract_gif_id(content)
                        log_id = extract_log_id(content)
                        status = determine_success_status(content)
                        break
            
            csv_data.append({
                'interaction_num': i,
                'vote_type': vote_type,
                'user_task': user_task[:200] + '...' if len(user_task) > 200 else user_task,
                'side': 'LEFT' if side == 0 else 'RIGHT',
                'model_name': model_name,
                'gif_id': gif_id,
                'log_id': log_id,
                'status': status
            })
    
    # Write to CSV
    print("Writing CSV file...")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['interaction_num', 'vote_type', 'user_task', 'side', 'model_name', 'gif_id', 'log_id', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
    
    print(f"CSV file created: {output_file}")
    print(f"Total rows: {len(csv_data)}")

if __name__ == "__main__":
    main()