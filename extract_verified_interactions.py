#!/usr/bin/env python3
"""
Extract all verified interactions with GIF IDs from the JSON file
"""

import json
import re
from typing import Dict, List, Any

def extract_gif_id(message_content: str) -> str:
    """Extract GIF ID from message content using regex"""
    # Pattern: <img src="/gradio_api/file=gifs/DD_MM_YYYY_HH_MM_SS_XXX.gif" alt="GIF" />
    gif_pattern = r'<img src="/gradio_api/file=gifs/([^"]+\.gif)"'
    match = re.search(gif_pattern, message_content)
    return match.group(1) if match else "NOT_FOUND"

def extract_log_id(message_content: str) -> str:
    """Extract Log ID from message content"""
    # Pattern: Log ID: DD_MM_YYYY_HH_MM_SS_XXX
    log_pattern = r'Log ID:\s*([^\s\n]+)'
    match = re.search(log_pattern, message_content)
    return match.group(1) if match else "NOT_FOUND"

def determine_success_status(message_content: str) -> str:
    """Determine if the task succeeded or failed based on message content"""
    # Look for failure indicators
    failure_indicators = [
        "failed", "error", "couldn't", "unable", "unsuccessful", 
        "not working", "didn't work", "exception", "timeout"
    ]
    
    success_indicators = [
        "successfully", "completed", "done", "finished", "success"
    ]
    
    content_lower = message_content.lower()
    
    # Check for explicit failure indicators
    for indicator in failure_indicators:
        if indicator in content_lower:
            return "FAILED"
    
    # Check for success indicators
    for indicator in success_indicators:
        if indicator in content_lower:
            return "SUCCESS"
    
    # If neither clear success nor failure, mark as UNCLEAR
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

def process_interaction(interaction: Dict, interaction_num: int) -> Dict:
    """Process a single interaction and extract all required information"""
    result = {
        "interaction_num": interaction_num,
        "vote_type": interaction.get("type", "UNKNOWN"),
        "user_task": "NOT_FOUND",
        "left_model": {},
        "right_model": {}
    }
    
    states = interaction.get("states", [])
    if len(states) != 2:
        print(f"Warning: Interaction {interaction_num} has {len(states)} states instead of 2")
        return result
    
    # Extract user task from the first state's messages
    if states[0].get("messages"):
        result["user_task"] = extract_user_task(states[0]["messages"])
    
    # Process left model (states[0])
    left_state = states[0]
    result["left_model"] = {
        "model_name": left_state.get("model_name", "UNKNOWN"),
        "gif_id": "NOT_FOUND",
        "log_id": "NOT_FOUND",
        "status": "UNCLEAR"
    }
    
    # Find the last assistant message in left model
    for message in reversed(left_state.get("messages", [])):
        if isinstance(message, list) and len(message) >= 2:
            role = message[0]
            content = message[1]
            if role.lower() in ["assistant", "model"]:
                result["left_model"]["gif_id"] = extract_gif_id(content)
                result["left_model"]["log_id"] = extract_log_id(content)
                result["left_model"]["status"] = determine_success_status(content)
                break
    
    # Process right model (states[1])
    right_state = states[1]
    result["right_model"] = {
        "model_name": right_state.get("model_name", "UNKNOWN"),
        "gif_id": "NOT_FOUND",
        "log_id": "NOT_FOUND",
        "status": "UNCLEAR"
    }
    
    # Find the last assistant message in right model
    for message in reversed(right_state.get("messages", [])):
        if isinstance(message, list) and len(message) >= 2:
            role = message[0]
            content = message[1]
            if role.lower() in ["assistant", "model"]:
                result["right_model"]["gif_id"] = extract_gif_id(content)
                result["right_model"]["log_id"] = extract_log_id(content)
                result["right_model"]["status"] = determine_success_status(content)
                break
    
    return result

def main():
    """Main function to process the JSON file and extract all interactions"""
    input_file = "/Users/davisbrown/browserarena/FastChat/verified_conv_log/verified_interactions-clean.json"
    output_file = "/Users/davisbrown/browserarena/extracted_verified_interactions.txt"
    
    print("Loading JSON file...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    print(f"Found {len(data)} interactions")
    
    # Process each interaction
    results = []
    for i, interaction in enumerate(data, 1):
        print(f"Processing interaction {i}/{len(data)}")
        result = process_interaction(interaction, i)
        results.append(result)
    
    # Write results to output file
    print("Writing results to output file...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EXTRACTED VERIFIED INTERACTIONS - ALL 126 INTERACTIONS\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"INTERACTION #{result['interaction_num']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Vote Type: {result['vote_type']}\n")
            f.write(f"User Task: {result['user_task'][:200]}{'...' if len(result['user_task']) > 200 else ''}\n\n")
            
            f.write("LEFT MODEL:\n")
            f.write(f"  Model Name: {result['left_model']['model_name']}\n")
            f.write(f"  GIF ID: {result['left_model']['gif_id']}\n")
            f.write(f"  Log ID: {result['left_model']['log_id']}\n")
            f.write(f"  Status: {result['left_model']['status']}\n\n")
            
            f.write("RIGHT MODEL:\n")
            f.write(f"  Model Name: {result['right_model']['model_name']}\n")
            f.write(f"  GIF ID: {result['right_model']['gif_id']}\n")
            f.write(f"  Log ID: {result['right_model']['log_id']}\n")
            f.write(f"  Status: {result['right_model']['status']}\n\n")
            
            f.write("=" * 80 + "\n\n")
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    
    vote_types = {}
    model_names = set()
    gif_found = 0
    log_found = 0
    
    for result in results:
        vote_type = result['vote_type']
        vote_types[vote_type] = vote_types.get(vote_type, 0) + 1
        
        model_names.add(result['left_model']['model_name'])
        model_names.add(result['right_model']['model_name'])
        
        if result['left_model']['gif_id'] != "NOT_FOUND":
            gif_found += 1
        if result['right_model']['gif_id'] != "NOT_FOUND":
            gif_found += 1
            
        if result['left_model']['log_id'] != "NOT_FOUND":
            log_found += 1
        if result['right_model']['log_id'] != "NOT_FOUND":
            log_found += 1
    
    print(f"Total interactions processed: {len(results)}")
    print(f"Vote type distribution: {vote_types}")
    print(f"Unique model names: {sorted(model_names)}")
    print(f"GIF IDs found: {gif_found} out of {len(results) * 2} total attempts")
    print(f"Log IDs found: {log_found} out of {len(results) * 2} total attempts")
    
    print(f"\nResults written to: {output_file}")

if __name__ == "__main__":
    main()