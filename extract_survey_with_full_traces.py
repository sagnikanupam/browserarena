#!/usr/bin/env python3

import json
import re

def extract_agent_trace_from_message(msg_content):
    """Extract structured actions and key information from agent messages"""
    trace_lines = []
    
    # Extract task start
    task_match = re.search(r'🚀 Starting task: (.+?)(?:\n|$)', msg_content)
    if task_match:
        trace_lines.append(f"TASK: {task_match.group(1)}")
    
    # Extract steps and actions
    steps = re.findall(r'📍 Step (\d+)', msg_content)
    if steps:
        trace_lines.append(f"STEPS: Executed {len(steps)} steps")
    
    # Extract evaluation results
    eval_matches = re.findall(r'(?:👍|🤷|❌) Eval: (.+?)(?:\n|$)', msg_content)
    for eval_result in eval_matches:
        trace_lines.append(f"EVAL: {eval_result}")
    
    # Extract goals
    goal_matches = re.findall(r'🎯 Next goal: (.+?)(?:\n|$)', msg_content)
    for goal in goal_matches:
        trace_lines.append(f"GOAL: {goal}")
    
    # Extract actions
    action_matches = re.findall(r'🛠️  Action \d+/\d+: (.+?)(?:\n|$)', msg_content)
    for action in action_matches:
        try:
            # Parse JSON action if possible
            action_json = json.loads(action)
            action_type = list(action_json.keys())[0]
            trace_lines.append(f"ACTION: {action_type}")
        except:
            trace_lines.append(f"ACTION: {action[:100]}...")
    
    # Extract specific action types
    if '🔍  Searched for' in msg_content:
        search_matches = re.findall(r'🔍  Searched for "(.+?)" in', msg_content)
        for search in search_matches:
            trace_lines.append(f"SEARCH: {search}")
    
    if '🖱️  Clicked' in msg_content:
        click_matches = re.findall(r'🖱️  Clicked (.+?)(?:\n|$)', msg_content)
        for click in click_matches:
            trace_lines.append(f"CLICK: {click[:50]}...")
    
    if '📄  Extracted' in msg_content:
        trace_lines.append("EXTRACT: Content extracted from page")
    
    # Extract final results
    if '"success":true' in msg_content:
        trace_lines.append("RESULT: Success")
    elif '❌ Stopping due to' in msg_content:
        trace_lines.append("RESULT: Failed - stopping due to errors")
    elif 'Failed to parse model output' in msg_content:
        trace_lines.append("RESULT: Failed - parsing error")
    
    return trace_lines

def extract_survey_interactions_with_traces():
    # Read the verified interactions file
    with open('/Users/davisbrown/browserarena/FastChat/verified_conv_log/verified_interactions-clean.json', 'r') as f:
        data = json.load(f)
    
    # Read the survey GIF list to know which interactions to include
    with open('/Users/davisbrown/browserarena/survey_gifs_list.txt', 'r') as f:
        survey_gifs = [line.strip() for line in f if line.strip()]
    
    # Create a set of GIF IDs for quick lookup
    survey_gif_set = set(survey_gifs)
    
    output_lines = []
    output_lines.append("=== BROWSERARENA SURVEY INTERACTIONS WITH AGENT TRACES ===")
    output_lines.append("Total interactions: 25 (subset with all GIFs available)")
    output_lines.append("=" * 70)
    output_lines.append("")
    
    interaction_count = 0
    
    for i, interaction in enumerate(data, 1):
        # Check if this interaction's GIFs are in our survey set
        gif_found = False
        
        for state in interaction.get('states', []):
            messages = state.get('messages', [])
            for msg in messages:
                if msg[0] in ['Assistant', 'model']:
                    gif_matches = re.findall(r'<img src="/gradio_api/file=gifs/([^"]+)" alt="GIF" />', msg[1])
                    if gif_matches and gif_matches[0] in survey_gif_set:
                        gif_found = True
                        break
            if gif_found:
                break
        
        if not gif_found:
            continue
            
        interaction_count += 1
        if interaction_count > 25:
            break
            
        output_lines.append(f"{'='*70}")
        output_lines.append(f"INTERACTION {interaction_count}")
        output_lines.append(f"Vote Type: {interaction.get('type', 'N/A')}")
        
        # Extract the actual user task (second human message)
        user_task = "N/A"
        states = interaction.get('states', [])
        if states:
            messages = states[0].get('messages', [])
            human_messages = [msg[1] for msg in messages if msg[0] == 'Human']
            if len(human_messages) >= 2:
                user_task = human_messages[1]
            elif len(human_messages) == 1 and "birthday" not in human_messages[0].lower():
                user_task = human_messages[0]
        
        output_lines.append(f"User Task: {user_task}")
        output_lines.append("=" * 70)
        
        # Process each model attempt
        for state_idx, state in enumerate(states):
            messages = state.get('messages', [])
            model_name = state.get('model_name', 'N/A')
            
            # Extract GIF and success status
            gif_id = "N/A"
            log_id = "N/A"
            success_status = "Unknown"
            
            # Get all assistant messages
            assistant_messages = [(msg[0], msg[1]) for msg in messages if msg[0] in ['Assistant', 'model']]
            
            # Extract trace from all assistant messages
            all_traces = []
            
            for msg_type, msg_content in assistant_messages:
                # Extract GIF ID
                gif_matches = re.findall(r'<img src="/gradio_api/file=gifs/([^"]+)" alt="GIF" />', msg_content)
                if gif_matches:
                    gif_id = gif_matches[0]
                
                # Extract Log ID
                log_match = re.search(r'Log ID: ([^"\n]+)', msg_content)
                if log_match:
                    log_id = log_match.group(1)
                
                # Determine success status
                if '✅ Task completed' in msg_content and '✅ Successfully' in msg_content:
                    success_status = "Success"
                elif '"success":true' in msg_content:
                    success_status = "Success"
                elif '❌ Stopping due to' in msg_content:
                    success_status = "Failed"
                elif '❌ Result failed' in msg_content:
                    success_status = "Failed"
                elif 'Failed to parse model output' in msg_content:
                    success_status = "Failed"
                
                # Extract trace from this message
                trace = extract_agent_trace_from_message(msg_content)
                all_traces.extend(trace)
            
            success_emoji = "✅" if success_status == "Success" else "❌" if success_status == "Failed" else "❓"
            
            output_lines.append("")
            output_lines.append(f"📱 MODEL ATTEMPT {state_idx + 1}: {success_emoji} {success_status}")
            output_lines.append(f"  Model: {model_name}")
            output_lines.append(f"  GIF ID: {gif_id}")
            output_lines.append(f"  Log ID: {log_id}")
            output_lines.append("")
            output_lines.append("  Agent Trace:")
            output_lines.append("  " + "-" * 50)
            
            # Add trace lines
            if all_traces:
                # Limit to first 15 trace items to keep it readable
                for trace_line in all_traces[:15]:
                    output_lines.append(f"  {trace_line}")
                if len(all_traces) > 15:
                    output_lines.append(f"  ... ({len(all_traces) - 15} more trace items)")
            else:
                output_lines.append("  No trace data available")
            
            output_lines.append("")
        
        output_lines.append("")
    
    return "\n".join(output_lines)

if __name__ == "__main__":
    result = extract_survey_interactions_with_traces()
    
    with open('/Users/davisbrown/browserarena/verified-gifs-only/survey_interactions.txt', 'w') as f:
        f.write(result)
    
    print("✅ Updated survey_interactions.txt with full agent traces!")
    print("📄 File: /Users/davisbrown/browserarena/verified-gifs-only/survey_interactions.txt")