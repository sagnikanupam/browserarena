#!/usr/bin/env python3

import json
import re

def extract_agent_trace(messages):
    """Extract the agent's actions and observations from messages"""
    trace_lines = []
    
    for msg_type, msg_content in messages:
        if msg_type in ['Assistant', 'model']:
            # Extract actions
            action_matches = re.findall(r'```action\n(.*?)\n```', msg_content, re.DOTALL)
            for action in action_matches:
                trace_lines.append(f"ACTION: {action.strip()}")
            
            # Extract observations
            obs_matches = re.findall(r'OBSERVATION:(.*?)(?=ACTION:|$)', msg_content, re.DOTALL)
            for obs in obs_matches:
                obs_text = obs.strip()
                if obs_text:
                    # Truncate very long observations
                    if len(obs_text) > 200:
                        obs_text = obs_text[:200] + "..."
                    trace_lines.append(f"OBSERVATION: {obs_text}")
    
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
            
            assistant_messages = [msg[1] for msg in messages if msg[0] in ['Assistant', 'model']]
            
            for msg in assistant_messages:
                # Extract GIF ID
                gif_matches = re.findall(r'<img src="/gradio_api/file=gifs/([^"]+)" alt="GIF" />', msg)
                if gif_matches:
                    gif_id = gif_matches[0]
                
                # Extract Log ID
                log_match = re.search(r'Log ID: ([^"\n]+)', msg)
                if log_match:
                    log_id = log_match.group(1)
                
                # Determine success status
                if '✅ Task completed' in msg and '✅ Successfully' in msg:
                    success_status = "Success"
                elif '"success":true' in msg:
                    success_status = "Success"
                elif '❌ Stopping due to' in msg and 'consecutive failures' in msg:
                    success_status = "Failed"
                elif '❌ Result failed' in msg:
                    success_status = "Failed"
            
            success_emoji = "✅" if success_status == "Success" else "❌" if success_status == "Failed" else "❓"
            
            output_lines.append("")
            output_lines.append(f"📱 MODEL ATTEMPT {state_idx + 1}: {success_emoji} {success_status}")
            output_lines.append(f"  Model: {model_name}")
            output_lines.append(f"  GIF ID: {gif_id}")
            output_lines.append(f"  Log ID: {log_id}")
            output_lines.append("")
            output_lines.append("  Agent Trace:")
            output_lines.append("  " + "-" * 50)
            
            # Extract and add agent trace
            trace = extract_agent_trace(messages)
            if trace:
                for trace_line in trace[:20]:  # Limit to first 20 actions/observations
                    output_lines.append(f"  {trace_line}")
                if len(trace) > 20:
                    output_lines.append(f"  ... ({len(trace) - 20} more actions/observations)")
            else:
                output_lines.append("  No actions recorded")
            
            output_lines.append("")
        
        output_lines.append("")
    
    return "\n".join(output_lines)

if __name__ == "__main__":
    result = extract_survey_interactions_with_traces()
    
    with open('/Users/davisbrown/browserarena/verified-gifs-only/survey_interactions.txt', 'w') as f:
        f.write(result)
    
    print("✅ Updated survey_interactions.txt with agent traces!")
    print("📄 File: /Users/davisbrown/browserarena/verified-gifs-only/survey_interactions.txt")