#!/usr/bin/env python3

import json
import re

def extract_verified_interactions():
    with open('/Users/davisbrown/browserarena/FastChat/verified_conv_log/verified_interactions-clean.json', 'r') as f:
        data = json.load(f)
    
    output_lines = []
    output_lines.append("=== VERIFIED BROWSERARENA INTERACTIONS ===")
    output_lines.append(f"Total interactions: {len(data)}")
    output_lines.append("=" * 50)
    output_lines.append("")
    
    for i, interaction in enumerate(data, 1):
        output_lines.append(f"INTERACTION {i}")
        output_lines.append(f"Vote Type: {interaction.get('type', 'N/A')}")
        
        # Extract the actual user task (should be the second human message)
        user_task = "N/A"
        state1_gifs = []
        state2_gifs = []
        state1_log_id = "N/A"
        state2_log_id = "N/A"
        
        # Process both states
        state1_model = "N/A"
        state2_model = "N/A"
        state1_success = "Unknown"
        state2_success = "Unknown"
        
        for state_idx, state in enumerate(interaction.get('states', [])):
            messages = state.get('messages', [])
            model_name = state.get('model_name', 'N/A')
            
            # Find the actual task (second human message)
            human_messages = [msg[1] for msg in messages if msg[0] == 'Human']
            if len(human_messages) >= 2:
                user_task = human_messages[1]  # The second human message is the actual task
            elif len(human_messages) == 1 and "birthday" not in human_messages[0].lower():
                user_task = human_messages[0]  # If only one message and it's not the birthday one
            
            # Extract GIFs from assistant messages and determine success
            assistant_messages = [msg[1] for msg in messages if msg[0] in ['Assistant', 'model']]
            gifs = []
            log_id = "N/A"
            success_status = "Unknown"
            
            for msg in assistant_messages:
                # Extract GIF IDs from HTML img tags
                gif_matches = re.findall(r'<img src="/gradio_api/file=gifs/([^"]+)" alt="GIF" />', msg)
                gifs.extend(gif_matches)
                
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
                elif '❌ Result failed' in msg and 'times:' in msg:
                    success_status = "Failed"
                elif 'Failed to parse model output' in msg:
                    success_status = "Failed"
            
            if state_idx == 0:
                state1_gifs = gifs
                state1_log_id = log_id
                state1_model = model_name
                state1_success = success_status
            else:
                state2_gifs = gifs
                state2_log_id = log_id
                state2_model = model_name
                state2_success = success_status
        
        output_lines.append(f"User Task: {user_task}")
        output_lines.append("")
        
        # State 1
        success_emoji1 = "✅" if state1_success == "Success" else "❌" if state1_success == "Failed" else "❓"
        output_lines.append(f"📱 MODEL ATTEMPT 1: {success_emoji1} {state1_success}")
        output_lines.append(f"  Model: {state1_model}")
        output_lines.append(f"  GIF ID: {state1_gifs[0] if state1_gifs else 'N/A'}")
        output_lines.append(f"  Log ID: {state1_log_id}")
        output_lines.append("")
        
        # State 2
        success_emoji2 = "✅" if state2_success == "Success" else "❌" if state2_success == "Failed" else "❓"
        output_lines.append(f"📱 MODEL ATTEMPT 2: {success_emoji2} {state2_success}")
        output_lines.append(f"  Model: {state2_model}")
        output_lines.append(f"  GIF ID: {state2_gifs[0] if state2_gifs else 'N/A'}")
        output_lines.append(f"  Log ID: {state2_log_id}")
        output_lines.append("")
        output_lines.append("-" * 80)
        output_lines.append("")
    
    return "\n".join(output_lines)

if __name__ == "__main__":
    result = extract_verified_interactions()
    
    with open('/Users/davisbrown/browserarena/extracted_verified_interactions_fixed.txt', 'w') as f:
        f.write(result)
    
    print("✅ Fixed extraction complete!")
    print("📄 File: /Users/davisbrown/browserarena/extracted_verified_interactions_fixed.txt")
