#!/usr/bin/env python3

import json
import re
from datetime import datetime

def extract_detailed_agent_trace(msg_content):
    """Extract detailed step-by-step agent trace including all actions and observations"""
    trace_lines = []
    
    # Extract task start
    task_match = re.search(r'🚀 Starting task: (.+?)(?:\n|$)', msg_content)
    if task_match:
        trace_lines.append(f"TASK: {task_match.group(1)}")
        trace_lines.append("")
    
    # Split content by steps to preserve order
    step_pattern = r'📍 Step (\d+)(.*?)(?=📍 Step \d+|$)'
    steps = re.findall(step_pattern, msg_content, re.DOTALL)
    
    for step_num, step_content in steps:
        trace_lines.append(f"📍 STEP {step_num}:")
        
        # Extract evaluation
        eval_match = re.search(r'(?:👍|🤷|❌) Eval: (.+?)(?:\n|$)', step_content)
        if eval_match:
            trace_lines.append(f"  Evaluation: {eval_match.group(1)}")
        
        # Extract memory/context
        memory_match = re.search(r'🧠 Memory: (.+?)(?:\n|$)', step_content)
        if memory_match:
            trace_lines.append(f"  Memory: {memory_match.group(1)}")
        
        # Extract goal
        goal_match = re.search(r'🎯 Next goal: (.+?)(?:\n|$)', step_content)
        if goal_match:
            trace_lines.append(f"  Goal: {goal_match.group(1)}")
        
        # Extract action details
        action_match = re.search(r'🛠️  Action \d+/\d+: (.+?)(?:\n|$)', step_content)
        if action_match:
            try:
                action_json = json.loads(action_match.group(1))
                action_type = list(action_json.keys())[0]
                action_params = action_json[action_type]
                trace_lines.append(f"  Action: {action_type}")
                if action_params:
                    for key, value in action_params.items():
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        trace_lines.append(f"    - {key}: {value}")
            except:
                trace_lines.append(f"  Action: {action_match.group(1)[:100]}...")
        
        # Extract specific action results
        if '🔍  Searched for' in step_content:
            search_match = re.search(r'🔍  Searched for "(.+?)" in', step_content)
            if search_match:
                trace_lines.append(f"  Result: Searched for \"{search_match.group(1)}\"")
        
        if '🖱️  Clicked' in step_content:
            click_match = re.search(r'🖱️  Clicked (.+?)(?:\n|$)', step_content)
            if click_match:
                click_text = click_match.group(1)[:100]
                trace_lines.append(f"  Result: Clicked {click_text}")
        
        if '📄  Extracted' in step_content:
            trace_lines.append("  Result: Extracted content from page")
        
        # Check for errors
        if 'Failed to parse model output' in step_content:
            trace_lines.append("  Error: Failed to parse model output")
        
        if '❌ Result failed' in step_content:
            fail_match = re.search(r'❌ Result failed (\d+/\d+) times', step_content)
            if fail_match:
                trace_lines.append(f"  Error: Result failed {fail_match.group(1)} times")
        
        trace_lines.append("")
    
    # Extract final result
    if '"success":true' in msg_content:
        # Extract the success text if available
        success_match = re.search(r'"text":"([^"]+)".*"success":true', msg_content)
        if success_match:
            text = success_match.group(1)[:200]
            trace_lines.append(f"✅ FINAL RESULT: Success - {text}...")
        else:
            trace_lines.append("✅ FINAL RESULT: Success")
    elif '❌ Stopping due to' in msg_content:
        stop_match = re.search(r'❌ Stopping due to (.+?)(?:\n|$)', msg_content)
        if stop_match:
            trace_lines.append(f"❌ FINAL RESULT: Failed - Stopping due to {stop_match.group(1)}")
        else:
            trace_lines.append("❌ FINAL RESULT: Failed")
    
    return trace_lines

def extract_survey_interactions_with_timestamp_and_traces():
    # Read the verified interactions file
    with open('/Users/davisbrown/browserarena/FastChat/verified_conv_log/verified_interactions-clean.json', 'r') as f:
        data = json.load(f)
    
    # Read the survey GIF list to know which interactions to include
    with open('/Users/davisbrown/browserarena/survey_gifs_list.txt', 'r') as f:
        survey_gifs = [line.strip() for line in f if line.strip()]
    
    # Create a set of GIF IDs for quick lookup
    survey_gif_set = set(survey_gifs)
    
    output_lines = []
    output_lines.append("=== BROWSERARENA SURVEY INTERACTIONS WITH DETAILED AGENT TRACES ===")
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
        
        # Add timestamp
        tstamp = interaction.get('tstamp', None)
        if tstamp:
            dt = datetime.fromtimestamp(tstamp)
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            output_lines.append(f"Date/Time: {date_str}")
        else:
            output_lines.append("Date/Time: N/A")
            
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
            
            # Process all assistant messages to build complete trace
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
                
                # Determine overall success status
                if '✅ Task completed' in msg_content and '✅ Successfully' in msg_content:
                    success_status = "Success"
                elif '"success":true' in msg_content:
                    success_status = "Success"
                elif '❌ Stopping due to' in msg_content:
                    success_status = "Failed"
                elif '❌ Result failed' in msg_content and 'consecutive failures' in msg_content:
                    success_status = "Failed"
                
                # Extract detailed trace from this message
                trace = extract_detailed_agent_trace(msg_content)
                if trace:
                    all_traces.extend(trace)
            
            success_emoji = "✅" if success_status == "Success" else "❌" if success_status == "Failed" else "❓"
            
            output_lines.append("")
            output_lines.append(f"📱 MODEL ATTEMPT {state_idx + 1}: {success_emoji} {success_status}")
            output_lines.append(f"  Model: {model_name}")
            output_lines.append(f"  GIF ID: {gif_id}")
            output_lines.append(f"  Log ID: {log_id}")
            output_lines.append("")
            output_lines.append("  Agent Trace:")
            output_lines.append("  " + "-" * 60)
            
            # Add detailed trace
            if all_traces:
                # Show all steps but limit very long traces
                max_lines = 100
                for i, trace_line in enumerate(all_traces[:max_lines]):
                    output_lines.append(f"  {trace_line}")
                if len(all_traces) > max_lines:
                    output_lines.append(f"  ... ({len(all_traces) - max_lines} more trace lines)")
            else:
                output_lines.append("  No detailed trace available")
            
            output_lines.append("")
        
        output_lines.append("")
    
    return "\n".join(output_lines)

if __name__ == "__main__":
    result = extract_survey_interactions_with_timestamp_and_traces()
    
    with open('/Users/davisbrown/browserarena/verified-gifs-only/survey_interactions.txt', 'w') as f:
        f.write(result)
    
    print("✅ Updated survey_interactions.txt with timestamps and detailed traces!")
    print("📄 File: /Users/davisbrown/browserarena/verified-gifs-only/survey_interactions.txt")