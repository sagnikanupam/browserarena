#!/usr/bin/env python3
import json
import re
from pathlib import Path

def clean_text(text):
    """Clean and escape text for HTML display."""
    # Remove ANSI color codes if any
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text)
    
    # Escape HTML characters
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&#39;')
    
    # Replace escaped newlines with actual newlines
    text = text.replace('\\n', '\n')
    
    # Convert newlines to <br> tags for proper HTML display
    text = text.replace('\n', '<br>')
    
    return text

def create_html_snippet(interaction_num, task, model1_output, model2_output):
    """Create a self-contained HTML snippet for a single interaction."""
    
    # Clean the outputs
    model1_output_clean = clean_text(model1_output)
    model2_output_clean = clean_text(model2_output)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            line-height: 1.5;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        
        .task-header {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .task-header h3 {{
            color: #1a73e8;
            font-size: 18px;
            margin-bottom: 8px;
        }}
        
        .task-text {{
            color: #555;
            font-size: 16px;
            line-height: 1.6;
        }}
        
        .comparison-container {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .agent-output {{
            flex: 1;
            min-width: 300px;
            border: 1px solid #ddd;
            border-radius: 6px;
            overflow: hidden;
        }}
        
        .agent-header {{
            background: #f8f9fa;
            padding: 12px 16px;
            border-bottom: 1px solid #e0e0e0;
            font-weight: 600;
            color: #333;
        }}
        
        .output-content {{
            padding: 16px;
            background: #fafafa;
            max-height: 600px;
            overflow-y: auto;
        }}
        
        .output-text {{
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            word-wrap: break-word;
            color: #333;
            line-height: 1.8;
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            
            .container {{
                padding: 15px;
            }}
            
            .comparison-container {{
                flex-direction: column;
                gap: 15px;
            }}
            
            .agent-output {{
                min-width: 100%;
            }}
            
            .task-header h3 {{
                font-size: 16px;
            }}
            
            .task-text {{
                font-size: 14px;
            }}
            
            .output-text {{
                font-size: 12px;
            }}
            
            .output-content {{
                max-height: 400px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="task-header">
            <h3>Interaction {interaction_num}</h3>
            <div class="task-text"><strong>Task:</strong> {task}</div>
        </div>
        
        <div class="comparison-container">
            <div class="agent-output">
                <div class="agent-header">Agent 1</div>
                <div class="output-content">
                    <div class="output-text">{model1_output_clean}</div>
                </div>
            </div>
            
            <div class="agent-output">
                <div class="agent-header">Agent 2</div>
                <div class="output-content">
                    <div class="output-text">{model2_output_clean}</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>'''
    
    return html

def process_interactions():
    """Process all interactions and generate HTML snippets."""
    
    # Read the survey interactions file
    with open('survey_interactions.txt', 'r') as f:
        content = f.read()
    
    # Parse interactions
    interactions = []
    current_interaction = None
    
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        if line.strip().startswith('INTERACTION'):
            if current_interaction:
                interactions.append(current_interaction)
            
            interaction_num = int(line.strip().split()[1])
            current_interaction = {
                'number': interaction_num,
                'vote_type': '',
                'task': '',
                'model1_log_id': '',
                'model2_log_id': ''
            }
        
        elif line.strip().startswith('Vote Type:'):
            current_interaction['vote_type'] = line.strip().split(':', 1)[1].strip()
        
        elif line.strip().startswith('User Task:'):
            current_interaction['task'] = line.strip().split(':', 1)[1].strip()
        
        elif 'Log ID:' in line and line.startswith('  '):
            log_id = line.strip().split(':', 1)[1].strip()
            if not current_interaction['model1_log_id']:
                current_interaction['model1_log_id'] = log_id
            else:
                current_interaction['model2_log_id'] = log_id
        
        i += 1
    
    if current_interaction:
        interactions.append(current_interaction)
    
    # Create output directory
    output_dir = Path('survey_html_snippets')
    output_dir.mkdir(exist_ok=True)
    
    # Process each interaction
    for interaction in interactions:
        print(f"Processing interaction {interaction['number']}...")
        
        # Read model outputs from JSON files
        try:
            # Model 1
            json_path1 = f"FastChat/prompts_and_outputs/{interaction['model1_log_id']}.json"
            with open(json_path1, 'r') as f:
                data1 = json.load(f)
                model1_output = data1.get('output', 'No output available')
            
            # Model 2
            json_path2 = f"FastChat/prompts_and_outputs/{interaction['model2_log_id']}.json"
            with open(json_path2, 'r') as f:
                data2 = json.load(f)
                model2_output = data2.get('output', 'No output available')
            
            # Generate HTML snippet
            html = create_html_snippet(
                interaction['number'],
                interaction['task'],
                model1_output,
                model2_output
            )
            
            # Save HTML file
            output_path = output_dir / f"interaction_{interaction['number']:02d}.html"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            print(f"  Created: {output_path}")
            
        except Exception as e:
            print(f"  Error processing interaction {interaction['number']}: {e}")
    
    print(f"\nAll HTML snippets saved to: {output_dir}")

if __name__ == "__main__":
    process_interactions()