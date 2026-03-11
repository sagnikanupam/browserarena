#!/usr/bin/env python3

import json
import re

with open('FastChat/verified_conv_log/verified_interactions-clean.json', 'r') as f:
    data = json.load(f)
    
# Look for patterns in assistant messages
found_patterns = set()
example_messages = []

for idx, interaction in enumerate(data[:10]):  # Check first 10 interactions
    for state in interaction.get('states', []):
        messages = state.get('messages', [])
        for msg_type, msg_content in messages:
            if msg_type in ['Assistant', 'model']:
                # Look for common patterns
                if 'navigate_to_url' in msg_content:
                    found_patterns.add('navigate_to_url')
                if 'click' in msg_content.lower():
                    found_patterns.add('click actions')
                if 'screenshot' in msg_content.lower():
                    found_patterns.add('screenshot')
                if 'scroll' in msg_content.lower():
                    found_patterns.add('scroll')
                if 'type' in msg_content.lower() and 'input' in msg_content.lower():
                    found_patterns.add('type/input')
                if '```' in msg_content:
                    found_patterns.add('code blocks')
                    
                # Look for GIF and save a sample message
                if '<img src="/gradio_api/file=gifs/' in msg_content and len(example_messages) < 3:
                    example_messages.append((idx, msg_content[:2000]))

print('Patterns found:', found_patterns)
print('\n' + '='*50 + '\n')
print('Example assistant messages with GIFs:')
for idx, msg in example_messages:
    print(f'\nInteraction {idx+1}:')
    print(msg)
    print('...')