#!/usr/bin/env python3
import os
import re

# Directory containing the HTML files
html_dir = 'survey_html_snippets'

# Get all HTML files
html_files = [f for f in os.listdir(html_dir) if f.endswith('.html')]

# Process each file
for filename in sorted(html_files):
    file_path = os.path.join(html_dir, filename)
    
    print(f"Processing {filename}...")
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove the h3 tag containing "Interaction N" completely
    # Pattern matches: <h3>Interaction [number]</h3>
    pattern = r'<h3>Interaction \d+</h3>\n\s*'
    
    # Replace the pattern with empty string
    updated_content = re.sub(pattern, '', content)
    
    # Write the updated content back
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"  ✓ Removed 'Interaction N' title from {filename}")

print("\nAll files have been updated successfully!")