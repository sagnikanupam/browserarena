import os
import re
from pathlib import Path

def truncate_middle(text, max_length=1800):
    """Truncate text in the middle to keep it under max_length characters."""
    if len(text) <= max_length:
        return text
    
    # Calculate how much to keep from start and end
    keep_chars = max_length - 50  # Reserve 50 chars for truncation message
    start_chars = keep_chars // 2
    end_chars = keep_chars - start_chars
    
    # Create truncated text
    truncated = text[:start_chars] + "<br><br>[... truncated for brevity ...]<br><br>" + text[-end_chars:]
    return truncated

def process_html_file(input_path, output_path):
    """Process a single HTML file to truncate agent output text."""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all output-text divs
    pattern = r'(<div class="output-text">)(.*?)(</div>)'
    
    def replacer(match):
        opening_tag = match.group(1)
        content = match.group(2)
        closing_tag = match.group(3)
        
        # Truncate the content
        truncated_content = truncate_middle(content, 1800)
        
        return opening_tag + truncated_content + closing_tag
    
    # Replace all output-text content
    new_content = re.sub(pattern, replacer, content, flags=re.DOTALL)
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

def main():
    input_dir = Path("survey_html_snippets")
    output_dir = Path("survey_html_snippets_truncated")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Process all HTML files
    html_files = sorted(input_dir.glob("interaction_*.html"))
    
    print(f"Processing {len(html_files)} HTML files...")
    
    for html_file in html_files:
        output_file = output_dir / html_file.name
        process_html_file(html_file, output_file)
        print(f"Processed: {html_file.name}")
    
    print(f"\nCompleted! Truncated files saved to: {output_dir}")

if __name__ == "__main__":
    main()