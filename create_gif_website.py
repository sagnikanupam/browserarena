#!/usr/bin/env python3
"""
Create a GitHub Pages website to display BrowserArena GIFs
"""
import os
import json
from pathlib import Path

def create_gif_website():
    """Create HTML pages for all GIFs with metadata"""
    
    # Read the verified interactions data
    base_dir = Path("/Users/davisbrown/browserarena")
    gifs_dir = base_dir / "FastChat" / "gifs"
    
    # Create website directory
    site_dir = base_dir / "gif-website"
    site_dir.mkdir(exist_ok=True)
    
    # Get all GIF files
    gif_files = list(gifs_dir.glob("*.gif"))
    
    # Create index.html
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BrowserArena Interaction GIFs</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .gif-container {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
        .gif-id {{ font-weight: bold; color: #333; }}
        .gif-image {{ max-width: 100%; height: auto; }}
        .metadata {{ background: #f5f5f5; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>BrowserArena Interaction GIFs</h1>
    <p>Total GIFs: {len(gif_files)}</p>
    <p>These GIFs show browser automation attempts from the BrowserArena dataset.</p>
    
"""
    
    # Add each GIF
    for gif_file in sorted(gif_files):
        gif_name = gif_file.name
        gif_id = gif_name.replace('.gif', '')
        
        html_content += f"""
    <div class="gif-container">
        <div class="gif-id">GIF ID: {gif_id}</div>
        <div class="metadata">
            <strong>File:</strong> {gif_name}<br>
            <strong>Size:</strong> {gif_file.stat().st_size / 1024:.1f} KB
        </div>
        <img src="gifs/{gif_name}" alt="{gif_id}" class="gif-image" loading="lazy">
    </div>
        """
    
    html_content += """
</body>
</html>"""
    
    # Write index.html
    with open(site_dir / "index.html", "w") as f:
        f.write(html_content)
    
    # Create gifs subdirectory and copy files
    site_gifs_dir = site_dir / "gifs"
    site_gifs_dir.mkdir(exist_ok=True)
    
    print(f"Created website structure at {site_dir}")
    print(f"Copy GIFs with: cp /Users/davisbrown/browserarena/FastChat/gifs/*.gif {site_gifs_dir}/")
    print(f"Total files to copy: {len(gif_files)}")
    
    # Create GitHub Pages deployment instructions
    instructions = """
# GitHub Pages Deployment Instructions

1. Create a new GitHub repository called 'browserarena-gifs'
2. Copy the website files:
   cp -r {site_dir}/* /path/to/your/repo/
3. Add and commit files:
   git add .
   git commit -m "Add BrowserArena GIF viewer"
   git push origin main
4. Enable GitHub Pages in repository settings
5. Your GIFs will be available at: https://yourusername.github.io/browserarena-gifs/

Note: GitHub has a 1GB repository limit. You may need to split into multiple repos.
""".format(site_dir=site_dir)
    
    with open(site_dir / "deployment-instructions.md", "w") as f:
        f.write(instructions)
    
    return site_dir

if __name__ == "__main__":
    site_dir = create_gif_website()
    print(f"Website created at: {site_dir}")